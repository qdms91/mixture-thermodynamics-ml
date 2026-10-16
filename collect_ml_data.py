"""Collect thermodynamics and stability labels in separate ML campaign folders.

Run beside mixture_thermodynamics.py, phase_stability.py and project_config.py.
Example: python collect_ml_data.py --campaign stability --workers 3 --limit 6
"""

import os

# Each worker uses one numerical thread. Three workers leave one laptop core free.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import redirect_stdout
import fcntl
import hashlib
from importlib.metadata import PackageNotFoundError, version
from itertools import combinations
import json
import multiprocessing
from pathlib import Path
import signal
import sqlite3
from time import perf_counter

import numpy as np
import pandas as pd

import mixture_thermodynamics as thermo
import phase_stability as stability
import project_config as config


TEMPERATURE = 298.15
MODEL = None


def grid_recipes(step):
    """Distribute the remaining 25 mol% after giving each solvent 15 mol%."""
    units = 25 // step
    for cuts in combinations(range(units + 4), 4):
        increments = np.diff((-1, *cuts, units + 4)) - 1
        yield tuple(int(value) for value in 15 + step * increments)


def make_manifest(campaign, seed):
    """Choose compositions and reserve composition blocks before seeing labels."""
    rng = np.random.default_rng(seed)
    fine = list(grid_recipes(1))
    coarse = list(grid_recipes(5))
    coarse_set = set(coarse)
    anchors = [(20,) * 5]
    anchors += [tuple(40 if i == j else 15 for i in range(5)) for j in range(5)]

    if campaign == "binary":
        # Trace concentrations give us more informative endpoint tests than a
        # binary grid restricted to 1-99%. The other three solvents are absent.
        trace = [1e-6, 1e-5, 1e-4, 0.001, 0.005]
        fractions = sorted(trace + [i / 100 for i in range(1, 100)]
                           + [1 - value for value in trace])
        compositions = [(v, 0.0, 0.0, 0.0, 1 - v) for v in fractions]
    else:
        remaining = [p for p in fine if p not in coarse_set]
        rng.shuffle(remaining)
        if campaign == "stability":
            percentages = anchors + [p for p in coarse if p not in anchors]
            percentages += remaining[:474]  # 126 coarse + 474 fine = 600 recipes.
        else:
            percentages = anchors + [p for p in fine if p not in anchors]
        compositions = [tuple(v / 100 for v in p) for p in percentages]

    rows = []
    for index, x in enumerate(compositions):
        recipe_id = "x_" + "_".join(f"{v:.8g}" for v in x)
        if campaign == "binary":
            block, split, sampling = "binary", "external_challenge", "binary"
        else:
            percentages = tuple(int(round(100 * v)) for v in x)
            # All recipes in the same 5-mol%-wide box of the first four
            # fractions stay together. The fifth fraction follows from the sum.
            block = "_".join(str((p - 15) // 5) for p in percentages[:4])
            digest = hashlib.sha256(f"{seed}:{block}".encode()).hexdigest()
            bucket = int(digest[:8], 16) % 100
            split = "train_pool" if bucket < 70 else ("validation" if bucket < 85 else "test")
            sampling = "coarse_grid" if percentages in coarse_set else "fine_grid"
        rows.append({"recipe_id": recipe_id, "recipe_number": index + 1,
                     "block_id": block, "split": split, "sampling_group": sampling,
                     **{f"x_{name}": value for name, value in zip(config.MOLECULES, x)}})
    return rows


def initialize_worker():
    """Build one independent COSMO-RS model per process and reuse it."""
    global MODEL
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    MODEL = thermo.build_model("calculated")
    MODEL.par.cosmospace_conv_thresh = 1e-10
    MODEL.par.cosmospace_max_iter = 3000


def evaluate_recipe(recipe, campaign, starts, seed, directory):
    """Save feed properties first; search results are separate label columns."""
    started = perf_counter()
    x = np.array([recipe[f"x_{name}"] for name in config.MOLECULES])
    x /= x.sum()
    table, properties = thermo.mixture_thermodynamics(MODEL, x, TEMPERATURE)
    _, wider = thermo.mixture_thermodynamics(MODEL, x, TEMPERATURE, dT=2.0)
    properties["h_step_change_kJ_mol"] = abs(wider["h_mix_kJ_mol"] - properties["h_mix_kJ_mol"])
    row = {**recipe, "temperature_K": TEMPERATURE, **properties}
    indexed = table.set_index("molecule")
    for name in config.MOLECULES:
        for column in ("ln_gamma", "gamma", "activity"):
            row[f"{column}_{name}"] = (
                float(indexed.loc[name, column]) if name in indexed.index else None
            )
    row["feature_seconds"] = perf_counter() - started
    search_started = perf_counter()

    if campaign == "thermo":
        row["label_status"] = "not_tested"
    else:
        present = x > 0
        z = x[present]

        def log_activity(w):
            # In a binary feed, absent solvents must also remain absent from
            # every trial phase. We optimize only the two present fractions.
            full_x = np.zeros(len(config.MOLECULES))
            full_x[present] = w
            data, _ = thermo.mixture_properties(MODEL, full_x, TEMPERATURE)
            return np.log(w) + data["ln_gamma"].to_numpy()

        search_seed = int(np.random.SeedSequence([seed, recipe["recipe_number"]])
                          .generate_state(1)[0])
        # The imported function prints each start. Keep its complete numerical
        # results in a CSV and print just one progress line per overall recipe.
        with open(os.devnull, "w") as quiet, redirect_stdout(quiet):
            best, best_w, trials = stability.search_tpd(log_activity, z, starts, search_seed)
        failed = int((~trials["converged"]).sum())
        split = stability.check_small_split(log_activity, z, best_w) if best < -stability.TPD_TOLERANCE else None
        if best < -stability.TPD_TOLERANCE and split is not None:
            label, status = 1, "instability_found"
        elif best < -stability.TPD_TOLERANCE or failed:
            label, status = None, "needs_review"
        else:
            label, status = 0, "no_instability_found"
        row.update({"instability_label": label, "label_status": status,
                    "lowest_tpd_over_RT": float(best), "starts": starts,
                    "search_seed": search_seed, "nonconverged_searches": failed})
        full_w = np.zeros(len(config.MOLECULES))
        full_w[present] = best_w
        row.update({f"best_trial_x_{name}": float(v) for name, v in zip(config.MOLECULES, full_w)})
        row["trial_fraction"] = float(split[0]) if split is not None else None
        row["split_change_kJ_mol"] = float(split[2]) if split is not None else None
        if split is not None:
            remainder = np.zeros(len(config.MOLECULES))
            remainder[present] = split[1]
            row.update({f"remaining_x_{name}": float(v) for name, v in zip(config.MOLECULES, remainder)})
        trial_file = Path(directory) / "searches" / f"{recipe['recipe_id']}.csv"
        temporary = trial_file.with_suffix(".tmp")
        trials.to_csv(temporary, index=False)
        temporary.replace(trial_file)

    row["stability_seconds"] = perf_counter() - search_started if campaign != "thermo" else 0.0
    row["elapsed_seconds"] = perf_counter() - started
    return row


def provenance(campaign, starts, seed):
    """Prevent a resumed campaign from silently mixing different calculations."""
    files = {f"surface:{name}": path for name, path in config.surface_paths("calculated").items()}
    for module in (thermo, stability, config):
        files[Path(module.__file__).name] = Path(module.__file__)
    files[Path(__file__).name] = Path(__file__)
    versions = {}
    for package in ("numpy", "scipy", "pandas", "opencosmorspy"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "metadata_unavailable"
    return {"campaign": campaign, "seed": seed, "starts": starts if campaign != "thermo" else None,
            "temperature_K": TEMPERATURE, "molecule_order": list(config.MOLECULES),
            "source": "calculated", "versions": versions, "temperature_steps_K": [1.0, 2.0],
            "cosmospace_convergence": 1e-10, "cosmospace_max_iter": 3000,
            "tpd_tolerance": stability.TPD_TOLERANCE, "mole_fraction_floor": stability.FLOOR,
            "sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in files.items()}}


def export_dataset(database, directory):
    """Write an ordinary CSV from the safely saved completed recipes."""
    rows = [json.loads(payload) for (payload,) in database.execute("SELECT payload FROM results")]
    if rows:
        frame = pd.DataFrame(rows).sort_values("recipe_number")
        temporary = directory / "dataset.tmp"
        frame.to_csv(temporary, index=False)
        temporary.replace(directory / "dataset.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", choices=("thermo", "stability", "binary"), required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=3)
    parser.add_argument("--starts", type=int, default=40)
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--limit", type=int, help="Only the first N manifest recipes, including already completed ones")
    parser.add_argument("--describe", action="store_true", help="Print recipe counts without running calculations")
    args = parser.parse_args()
    if args.seed < 0 or args.starts < 16 or (args.limit is not None and args.limit < 1):
        parser.error("Require seed >= 0, starts >= 16, and a positive limit.")
    if tuple(config.MOLECULES) != ("water", "methanol", "acetone", "acetonitrile", "benzene"):
        parser.error("This campaign requires the established five-solvent order.")
    if stability.TEMPERATURE != TEMPERATURE:
        parser.error("The imported stability calculation must also use 298.15 K.")
    recipes = make_manifest(args.campaign, args.seed)
    print(f"Campaign: {args.campaign}; total recipes: {len(recipes):,}")
    print("Molecule order:", ", ".join(config.MOLECULES))
    print(pd.Series([row["split"] for row in recipes]).value_counts().to_string())
    if args.describe:
        return

    tag = f"{args.campaign}_seed{args.seed}"
    if args.campaign != "thermo":
        tag += f"_starts{args.starts}"
    directory = config.PROJECT / "results/ml" / tag
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "run.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("This campaign is already running. Use its existing terminal.")
        metadata = provenance(args.campaign, args.starts, args.seed)
        metadata_file = directory / "metadata.json"
        if metadata_file.exists() and json.loads(metadata_file.read_text()) != metadata:
            parser.error("Campaign inputs or code changed. Keep this folder intact and inspect metadata.json before resuming.")
        if not metadata_file.exists():
            metadata_file.write_text(json.dumps(metadata, indent=2) + "\n")
        pd.DataFrame(recipes).to_csv(directory / "recipes.csv", index=False)
        (directory / "searches").mkdir(exist_ok=True)

        with sqlite3.connect(directory / "checkpoint.sqlite") as database:
            database.execute("CREATE TABLE IF NOT EXISTS results (recipe_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            completed = {r[0] for r in database.execute("SELECT recipe_id FROM results")}
            selected = recipes[:args.limit] if args.limit is not None else recipes
            pending = [r for r in selected if r["recipe_id"] not in completed]
            print(f"Selected total: {len(selected)}; already completed: {len(selected) - len(pending)}; remaining: {len(pending)}", flush=True)
            print("Output:", directory, flush=True)
            started, finished = perf_counter(), 0
            pool = ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                                       mp_context=multiprocessing.get_context("spawn")) if pending else None
            futures = {}
            todo = iter(pending)

            def submit_next():
                recipe = next(todo, None)
                if recipe is not None:
                    future = pool.submit(evaluate_recipe, recipe, args.campaign, args.starts, args.seed, str(directory))
                    futures[future] = recipe

            try:
                for _ in range(min(args.workers, len(pending))):
                    submit_next()
                while futures:
                    ready, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in ready:
                        recipe = futures.pop(future)
                        row = future.result()
                        database.execute("INSERT INTO results VALUES (?, ?)",
                                         (recipe["recipe_id"], json.dumps(row, allow_nan=False)))
                        database.commit()  # Each completed recipe survives a later interruption.
                        finished += 1
                        export_every = 100 if args.campaign == "thermo" else 1
                        if finished % export_every == 0 or finished == 1:
                            export_dataset(database, directory)
                            remaining_h = (perf_counter() - started) / finished * (len(pending) - finished) / 3600
                            print(f"Completed this run {finished}/{len(pending)}: {recipe['recipe_id']}; "
                                  f"{row['label_status']}; estimated remaining {remaining_h:.2f} h", flush=True)
                        submit_next()
            except KeyboardInterrupt:
                print("\nStopping; waiting for current worker jobs to finish. Saved recipes will be reused.", flush=True)
            finally:
                for future in futures:
                    future.cancel()
                if pool is not None:
                    pool.shutdown(wait=True, cancel_futures=True)
                export_dataset(database, directory)
            print("Saved:", directory / "dataset.csv")


if __name__ == "__main__":
    main()
