"""Screen five-solvent recipes with at least 15 mol% of every solvent.

Run beside phase_stability.py, mixture_thermodynamics.py and project_config.py.
Examples:
    python phase_stability_grid.py --count-only
    python phase_stability_grid.py --limit 10
    python phase_stability_grid.py --step 5
    python phase_stability_grid.py --step 1
Repeating a command resumes completed recipes. --limit counts NEW recipes.
"""

import argparse
from contextlib import redirect_stdout
import fcntl
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import sqlite3
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.constants import R

import phase_stability as stability
import mixture_thermodynamics as thermo
import project_config as config


def recipe_grid(step=1, seed=88):
    """Distribute the remaining 25 percentage points after giving each 15%."""
    recipes = []
    for first_four in product(range(0, 26, step), repeat=4):
        fifth = 25 - sum(first_four)
        if 0 <= fifth <= 25 and fifth % step == 0:
            recipes.append(tuple(15 + n for n in (*first_four, fifth)))

    # Start with the equimolar recipe and the five component-rich corners.
    # Randomize the remaining order so a short run covers varied recipes.
    priority = [(20,) * 5]
    priority += [tuple(40 if i == j else 15 for i in range(5))
                 for j in range(5)]
    remaining = [x for x in recipes if x not in priority]
    np.random.default_rng(seed).shuffle(remaining)
    return priority + remaining


def export_csv(database, directory):
    """Export completed recipes; the database also retains every search row."""
    rows = []
    for (payload,) in database.execute("SELECT payload FROM recipes ORDER BY id"):
        row = json.loads(payload)
        row.pop("searches")
        rows.append(row)
    if rows:
        temporary = directory / "grid_summary.tmp.csv"
        pd.DataFrame(rows).to_csv(temporary, index=False)
        temporary.replace(directory / "grid_summary.csv")


def screen_recipe(model, percentages, starts, seed):
    """Test one overall recipe against freely varying trial-phase compositions."""
    z = np.asarray(percentages, dtype=float) / 100
    z /= z.sum()
    table, energies = thermo.mixture_properties(model, z, stability.TEMPERATURE)

    def log_activity(x):
        activities, _ = thermo.mixture_properties(model, x, stability.TEMPERATURE)
        return np.log(x) + activities["ln_gamma"].to_numpy()

    # Give count and seed explicitly: changes to the imported function's
    # defaults cannot accidentally request 4,000 searches for every recipe.
    # Keep the individual search records, but print just one line per recipe.
    with open(os.devnull, "w") as quiet, redirect_stdout(quiet):
        value, trial, searches = stability.search_tpd(
            log_activity, z, count=starts, seed=seed
        )
    failures = int((~searches["converged"]).sum())
    if value < -stability.TPD_TOLERANCE:
        status = "instability_found"
    elif failures:
        status = "no_instability_found_some_searches_failed"
    else:
        status = "no_instability_found"

    result = {
        "temperature_K": stability.TEMPERATURE,
        "starts": starts, "seed": seed, "status": status,
        "lowest_tpd_over_RT": value,
        "lowest_tpd_kJ_mol_trial": value * R * stability.TEMPERATURE / 1000,
        "nonconverged_searches": failures,
        **energies,
        "split_verified": None,
        "trial_phase_fraction": None,
        "split_change_kJ_mol": None,
    }
    for i, name in enumerate(config.MOLECULES):
        result[f"x_{name}"] = float(z[i])
        result[f"gamma_{name}"] = float(table["gamma"].iloc[i])
        result[f"activity_{name}"] = float(table["activity"].iloc[i])
        result[f"best_trial_x_{name}"] = float(trial[i])
        result[f"remaining_x_{name}"] = None

    # For a negative TPD, also check an actual mass-balanced trial split.
    # These compositions are a demonstration, not final equilibrium layers.
    if status == "instability_found":
        split = stability.check_small_split(log_activity, z, trial)
        result["split_verified"] = split is not None
        if split is not None:
            fraction, remaining, change = split
            result.update(trial_phase_fraction=float(fraction),
                          split_change_kJ_mol=float(change))
            for i, name in enumerate(config.MOLECULES):
                result[f"remaining_x_{name}"] = float(remaining[i])
    result["searches"] = searches.to_dict(orient="records")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, choices=(1, 5), default=1,
                        help="Grid spacing in mol percentage points")
    parser.add_argument("--starts", type=int, default=40)
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--limit", type=int,
                        help="Maximum NEW recipes to finish during this run")
    parser.add_argument("--count-only", action="store_true")
    args = parser.parse_args()
    if args.starts < 16 or args.seed < 0 or (args.limit is not None and args.limit < 1):
        parser.error("Require starts >= 16, seed >= 0 and limit >= 1.")

    recipes = recipe_grid(args.step, args.seed)
    print("Molecule order:", ", ".join(config.MOLECULES))
    print(f"Grid: {len(recipes):,} recipes; {args.starts} searches per recipe.")
    if args.count_only:
        return

    directory = config.PROJECT / "results/calculated/phase_stability_grid" / (
        f"min15_step{args.step}_starts{args.starts}_seed{args.seed}"
    )
    directory.mkdir(parents=True, exist_ok=True)

    # The separate directory cannot overwrite the equimolar search results.
    # A lock prevents two processes writing this same grid run simultaneously.
    with (directory / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("This grid run is already active in another terminal.")

        # Refuse to mix old checkpoints with changed surfaces or scripts.
        paths = {**config.surface_paths("calculated"),
                 "grid_script": Path(__file__),
                 "stability_script": Path(stability.__file__),
                 "thermodynamics_script": Path(thermo.__file__),
                 "configuration_script": Path(config.__file__)}
        metadata = {
            "temperature_K": stability.TEMPERATURE,
            "molecules": list(config.MOLECULES), "minimum_mol_percent": 15,
            "step": args.step, "starts": args.starts, "seed": args.seed,
            "tpd_tolerance": stability.TPD_TOLERANCE,
            "surface_and_script_sha256": {
                name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in paths.items()
            },
        }
        database = sqlite3.connect(directory / "checkpoint.sqlite")
        try:
            database.execute("CREATE TABLE IF NOT EXISTS metadata (payload TEXT)")
            database.execute(
                "CREATE TABLE IF NOT EXISTS recipes (id TEXT PRIMARY KEY, payload TEXT)"
            )
            previous = database.execute("SELECT payload FROM metadata").fetchone()
            if previous and json.loads(previous[0]) != metadata:
                raise RuntimeError("Settings, scripts or surfaces changed. Use a new --seed "
                                   "to start a separate run, or restore the original files.")
            if previous is None:
                database.execute("INSERT INTO metadata VALUES (?)", (json.dumps(metadata),))
                database.commit()
            completed = {row[0] for row in database.execute("SELECT id FROM recipes")}
            print(f"Previously completed: {len(completed):,}. Saved in: {directory}")
            model = thermo.build_model("calculated")
            model.par.cosmospace_conv_thresh = 1e-10
            model.par.cosmospace_max_iter = 3000
            finished = 0
            began = perf_counter()
            try:
                for index, percentages in enumerate(recipes):
                    key = "_".join(map(str, percentages))
                    if key in completed:
                        continue
                    # Each recipe gets reproducible random starts of its own.
                    seed = int(np.random.SeedSequence([args.seed, index]).generate_state(1)[0])
                    start_time = perf_counter()
                    row = screen_recipe(model, percentages, args.starts, seed)
                    row["seconds"] = perf_counter() - start_time
                    row["recipe_id"] = key
                    database.execute("INSERT INTO recipes VALUES (?, ?)",
                                     (key, json.dumps(row, allow_nan=False)))
                    database.commit()  # Checkpoint after EVERY completed recipe.
                    completed.add(key)
                    finished += 1
                    mean_seconds = (perf_counter() - began) / finished
                    remaining_hours = (len(recipes) - len(completed)) * mean_seconds / 3600
                    print(f"{len(completed):5d}/{len(recipes)}: {percentages}; "
                          f"{row['status']}; TPD/RT={row['lowest_tpd_over_RT']:+.8f}; "
                          f"estimated full-grid remainder {remaining_hours:.1f} h", flush=True)
                    if finished % 100 == 0:
                        export_csv(database, directory)
                    if args.limit is not None and finished >= args.limit:
                        break
            except KeyboardInterrupt:
                print("\nStopped. Completed recipes are saved; repeat the command to resume.")
            finally:
                export_csv(database, directory)
            print("Summary:", directory / "grid_summary.csv")
            print("No instability found means no negative TPD was found in this finite search.")
        finally:
            database.close()


if __name__ == "__main__":
    main()
