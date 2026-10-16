"""Rank recipes by homogeneous mixing free energy before testing stability.

Run beside the existing project scripts. Each solvent is at least 15 mol%.
Example: python screen_mixing_free_energy.py --step 1 --workers 4
"""

import os

# Each worker uses one numerical thread, so four workers share four CPU cores.
# These settings must be applied before importing NumPy through our modules.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import multiprocessing
from pathlib import Path
import sqlite3
from time import perf_counter

import numpy as np
import pandas as pd

import mixture_thermodynamics as thermo
import phase_stability_grid as grid
import project_config as config


TEMPERATURE = 298.15
MODEL = None


def package_version(name):
    """Record installed versions; source-only checkouts may lack metadata."""
    try:
        return version(name)
    except PackageNotFoundError:
        return "metadata_unavailable"


def initialize_worker():
    """Each process builds its own model once and reuses it for many recipes."""
    global MODEL
    MODEL = thermo.build_model("calculated")
    MODEL.par.cosmospace_conv_thresh = 1e-10
    MODEL.par.cosmospace_max_iter = 3000


def evaluate_recipe(percentages):
    """Calculate one homogeneous liquid; no TPD optimization is performed."""
    x = np.asarray(percentages, dtype=float) / 100
    x /= x.sum()
    _, properties = thermo.mixture_properties(MODEL, x, TEMPERATURE)
    return {
        "recipe_id": "_".join(map(str, percentages)),
        "temperature_K": TEMPERATURE,
        **{f"x_{name}": float(value) for name, value in zip(config.MOLECULES, x)},
        **properties,
        "stability_status": "not_tested_in_this_screen",
    }


def export_ranking(database, directory):
    """Save all completed recipes in order of increasing mixing free energy."""
    rows = [json.loads(payload) for (payload,) in
            database.execute("SELECT payload FROM recipes")]
    if not rows:
        return
    ranked = pd.DataFrame(rows).sort_values(["g_mix_kJ_mol", "recipe_id"])
    baseline = ranked.loc[ranked.recipe_id == "20_20_20_20_20", "g_mix_kJ_mol"]
    if len(baseline):
        ranked["change_g_mix_vs_equimolar_kJ_mol"] = ranked.g_mix_kJ_mol - baseline.iloc[0]
    temporary = directory / "free_energy_ranking.tmp.csv"
    ranked.to_csv(temporary, index=False)
    temporary.replace(directory / "free_energy_ranking.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, choices=(1, 5), default=1)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--limit", type=int, help="Maximum additional recipes this run")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1.")

    # Use the same grid as the stability script, including the equimolar point.
    recipes = grid.recipe_grid(args.step, seed=88)
    directory = config.PROJECT / "results/calculated/free_energy_screen" / f"step{args.step}"
    directory.mkdir(parents=True, exist_ok=True)
    files = {**config.surface_paths("calculated"),
             "screen_script": Path(__file__), "grid_script": Path(grid.__file__),
             "thermodynamics_script": Path(thermo.__file__),
             "configuration_script": Path(config.__file__)}
    metadata = {
        "temperature_K": TEMPERATURE, "step": args.step,
        "molecule_order": list(config.MOLECULES),
        "cosmospace_convergence": 1e-10, "cosmospace_max_iter": 3000,
        "sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for name, path in files.items()},
        "versions": {name: package_version(name) for name in ("opencosmorspy", "numpy", "scipy")},
    }

    # Only the parent process writes results. Independent workers read surfaces.
    # The database allows resuming, and the lock prevents two writers at once.
    with (directory / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("This free-energy screen is already running.")
        database = sqlite3.connect(directory / "checkpoint.sqlite")
        pool = None
        try:
            database.execute("CREATE TABLE IF NOT EXISTS metadata (payload TEXT)")
            database.execute("CREATE TABLE IF NOT EXISTS recipes (id TEXT PRIMARY KEY, payload TEXT)")
            old = database.execute("SELECT payload FROM metadata").fetchone()
            if old and json.loads(old[0]) != metadata:
                raise RuntimeError("Input files, scripts or package versions changed. "
                                   "Use the original settings to resume, or move the existing "
                                   "free_energy_screen results directory before starting fresh.")
            if old is None:
                database.execute("INSERT INTO metadata VALUES (?)", (json.dumps(metadata),))
                database.commit()
            completed = {row[0] for row in database.execute("SELECT id FROM recipes")}
            pending = [p for p in recipes if "_".join(map(str, p)) not in completed]
            if args.limit is not None:
                pending = pending[:args.limit]
            print(f"Grid: {len(recipes):,} recipes; already saved: {len(completed):,}.")
            print(f"THIS RUN: {len(pending):,} additional free-energy calculations, "
                  f"using {args.workers} worker(s).", flush=True)
            print("Stability searches are not part of this screening stage.", flush=True)

            if pending:
                if args.workers == 1:
                    initialize_worker()
                else:
                    pool = ProcessPoolExecutor(
                        max_workers=args.workers,
                        mp_context=multiprocessing.get_context("spawn"),
                        initializer=initialize_worker,
                    )
            began = perf_counter()
            finished = 0
            try:
                # Small batches limit queued work, so Ctrl+C does not leave a
                # full grid running in the background. A batch has <=64 recipes.
                for offset in range(0, len(pending), 64):
                    batch = pending[offset:offset + 64]
                    results = (pool.map(evaluate_recipe, batch, chunksize=4) if pool
                               else map(evaluate_recipe, batch))
                    for row in results:
                        database.execute("INSERT INTO recipes VALUES (?, ?)",
                                         (row["recipe_id"], json.dumps(row, allow_nan=False)))
                        finished += 1
                    database.commit()
                    remaining_minutes = ((len(pending) - finished) *
                                         (perf_counter() - began) / finished / 60)
                    print(f"This run: {finished:,}/{len(pending):,}; estimated remaining "
                          f"{remaining_minutes:.1f} min", flush=True)
                    if finished % 1024 == 0:
                        export_ranking(database, directory)
            except KeyboardInterrupt:
                print("\nStopping after the active batch; received results will be saved.")
            finally:
                database.commit()
                if pool is not None:
                    pool.shutdown(wait=True, cancel_futures=True)
                export_ranking(database, directory)
            print("Saved ranking:", directory / "free_energy_ranking.csv")
            print("Repeat the command to resume; completed recipes are skipped.")
        finally:
            database.close()


if __name__ == "__main__":
    main()
