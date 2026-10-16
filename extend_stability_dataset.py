"""Reuse completed stability labels and collect additional recipes for Project 2.

Run beside collect_ml_data.py. The source campaign is read only.
Example: python extend_stability_dataset.py --total 2000 --workers 8 --limit 620
The limit counts all recipes, including the original 600.
"""

import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import fcntl
import hashlib
import json
import multiprocessing
from pathlib import Path
import sqlite3
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

import collect_ml_data as campaign


PROJECT = campaign.config.PROJECT
COMPOSITION = [f"x_{name}" for name in campaign.config.MOLECULES]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source(directory):
    """Read completed rows and verify their compositions and search protocol."""
    metadata = json.loads((directory / "metadata.json").read_text())
    data = pd.read_csv(directory / "dataset.csv")
    required = ["recipe_id", "recipe_number", "block_id", "split", "sampling_group",
                "label_status", "instability_label", "starts", *COMPOSITION]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Source dataset is missing columns: {missing}")
    if metadata["campaign"] != "stability" or metadata["starts"] != 40:
        raise ValueError("Use the existing five-solvent, 40-start stability campaign.")
    if data["recipe_id"].duplicated().any() or data.duplicated(COMPOSITION).any():
        raise ValueError("The source contains duplicate recipes.")
    data = data.sort_values("recipe_number").reset_index(drop=True)
    if not np.array_equal(data["recipe_number"], np.arange(1, len(data) + 1)):
        raise ValueError("Use a completed source campaign with consecutive recipe numbers.")
    x = data[COMPOSITION].to_numpy(dtype=float)
    if (not np.isfinite(x).all() or np.any(x < 0.15 - 1e-12)
            or np.any(x > 0.40 + 1e-12)
            or not np.allclose(x.sum(axis=1), 1, rtol=0, atol=1e-12)):
        raise ValueError("Source compositions must lie in the established 15–40 mol% region.")
    if not data["starts"].eq(40).all():
        raise ValueError("Not every source recipe used 40 starts.")
    allowed = {"instability_found", "no_instability_found", "needs_review"}
    if not set(data["label_status"]).issubset(allowed):
        raise ValueError("The source has unexpected label statuses.")
    for status, label in (("instability_found", 1), ("no_instability_found", 0)):
        if not data.loc[data["label_status"] == status, "instability_label"].eq(label).all():
            raise ValueError("A source classification label disagrees with its status.")
    return data, metadata


def make_manifest(source, total, sample_seed, search_seed):
    """Keep the original recipes first, then uniformly sample unseen grid recipes."""
    full_grid = campaign.make_manifest("thermo", search_seed)
    lookup = {row["recipe_id"]: row for row in full_grid}
    source_ids = set(source["recipe_id"])
    if not source_ids.issubset(lookup):
        raise ValueError("Some source recipes do not belong to the original 1 mol% grid.")
    if not len(source) <= total <= len(full_grid):
        raise ValueError(f"Total must be between {len(source)} and {len(full_grid)}.")
    rows = []
    for row in source.to_dict("records"):
        reference = lookup[row["recipe_id"]]
        if (row["block_id"] != reference["block_id"] or row["split"] != reference["split"]
                or not np.allclose([row[c] for c in COMPOSITION],
                                   [reference[c] for c in COMPOSITION], rtol=0, atol=1e-12)):
            raise ValueError("A source recipe or its original split differs from the recorded grid.")
        rows.append({key: row[key] for key in
                     ["recipe_id", "recipe_number", "block_id", "split", "sampling_group", *COMPOSITION]})
        rows[-1]["data_origin"] = "initial_campaign"
    remaining = [row for row in full_grid if row["recipe_id"] not in source_ids]
    order = np.random.default_rng(sample_seed).permutation(len(remaining))
    for index in order[:total - len(source)]:
        row = dict(remaining[index])
        row["recipe_number"] = len(rows) + 1
        row["data_origin"] = "uniform_extension"
        rows.append(row)
    for row in rows:
        # Preserve the old split as provenance. Project 2 will get its own
        # grouped, stratified split after collection and before model fitting.
        row["original_split"] = row["split"]
        row["split"] = "unassigned"
    return rows


def save_row(database, row):
    """Commit one completed recipe so it survives interruption."""
    database.execute("INSERT INTO results VALUES (?, ?)",
                     (row["recipe_id"], json.dumps(row, allow_nan=False)))
    database.commit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT / "results/ml/stability_seed88_starts40")
    parser.add_argument("--total", type=int, default=2000, help="Final count, including imported recipes")
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--limit", type=int, help="Absolute manifest prefix to complete, including imported rows")
    parser.add_argument("--describe", action="store_true", help="Print the plan without creating files or calculating")
    args = parser.parse_args()
    if args.sample_seed < 0:
        parser.error("Use a nonnegative sampling seed.")
    if tuple(campaign.config.MOLECULES) != ("water", "methanol", "acetone", "acetonitrile", "benzene"):
        parser.error("Use the established five-solvent order.")
    source, source_metadata = load_source(args.source)
    starts, search_seed = source_metadata["starts"], source_metadata["seed"]
    if args.limit is not None and not len(source) <= args.limit <= args.total:
        parser.error(f"The limit must be between {len(source)} imported rows and --total.")
    recipes = make_manifest(source, args.total, args.sample_seed, search_seed)
    selected = recipes[:args.limit] if args.limit is not None else recipes
    directory = PROJECT / "results/ml/project2_stability" / (
        f"total{args.total}_sample{args.sample_seed}_search{search_seed}_starts{starts}")
    print(f"Planned total: {len(recipes)}; reuse: {len(source)}; new labels: {len(recipes) - len(source)}")
    print(f"This command selects {len(selected)} total recipes, including the imported {len(source)}.")
    print("Output:", directory, flush=True)
    print("Project 2 split assignments: pending until collection is complete.", flush=True)
    if args.describe:
        return

    # The numerical protocol is exactly the existing collector's protocol. A
    # worker still performs 40 TPD minimizations and checks any negative result
    # with an explicit mass-balanced split. The source campaign stays read only.
    current = campaign.provenance("stability", starts, search_seed)
    if current != source_metadata:
        raise ValueError("The source calculation inputs, code or environment changed; inspect metadata.json before combining labels.")
    manifest_text = pd.DataFrame(recipes).to_csv(index=False)
    metadata = {
        "campaign": "project2_stability_extension", "total": args.total,
        "source_directory": str(args.source.resolve()), "source_rows": len(source),
        "source_dataset_sha256": digest(args.source / "dataset.csv"),
        "extension_script_sha256": digest(Path(__file__)),
        "manifest_sha256": hashlib.sha256(manifest_text.encode()).hexdigest(),
        "sample_seed": args.sample_seed, "calculation_metadata": current,
        "split_policy": "Pending Project 2 split: aim 60/15/25 train/validation/test, stratified by label with composition blocks kept together; no classifier fitted yet",
    }
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "run.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("This extension is already running; use its existing terminal.")
        metadata_file = directory / "metadata.json"
        if metadata_file.exists():
            if json.loads(metadata_file.read_text()) != metadata:
                raise ValueError("Extension inputs changed; keep this directory intact and inspect metadata.json.")
        else:
            metadata_file.write_text(json.dumps(metadata, indent=2) + "\n")
        manifest_file = directory / "recipes.csv"
        if manifest_file.exists() and manifest_file.read_text() != manifest_text:
            raise ValueError("The saved extension manifest differs from the planned recipes.")
        if not manifest_file.exists():
            manifest_file.write_text(manifest_text)
        (directory / "searches").mkdir(exist_ok=True)

        with sqlite3.connect(directory / "checkpoint.sqlite") as database:
            database.execute("CREATE TABLE IF NOT EXISTS results (recipe_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            completed = {row[0] for row in database.execute("SELECT recipe_id FROM results")}
            clean_source = source.astype(object).where(pd.notna(source), None)
            for old, recipe in zip(clean_source.to_dict("records"), recipes[:len(source)]):
                if old["recipe_id"] not in completed:
                    old.update(recipe)
                    old["searches_file"] = os.path.relpath(args.source / "searches" / f"{old['recipe_id']}.csv", PROJECT)
                    save_row(database, old)
                    completed.add(old["recipe_id"])
            campaign.export_dataset(database, directory)
            pending = [row for row in selected if row["recipe_id"] not in completed]
            print(f"Already saved: {len(completed)}; new calculations requested now: {len(pending)}", flush=True)
            started, finished = perf_counter(), 0
            pool = ProcessPoolExecutor(max_workers=args.workers, initializer=campaign.initialize_worker,
                                       mp_context=multiprocessing.get_context("spawn")) if pending else None
            futures, todo = {}, iter(pending)

            def submit_next():
                recipe = next(todo, None)
                if recipe is not None:
                    future = pool.submit(campaign.evaluate_recipe, recipe, "stability", starts, search_seed, str(directory))
                    futures[future] = recipe

            def record_result(future):
                nonlocal finished
                row = future.result()
                row["searches_file"] = os.path.relpath(directory / "searches" / f"{row['recipe_id']}.csv", PROJECT)
                save_row(database, row)
                finished += 1
                campaign.export_dataset(database, directory)
                remaining_h = (perf_counter() - started) / finished * (len(pending) - finished) / 3600
                print(f"New labels this invocation: {finished}/{len(pending)}; {row['label_status']}; "
                      f"estimated remaining {remaining_h:.2f} h", flush=True)

            try:
                for _ in range(min(args.workers, len(pending))):
                    submit_next()
                while futures:
                    ready, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in ready:
                        futures.pop(future)
                        record_result(future)
                        submit_next()
            except KeyboardInterrupt:
                print("\nStopping submissions; finishing and saving active recipes before exit.", flush=True)
                for future in futures:
                    future.cancel()
                for future in futures:
                    if not future.cancelled():
                        record_result(future)
            finally:
                for future in futures:
                    future.cancel()
                if pool is not None:
                    pool.shutdown(wait=True, cancel_futures=True)
                campaign.export_dataset(database, directory)
            result = pd.read_csv(directory / "dataset.csv")
            print("\nSaved status counts:")
            print(result["label_status"].value_counts().to_string())
            print("Saved:", directory / "dataset.csv")


if __name__ == "__main__":
    main()
