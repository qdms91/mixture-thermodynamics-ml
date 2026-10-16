"""Project 3: compare recipe selection while hiding unqueried training labels.

Run this script first, then evaluate_active_learning.py. This stage saves
predictions but never computes validation/test metrics or selects a threshold.
"""

import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from datetime import datetime, timezone
import fcntl
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import joblib
from joblib import Parallel, delayed, parallel_config
import numpy as np
import pandas as pd

import project2_common as common
import project3_common as al


def save_package(path, package):
    temporary = path.with_suffix(".tmp")
    joblib.dump(package, temporary, compress=3)
    temporary.replace(path)


def verify_checkpoints(directory, state):
    for item in state["checkpoints"]:
        for kind in ["predictions", "model"]:
            if common.sha256(directory / item[kind + "_file"]) != item[kind + "_sha256"]:
                raise ValueError(f"A saved checkpoint changed: {directory}, {kind}.")


def run_trajectory(pool, answers, historical_costs, heldout, config, strategy, seed, directory):
    """One independent process handles one strategy/seed and its checkpoints."""
    directory.mkdir(parents=True, exist_ok=True)
    state_file = directory / "state.json"
    reference = strategy == "full_pool_reference"
    maximum = len(pool) if reference else max(config["budgets"])
    if state_file.exists():
        state = json.loads(state_file.read_text())
        al.require(state["strategy"] == strategy and state["seed"] == seed, "Checkpoint identity mismatch.")
        verify_checkpoints(directory, state)
        if state["complete"]:
            print(f"Reusing completed {strategy}, seed {seed}.", flush=True)
            return str(directory)
    else:
        selected = list(range(len(pool))) if reference else al.initial_positions(len(pool), config["initial"], seed)
        state = {"strategy": strategy, "seed": seed, "selected": selected,
                 "acquired_after_budget": [0] * len(selected), "checkpoints": [], "rounds": [],
                 "fit_seconds_cumulative": 0., "acquisition_seconds_cumulative": 0.,
                 "heldout_scoring_seconds_cumulative": 0., "complete": False}

    x = pool[common.COMPOSITION].to_numpy(float)
    x5 = pool[common.X_COLUMNS].to_numpy(float)
    held_x = heldout[common.COMPOSITION].to_numpy(float)
    oracle = al.LabelOracle(answers)
    while not state["complete"]:
        selected = state["selected"]
        y = oracle.reveal(selected)
        budget = len(selected)
        package, diagnostics = al.fit_committee(x[selected], y, seed, config)
        state["fit_seconds_cumulative"] += diagnostics["fit_seconds"]
        state["rounds"].append({"training_labels": budget, "positives_revealed": int(y.sum()),
                                **diagnostics})

        if reference or budget in config["budgets"]:
            # Scores are saved without labels. Neither held-out split can affect
            # the next query; threshold selection occurs only in the report step.
            started = perf_counter()
            scores = al.committee_scores(package, held_x)
            prediction = heldout[["recipe_id", "block_id", "split"]].copy()
            prediction["score"] = scores.mean(axis=0)
            prediction["committee_score_sd"] = scores.std(axis=0)
            for member, values in enumerate(scores):
                prediction[f"member_{member}_score"] = values
            state["heldout_scoring_seconds_cumulative"] += perf_counter() - started
            model_file = directory / f"model_{budget:04d}.joblib"
            prediction_file = directory / f"predictions_{budget:04d}.csv"
            package.update({"columns": common.COMPOSITION, "seed": seed,
                            "training_recipe_ids": pool.iloc[selected].recipe_id.tolist(),
                            "config": config, "fit_diagnostics": diagnostics})
            save_package(model_file, package)
            common.write_csv(prediction_file, prediction)
            state["checkpoints"].append({
                "training_labels": budget, "positives_revealed": int(y.sum()),
                "model_file": model_file.name, "model_sha256": common.sha256(model_file),
                "predictions_file": prediction_file.name,
                "predictions_sha256": common.sha256(prediction_file),
                "constant_class": package["constant"],
                "fit_seconds_cumulative": state["fit_seconds_cumulative"],
                "acquisition_seconds_cumulative": state["acquisition_seconds_cumulative"],
                "heldout_scoring_seconds_cumulative": state["heldout_scoring_seconds_cumulative"],
                "recorded_search_seconds_sum": float(historical_costs[selected].sum()),
                "fits_with_warning_this_round": len(diagnostics["warnings"]),
            })

        print(f"{strategy:23s} seed={seed:4d}: {budget:4d}/{maximum} labels; "
              f"{int(y.sum())} positives revealed; warnings={len(diagnostics['warnings'])}", flush=True)
        if budget == maximum:
            state["complete"] = True
        else:
            # Stop at requested reporting budgets even when batch size does not
            # divide the interval exactly. Every strategy receives equal budgets.
            next_checkpoint = min(b for b in config["budgets"] if b > budget)
            count = min(config["batch"], maximum - budget, next_checkpoint - budget)
            started = perf_counter()
            member_scores = al.committee_scores(package, x) if strategy == "uncertainty_diversity" else None
            chosen = al.choose_batch(strategy, x5, selected, count, seed, member_scores)
            al.require(not set(chosen).intersection(selected), "A recipe was selected twice.")
            oracle.reveal(chosen)  # This lookup replaces a new expensive calculation.
            state["selected"] = selected + chosen
            state["acquired_after_budget"] += [budget] * len(chosen)
            state["acquisition_seconds_cumulative"] += perf_counter() - started

        # Commit after each batch. A restart reconstructs the revealed-label set
        # and repeats at most an unfinished batch; it never changes the protocol.
        queries = pool.iloc[state["selected"]][["recipe_id", "block_id", *common.X_COLUMNS]].copy()
        queries.insert(0, "query_order", np.arange(1, len(queries) + 1))
        queries["acquired_after_budget"] = state["acquired_after_budget"]
        queries["revealed_label"] = oracle.reveal(state["selected"])
        queries["recorded_search_seconds"] = historical_costs[state["selected"]]
        common.write_csv(directory / "queries.csv", queries)
        common.write_json(state_file, state)
    return str(directory)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=al.DEFAULT_PREPARED)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    parser.add_argument("--initial", type=int, default=100)
    parser.add_argument("--budgets", type=int, nargs="+", default=[200, 400, 800])
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--committee", type=int, default=3)
    parser.add_argument("--layers", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--alpha", type=float, default=.01)
    parser.add_argument("--max-epochs", type=int, default=1200)
    parser.add_argument("--target-ap", type=float, default=.98)
    parser.add_argument("--target-recall", type=float, default=.90)
    args = parser.parse_args()
    if min(args.workers, args.initial, args.batch, args.committee, args.max_epochs, *args.layers) < 1:
        parser.error("Counts, layer sizes and workers must be positive.")
    if args.committee < 2:
        parser.error("Use at least two committee members.")
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct nonnegative integers.")
    if not np.isfinite(args.alpha) or args.alpha <= 0:
        parser.error("alpha must be finite and positive.")
    if not (0 < args.target_ap <= 1 and 0 < args.target_recall <= 1):
        parser.error("Metric targets must be in (0, 1].")
    if min(args.budgets) < args.initial or max(args.budgets) <= args.initial:
        parser.error("Budgets must include a value larger than the initial sample.")
    if args.resume and args.run_dir is None:
        parser.error("Resuming requires --run-dir.")
    return args


def main():
    args = parse_args()
    data, _ = al.load_data(args.prepared)
    pool_rows = data[data.split == "train"].sort_values("recipe_id").reset_index(drop=True)
    heldout = data[data.split != "train"].sort_values("recipe_id").reset_index(drop=True)
    maximum = max(args.budgets)
    al.require(maximum <= len(pool_rows), f"Only {len(pool_rows)} training-pool recipes are available.")
    costs = pool_rows.stability_seconds.to_numpy(float)
    al.require(np.isfinite(costs).all() and (costs >= 0).all(), "Invalid recorded search timings.")
    config = {key: getattr(args, key) for key in
              ["initial", "batch", "committee", "layers", "alpha", "max_epochs", "target_ap", "target_recall"]}
    config["budgets"] = sorted(set([args.initial, *args.budgets]))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    directory = (args.run_dir or al.ROOT / "results/ml/project3_active_learning" / f"run_{stamp}").resolve()
    if directory.exists() and not args.resume:
        raise ValueError(f"Directory exists: {directory}. Use --resume with identical settings, or choose a new directory.")
    if args.resume and not (directory / "protocol.json").is_file():
        raise ValueError("No complete protocol exists in the requested resume directory.")
    directory.mkdir(parents=True, exist_ok=True)

    code_files = [Path(__file__), al.ROOT / "project3_common.py",
                  al.ROOT / "evaluate_active_learning.py", Path(common.__file__)]
    protocol = {
        "prepared_directory": str(args.prepared.resolve()),
        "preparation_sha256": common.sha256(args.prepared / "preparation.json"),
        "config": config, "seeds": args.seeds, "strategies": al.STRATEGIES,
        "columns": common.COMPOSITION, "acquisition_distance_columns": common.X_COLUMNS,
        "split_counts": {str(k): int(v) for k, v in data.split.value_counts().items()},
        "code_sha256": {p.name: common.sha256(p) for p in code_files},
        "versions": {name: version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl"]},
        "learner": "Fresh balanced MLP committee; fixed settings, no hyperparameter search or pretrained weights. One training-only continuation at twice the iteration cap if needed.",
        "acquisition": "Jensen-Shannon disagreement + .25 predictive entropy; top 5*informed shortlist; farthest-first; 20% random exploration.",
        "heldout_policy": "Validation labels used only for final thresholds; test labels only for final reporting.",
        "interpretation": "Retrospective follow-up after Project 2; existing test has already been inspected.",
    }
    started = perf_counter()
    with (directory / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is using this run directory.") from error
        protocol_path = directory / "protocol.json"
        if args.resume:
            al.require(json.loads(protocol_path.read_text()) == protocol,
                       "Protocol, code, data or software changed. Use a new run directory.")
        else:
            common.write_json(protocol_path, protocol)
            common.write_csv(directory / "frozen_roles.csv", data[["recipe_id", "block_id", "split"]])
        if (directory / "learning_complete.json").exists():
            print("Learning is already complete. Run evaluate_active_learning.py for the report.")
            return
        print("Prepared:", args.prepared.resolve())
        print("Frozen roles:", protocol["split_counts"])
        print("Training budgets:", config["budgets"], "plus a separate full-pool reference.")
        print("Strategies:", ", ".join(al.STRATEGIES))
        print("No validation/test labels enter learning or recipe selection.")
        print("Output:", directory, flush=True)
        features = ["recipe_id", "block_id", "split", *common.X_COLUMNS]
        pool = pool_rows[features].copy()
        labels = pool_rows.instability_label.to_numpy(int)
        held_features = heldout[features].copy()
        try:
            with parallel_config(backend="loky", n_jobs=args.workers, inner_max_num_threads=1):
                tasks = Parallel()(delayed(run_trajectory)(
                    pool, labels, costs, held_features, config, strategy, seed,
                    directory / f"seed_{seed}" / strategy)
                    for seed in args.seeds for strategy in al.STRATEGIES)
                # A matched full-data reference is trained only after all query
                # paths finish. It cannot influence any active-learning query.
                references = Parallel()(delayed(run_trajectory)(
                    pool, labels, costs, held_features, config, "full_pool_reference", seed,
                    directory / f"seed_{seed}" / "full_pool_reference") for seed in args.seeds)
            manifest = [{"directory": str(Path(p).relative_to(directory)),
                         "state_sha256": common.sha256(Path(p) / "state.json"),
                         "queries_sha256": common.sha256(Path(p) / "queries.csv")} for p in tasks + references]
            common.write_json(directory / "learning_complete.json", {
                "protocol_sha256": common.sha256(protocol_path), "trajectories": manifest})
            print("Learning finished. Run evaluate_active_learning.py --run-dir", directory, flush=True)
        except KeyboardInterrupt:
            print("\nInterrupted. Repeat the identical command with --resume; completed batches are saved.", flush=True)
        finally:
            with (directory / "invocations.jsonl").open("a") as stream:
                stream.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(),
                                         "workers": args.workers,
                                         "elapsed_seconds": perf_counter() - started}) + "\n")


if __name__ == "__main__":
    main()
