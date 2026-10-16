"""Freeze validation cutoffs, then evaluate completed active-learning paths.

No fitting or recipe selection happens here. Run only after run_active_learning.py
has completed every strategy and seed. Outputs are written to RUN_DIR/report/.
"""

import argparse
import fcntl
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import project2_common as common
import project3_common as al


def collect_checkpoints(directory, protocol, manifest, data):
    """Verify stored paths and pick thresholds using validation labels alone."""
    indexed = data.set_index("recipe_id")
    records, validation_metrics, test_scores = [], [], []
    for trajectory in manifest["trajectories"]:
        path = directory / trajectory["directory"]
        al.require(common.sha256(path / "state.json") == trajectory["state_sha256"], "State changed.")
        al.require(common.sha256(path / "queries.csv") == trajectory["queries_sha256"], "Queries changed.")
        state = json.loads((path / "state.json").read_text())
        al.require(state["complete"], "An acquisition path is unfinished.")
        queries = pd.read_csv(path / "queries.csv")
        al.require(queries.recipe_id.is_unique, "Repeated query within an acquisition path.")
        al.require(indexed.loc[queries.recipe_id, "split"].eq("train").all(), "A held-out label was queried.")
        for checkpoint in state["checkpoints"]:
            for kind in ["predictions", "model"]:
                al.require(common.sha256(path / checkpoint[kind + "_file"]) == checkpoint[kind + "_sha256"],
                           f"Saved {kind} changed: {path}.")
            p = pd.read_csv(path / checkpoint["predictions_file"])
            al.require(p.recipe_id.is_unique and np.isfinite(p.score).all(), "Invalid saved scores.")
            for role in ["validation", "test"]:
                al.require(set(p.loc[p.split == role, "recipe_id"]) == set(data.loc[data.split == role, "recipe_id"]),
                           f"{role} recipes do not match the frozen split.")
            al.require(np.array_equal(p.block_id, indexed.loc[p.recipe_id, "block_id"]), "Block mismatch.")
            v = p[p.split == "validation"]
            y_val = indexed.loc[v.recipe_id, "instability_label"].to_numpy(int)
            threshold = .5 if checkpoint["constant_class"] is not None else common.choose_threshold(y_val, v.score)
            ids = {"strategy": state["strategy"], "seed": state["seed"],
                   "training_labels": checkpoint["training_labels"]}
            record = {**ids, "threshold": threshold, **{key: checkpoint[key] for key in
                      ["positives_revealed", "fit_seconds_cumulative", "acquisition_seconds_cumulative",
                       "heldout_scoring_seconds_cumulative", "recorded_search_seconds_sum",
                       "fits_with_warning_this_round"]}}
            record.update({"validation_label_overhead": protocol["split_counts"]["validation"],
                           "test_labels_for_evaluation": protocol["split_counts"]["test"]})
            record["labels_for_training_and_threshold"] = record["training_labels"] + record["validation_label_overhead"]
            record["labels_including_evaluation"] = record["labels_for_training_and_threshold"] + record["test_labels_for_evaluation"]
            records.append(record)
            validation_metrics.append({**record, **common.metrics(y_val, v.score, threshold)})
            test_scores.append(p[p.split == "test"].assign(**ids, threshold=threshold))
    # Saved before the first access to test answers in the evaluation loop.
    return records, validation_metrics, test_scores


def plot_curves(metrics, destination):
    colors = {"random": "#777777", "space_filling": "#218c8d", "uncertainty_diversity": "#b54b39"}
    labels = {"random": "Random", "space_filling": "Space filling", "uncertainty_diversity": "Uncertainty + diversity"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    columns = [("average_precision", "Average precision"),
               ("recall_instability", "Instability recall"),
               ("f1_instability", "Instability F1"),
               ("recorded_search_seconds_sum", "Sum of recorded search times (hours)")]
    for ax, (column, ylabel) in zip(axes.flat, columns):
        for strategy in al.STRATEGIES:
            rows = metrics[metrics.strategy == strategy]
            stats = rows.groupby("training_labels")[column].agg(["mean", "std"]).fillna(0)
            scale = 3600 if column == "recorded_search_seconds_sum" else 1
            x = stats.index.to_numpy(); mean = stats["mean"].to_numpy() / scale; sd = stats["std"].to_numpy() / scale
            ax.plot(x, mean, "o-", color=colors[strategy], label=labels[strategy])
            lo, hi = mean - sd, mean + sd
            if scale == 1:
                lo, hi = np.maximum(0, lo), np.minimum(1, hi)
            ax.fill_between(x, lo, hi, color=colors[strategy], alpha=.12)
        if column != "recorded_search_seconds_sum":
            reference = metrics[metrics.strategy == "full_pool_reference"][column].mean()
            ax.axhline(reference, color="#333333", linestyle="--", label="Same learner, full pool")
            ax.set_ylim(0, 1.025)
        ax.set_xlabel("Training-pool labels revealed")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Project 3: label efficiency on the fixed Project 2 test recipes")
    fig.text(.5, .02, "Bands: ±1 SD across seeds, not confidence intervals. Recorded search times are historical cost proxies.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .97))
    fig.savefig(destination / "learning_curves.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, help="Optional relocation of the same frozen prepared dataset")
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    protocol = json.loads((directory / "protocol.json").read_text())
    manifest = json.loads((directory / "learning_complete.json").read_text())
    al.require(common.sha256(directory / "protocol.json") == manifest["protocol_sha256"], "Protocol changed.")
    for filename, expected in protocol["code_sha256"].items():
        path = Path(common.__file__) if filename == "project2_common.py" else al.ROOT / filename
        al.require(common.sha256(path) == expected, f"Code changed: {filename}. Keep a versioned new experiment.")
    prepared = args.prepared or Path(protocol["prepared_directory"])
    al.require(common.sha256(prepared / "preparation.json") == protocol["preparation_sha256"], "Different prepared dataset.")
    data, _ = al.load_data(prepared)
    destination = directory / "report"
    with (directory / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Learning or evaluation is already running here.") from error
        if (destination / "evaluation_complete.json").exists():
            print("The frozen evaluation is already complete:", destination)
            return
        destination.mkdir(exist_ok=True)
        records, validation, test_scores = collect_checkpoints(directory, protocol, manifest, data)
        common.write_json(destination / "frozen_thresholds.json", records)
        common.write_csv(destination / "validation_metrics.csv", pd.DataFrame(validation))
        print("All paths and validation cutoffs are fixed. Evaluating test labels now.", flush=True)

        indexed = data.set_index("recipe_id")
        results, predictions = [], []
        for record, p in zip(records, test_scores):
            y = indexed.loc[p.recipe_id, "instability_label"].to_numpy(int)
            results.append({**record, **common.metrics(y, p.score, record["threshold"])})
            p = p.copy(); p["label"] = y; p["predicted"] = (p.score >= record["threshold"]).astype(int)
            predictions.append(p)
        metrics = pd.DataFrame(results)
        common.write_csv(destination / "test_metrics.csv", metrics)
        common.write_csv(destination / "test_predictions.csv", pd.concat(predictions, ignore_index=True))

        measure = ["average_precision", "pr_auc_trapezoid", "recall_instability", "precision_instability",
                   "f1_instability", "fn", "fp", "recorded_search_seconds_sum",
                   "fit_seconds_cumulative", "acquisition_seconds_cumulative"]
        summary = metrics.groupby(["strategy", "training_labels"])[measure].agg(["mean", "std"])
        summary.columns = [f"{a}_{b}" for a, b in summary.columns]
        summary = summary.reset_index()
        common.write_csv(destination / "summary.csv", summary)

        # Pair strategies using the same seed and initial recipes. This compares
        # selection policies at equal budgets rather than comparing unlike runs.
        random = metrics[metrics.strategy == "random"].set_index(["seed", "training_labels"])
        paired, milestones = [], []
        for row in results:
            if row["strategy"] in {"random", "full_pool_reference"}:
                continue
            baseline = random.loc[(row["seed"], row["training_labels"])]
            paired.append({key: row[key] for key in ["strategy", "seed", "training_labels"]} |
                          {f"difference_{key}": row[key] - baseline[key] for key in measure})
        common.write_csv(destination / "paired_differences_vs_random.csv", pd.DataFrame(paired))
        for (strategy, seed), group in metrics[metrics.strategy != "full_pool_reference"].groupby(["strategy", "seed"]):
            passed = group[(group.average_precision >= protocol["config"]["target_ap"])
                           & (group.recall_instability >= protocol["config"]["target_recall"])]
            milestones.append({"strategy": strategy, "seed": int(seed),
                               "target_ap": protocol["config"]["target_ap"],
                               "target_recall": protocol["config"]["target_recall"],
                               "first_evaluated_budget_meeting_both": int(passed.training_labels.min()) if len(passed) else None})
        common.write_csv(destination / "label_efficiency.csv", pd.DataFrame(milestones))
        plot_curves(metrics, destination)
        print(summary[["strategy", "training_labels", "average_precision_mean",
                       "recall_instability_mean", "f1_instability_mean"]].round(4).to_string(index=False))
        common.write_json(destination / "evaluation_complete.json", {
            "protocol_sha256": common.sha256(directory / "protocol.json"),
            "frozen_thresholds_sha256": common.sha256(destination / "frozen_thresholds.json"),
            "test_metrics_sha256": common.sha256(destination / "test_metrics.csv"),
            "notes": ["No threshold or setting was chosen using test labels.",
                      "This reuses Project 2's already-inspected test as a retrospective follow-up.",
                      "Seed SD is not uncertainty over new experimental mixtures.",
                      "Search-time sums describe old campaign costs, not parallel wall-time predictions."]})
        print("\nSaved:", destination)
        print("Compare strategies at the same label budget; inspect warnings before interpreting a weak learner.")


if __name__ == "__main__":
    main()
