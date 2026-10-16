"""Compare four classifier families and five feature groups on frozen splits.

Default: grouped cross-validation, validation thresholds, then one test report.
Run beside project2_common.py. No ORCA or COSMO-RS calculation is started here.
"""

import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from datetime import datetime, timezone
import fcntl
from importlib.metadata import version
from itertools import product
import json
from pathlib import Path
from time import perf_counter
import warnings

import joblib
from joblib import Parallel, delayed, parallel_config
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

import project2_common as common

FAMILIES = ["majority", "logistic", "svm", "mlp"]
MLP_CV_SEEDS = [88, 2026]
MLP_FINAL_SEEDS = [88, 2026, 31415]


def settings():
    """Declare the entire search before reading any model evaluation results."""
    candidates = []
    for C, weighted in product([.1, 1., 10.], [False, True]):
        candidates.append(dict(family="logistic", C=C, weighted=weighted))
    for C, gamma, weighted in product([.1, 1., 10., 100.], ["scale", .1, 1.], [False, True]):
        candidates.append(dict(family="svm", C=C, gamma=gamma, weighted=weighted))
    for layers, alpha, weighted in product([(64, 32), (128, 64), (256, 128, 64)],
                                            [.0001, .01, .1], [False, True]):
        candidates.append(dict(family="mlp", layers=list(layers), alpha=alpha, weighted=weighted))
    for i, candidate in enumerate(candidates):
        candidate["setting_id"] = f"{candidate['family']}_{i:02d}"
    return candidates


def build_estimator(setting, seed):
    family = setting["family"]
    if family == "majority":
        model = DummyClassifier(strategy="most_frequent")
    elif family == "logistic":
        model = LogisticRegression(C=setting["C"], solver="lbfgs", max_iter=3000, tol=1e-7)
    elif family == "svm":
        # Use the decision score directly. probability=True would add an
        # internal calibration split that does not preserve our blocks.
        model = SVC(C=setting["C"], gamma=setting["gamma"], kernel="rbf",
                    probability=False, cache_size=128, tol=1e-5)
    else:
        # Stopping uses training loss, not an internal random validation split.
        # This preserves groups and reserves our validation set for thresholds.
        model = MLPClassifier(hidden_layer_sizes=tuple(setting["layers"]),
            alpha=setting["alpha"], solver="adam", activation="relu", batch_size=64,
            learning_rate_init=.001, max_iter=1200, tol=1e-5,
            n_iter_no_change=40, early_stopping=False, random_state=seed)
    return make_pipeline(StandardScaler(), model)


def fit_estimator(setting, x, y, seed):
    estimator = build_estimator(setting, seed)
    name = estimator.steps[-1][0]
    fit_arguments = {}
    if setting.get("weighted", False):
        # Compute weights only from this fit's training labels, never validation
        # or test labels. No synthetic recipes are generated.
        fit_arguments[f"{name}__sample_weight"] = compute_sample_weight("balanced", y)
    started = perf_counter()
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(x, y, **fit_arguments)
    convergence = [str(w.message) for w in captured if issubclass(w.category, ConvergenceWarning)]
    return estimator, dict(fit_seconds=perf_counter()-started,
                          convergence_warnings=convergence,
                          iterations=np.asarray(getattr(estimator.steps[-1][1], "n_iter_", [])).tolist())


def scores(estimators, x, family):
    if family == "svm":
        return estimators[0].decision_function(x)
    return np.mean([model.predict_proba(x)[:, 1] for model in estimators], axis=0)


def evaluate_setting(group, columns, setting, training, directory):
    """One parallel task: evaluate a setting using training data only."""
    output = directory / "search" / f"{group}__{setting['setting_id']}.json"
    if output.exists():
        return json.loads(output.read_text())
    x = training[columns].to_numpy(float)
    y = training.instability_label.to_numpy(int)
    fold = training.cv_fold.to_numpy(int)
    seeds = MLP_CV_SEEDS if setting["family"] == "mlp" else [88]
    records = []
    for held_fold in sorted(np.unique(fold)):
        fit, held = fold != held_fold, fold == held_fold
        if len(np.unique(y[fit])) != 2 or len(np.unique(y[held])) != 2:
            raise ValueError("A grouped training fold is missing a class.")
        if set(training.loc[fit, "block_id"]) & set(training.loc[held, "block_id"]):
            raise ValueError("Composition blocks cross an inner-CV boundary.")
        for seed in seeds:
            model, diagnostics = fit_estimator(setting, x[fit], y[fit], seed)
            score = scores([model], x[held], setting["family"])
            records.append({"fold": int(held_fold), "seed": seed,
                "average_precision": float(average_precision_score(y[held], score)), **diagnostics})
    values = [row["average_precision"] for row in records]
    result = {"group": group, "setting": setting,
        "mean_cv_average_precision": float(np.mean(values)),
        "std_cv_average_precision": float(np.std(values, ddof=1)),
        "fits_with_convergence_warning": sum(bool(row["convergence_warnings"]) for row in records),
        "fit_seconds": float(sum(row["fit_seconds"] for row in records)), "fold_results": records}
    common.write_json(output, result)
    return result


def fit_selected(group, columns, setting, training, validation, directory):
    """Refit on training only; select a threshold using validation predictions."""
    tag = f"{group}__{setting['family']}"
    destination = directory / "models" / f"{tag}.joblib"
    x = training[columns].to_numpy(float)
    y = training.instability_label.to_numpy(int)
    seeds = MLP_FINAL_SEEDS if setting["family"] == "mlp" else [88]
    if destination.exists():
        package = joblib.load(destination)  # Only load files created by this run.
    else:
        fits = [fit_estimator(setting, x, y, seed) for seed in seeds]
        package = {"models": [item[0] for item in fits], "columns": columns,
            "group": group, "family": setting["family"], "setting": setting,
            "seeds": seeds, "fit_diagnostics": [item[1] for item in fits]}
        temporary = destination.with_suffix(".tmp")
        joblib.dump(package, temporary)
        temporary.replace(destination)
    score = scores(package["models"], validation[columns].to_numpy(float), setting["family"])
    y_validation = validation.instability_label.to_numpy(int)
    # The majority baseline remains a genuine majority rule; do not tune it
    # into an all-positive classifier merely to increase its F1.
    threshold = .5 if setting["family"] == "majority" else common.choose_threshold(y_validation, score)
    result = {"group": group, "family": setting["family"], "setting": setting,
        "threshold": threshold, "columns": columns, "seeds": seeds,
        "model_file": str(destination.relative_to(directory)),
        "model_sha256": common.sha256(destination),
        "fit_diagnostics": package["fit_diagnostics"],
        "validation_metrics": common.metrics(y_validation, score, threshold)}
    common.write_csv(directory / "validation_predictions" / f"{tag}.csv", pd.DataFrame({
        "recipe_id": validation.recipe_id, "label": y_validation, "score": score,
        "predicted": (score >= threshold).astype(int)}))
    return result


def bootstrap_intervals(y, score, threshold, blocks, count, seed=2026):
    """Resample whole test blocks; intervals describe this fitted model only."""
    rng = np.random.default_rng(seed)
    unique = np.unique(blocks)
    locations = [np.flatnonzero(blocks == block) for block in unique]
    rows = []
    for _ in range(count):
        index = np.concatenate([locations[i] for i in rng.integers(0, len(unique), len(unique))])
        if len(np.unique(y[index])) < 2:
            continue
        rows.append(common.metrics(y[index], score[index], threshold))
    result = {"valid_resamples": len(rows), "requested_resamples": count}
    for key in ["precision_instability", "recall_instability", "f1_instability",
                "average_precision", "pr_auc_trapezoid", "f1_no_instability"]:
        if rows:
            result[f"{key}_ci_low"], result[f"{key}_ci_high"] = [float(v) for v in
                np.quantile([row[key] for row in rows], [.025, .975])]
    return result


def report_test(data, selections, directory, bootstrap):
    """Use frozen estimators and thresholds; never optimize anything on test."""
    test = data.loc[data.split.eq("test")].reset_index(drop=True)
    y = test.instability_label.to_numpy(int)
    rows, predictions, uncertainty, subgroup_rows, cost_rows = [], [], [], [], []
    fig, axes = plt.subplots(1, len(common.feature_groups()), figsize=(22, 4.6), sharey=True)
    for group_index, group in enumerate(common.feature_groups()):
        ax = axes[group_index]
        ax.plot([0, 1], [y.mean(), y.mean()], "--", color="gray", label="Prevalence")
        for chosen in [s for s in selections["models"] if s["group"] == group]:
            print(f"Test report: {group}; {chosen['family']}", flush=True)
            model_file = directory / chosen["model_file"]
            if common.sha256(model_file) != chosen["model_sha256"]:
                raise ValueError("A selected model changed before test evaluation.")
            package = joblib.load(model_file)
            x = test[chosen["columns"]].to_numpy(float)
            # Timings here exclude feature computation. Report that separately.
            scores(package["models"], x[:1], chosen["family"])
            times = []
            for _ in range(5):
                started = perf_counter()
                score = scores(package["models"], x, chosen["family"])
                times.append((perf_counter() - started) / len(x))
            identifiers = {"group": group, "family": chosen["family"]}
            threshold = chosen["threshold"]
            rows.append({**identifiers, "threshold": threshold, **common.metrics(y, score, threshold)})
            uncertainty.append({**identifiers, **bootstrap_intervals(y, score, threshold,
                test.block_id.to_numpy(), bootstrap)})
            predictions.append(pd.DataFrame({**identifiers, "recipe_id": test.recipe_id,
                "block_id": test.block_id, "sampling_group": test.sampling_group,
                "label": y, "score": score, "predicted": (score >= threshold).astype(int),
                "threshold": threshold}))
            for subgroup in sorted(test.sampling_group.unique()):
                mask = test.sampling_group.eq(subgroup).to_numpy()
                subgroup_rows.append({**identifiers, "sampling_group": subgroup,
                    **common.metrics(y[mask], score[mask], threshold)})
            precision, recall, _ = precision_recall_curve(y, score)
            if chosen["family"] != "majority":
                ax.plot(recall, precision, label=chosen["family"])
            cost_rows.append({**identifiers, "batch_prediction_seconds_per_recipe": float(np.median(times)),
                "model_bytes": model_file.stat().st_size,
                "requires_cosmors_features": group in {"c_all", "d_thermodynamics", "e_activity"},
                "feed_temperature_evaluations_without_step_check": 3 if group in {"c_all", "d_thermodynamics"} else (1 if group == "e_activity" else 0),
                "campaign_all_features_median_seconds": float(test.feature_seconds.median()),
                "campaign_stability_median_seconds": float(test.stability_seconds.median())})
        ax.set(title=group, xlabel="Instability recall", xlim=(0, 1), ylim=(0, 1.02))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Instability precision")
    fig.tight_layout()
    fig.savefig(directory / "test_precision_recall.png", dpi=180)
    plt.close(fig)
    table = pd.DataFrame(rows)
    common.write_csv(directory / "test_metrics.csv", table)
    common.write_csv(directory / "test_predictions.csv", pd.concat(predictions, ignore_index=True))
    common.write_csv(directory / "test_group_bootstrap_intervals.csv", pd.DataFrame(uncertainty))
    common.write_csv(directory / "test_metrics_by_sampling_group.csv", pd.DataFrame(subgroup_rows))
    common.write_csv(directory / "prediction_costs.csv", pd.DataFrame(cost_rows))
    fig, axes = plt.subplots(len(common.feature_groups()), len(FAMILIES), figsize=(12, 13))
    for i, group in enumerate(common.feature_groups()):
        for j, family in enumerate(FAMILIES):
            row = table.loc[table.group.eq(group) & table.family.eq(family)].iloc[0]
            matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
            ax = axes[i, j]
            ax.imshow(matrix, cmap="Blues")
            for (r, c), number in np.ndenumerate(matrix):
                ax.text(c, r, str(number), ha="center", va="center",
                    color="white" if number > matrix.max()/2 else "black")
            ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="Recorded",
                   title=f"{group}\n{family}")
    fig.tight_layout()
    fig.savefig(directory / "test_confusion_matrices.png", dpi=180)
    plt.close(fig)
    common.write_json(directory / "test_complete.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "selections_sha256": common.sha256(directory / "selections.json"),
        "preferred_by_validation": selections["preferred_by_validation"],
        "test_used_for_selection": False,
    })
    print("\nFinal test comparison (positive class = detected instability):")
    print(table[["group", "family", "average_precision", "pr_auc_trapezoid",
        "f1_instability", "precision_instability", "recall_instability", "fn", "fp"]]
        .round(4).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=common.DEFAULT_PREPARED)
    parser.add_argument("--output", type=Path, default=common.PROJECT / "results/ml/project2_classification/benchmark_v1")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--development-only", action="store_true", help="Stop after saving selections; do not evaluate test")
    parser.add_argument("--describe", action="store_true", help="Print the frozen plan without fitting or creating files")
    args = parser.parse_args()
    if tuple(int(v) for v in version("scikit-learn").split(".")[:2]) < (1, 8):
        parser.error("Use scikit-learn >=1.8 for weighted MLP fitting; the validated version is 1.8.0.")
    if args.bootstrap < 100:
        parser.error("Use at least 100 bootstrap resamples (default 1000).")
    data, groups, preparation = common.load_prepared(args.prepared)
    training = data.loc[data.split.eq("train")].reset_index(drop=True)
    validation = data.loc[data.split.eq("validation")].reset_index(drop=True)
    candidates = settings()
    print("Feature groups:", {k: len(v) for k, v in groups.items()})
    print("Settings per group:", pd.Series([s["family"] for s in candidates]).value_counts().to_dict())
    print("Inner CV: 3 composition-block folds; MLP: 2 seeds in CV, 3-model final ensemble.")
    print("Selection: training CV average precision; threshold: validation F1; final test is reserved.")
    print("Output:", args.output, flush=True)
    if args.describe:
        return
    metadata = {"preparation_sha256": common.sha256(args.prepared / "preparation.json"),
        "prepared_directory": str(args.prepared.resolve()),
        "code_sha256": {p.name: common.sha256(p) for p in [Path(__file__), Path(common.__file__)]},
        "versions": {p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "matplotlib", "threadpoolctl"]},
        "settings": candidates, "feature_groups": groups, "bootstrap": args.bootstrap,
        "mlp_cv_seeds": MLP_CV_SEEDS, "mlp_final_seeds": MLP_FINAL_SEEDS,
        "hyperparameter_metric": "mean grouped training-CV average precision",
        "threshold_metric": "validation instability F1; ties prefer recall then lower threshold",
        "final_fit": "training only; validation is not absorbed after choosing thresholds",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "run.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("This benchmark is already running in another terminal.")
        metadata_file = args.output / "run_metadata.json"
        if metadata_file.exists() and json.loads(metadata_file.read_text()) != metadata:
            raise ValueError("Run inputs changed. Preserve the existing run; do not overwrite it with a changed benchmark.")
        common.write_json(metadata_file, metadata)
        with open(args.output / "invocations.jsonl", "a") as log:
            log.write(json.dumps({"started_utc": datetime.now(timezone.utc).isoformat(),
                "workers": args.workers, "development_only": args.development_only,
                "numerical_threads_per_worker": 1}) + "\n")
        if (args.output / "test_complete.json").exists():
            print("This benchmark already completed its final test. Existing results are unchanged.")
            return
        for name in ["search", "models", "validation_predictions"]:
            (args.output / name).mkdir(exist_ok=True)
        selection_path = args.output / "selections.json"
        started = perf_counter()
        if selection_path.exists():
            selections = json.loads(selection_path.read_text())
            print("Reusing frozen model selections and thresholds.", flush=True)
        else:
            tasks = [(group, columns, setting) for group, columns in groups.items() for setting in candidates]
            results = []
            # Only this outer level runs in parallel. Numerical threads inside
            # each worker are limited to one, avoiding 8 x 8 oversubscription.
            with parallel_config(backend="loky", inner_max_num_threads=1):
                generator = Parallel(n_jobs=args.workers, return_as="generator_unordered")(
                    delayed(evaluate_setting)(group, columns, setting, training, args.output)
                    for group, columns, setting in tasks)
                for number, result in enumerate(generator, 1):
                    results.append(result)
                    print(f"CV settings complete: {number}/{len(tasks)}; {result['group']}; "
                          f"{result['setting']['setting_id']}", flush=True)
            cv_table = pd.DataFrame([{"group": row["group"], **row["setting"],
                "mean_cv_average_precision": row["mean_cv_average_precision"],
                "std_cv_average_precision": row["std_cv_average_precision"],
                "fits_with_convergence_warning": row["fits_with_convergence_warning"],
                "fit_seconds": row["fit_seconds"]} for row in results])
            common.write_csv(args.output / "cv_results.csv", cv_table.sort_values(["group", "family", "setting_id"]))
            selected = []
            for group, columns in groups.items():
                for family in FAMILIES:
                    if family == "majority":
                        setting = {"family": "majority", "setting_id": "majority", "weighted": False}
                    else:
                        eligible = [r for r in results if r["group"] == group and r["setting"]["family"] == family]
                        best = sorted(eligible, key=lambda r: (-r["mean_cv_average_precision"],
                            r["std_cv_average_precision"], r["setting"]["setting_id"]))[0]
                        setting = best["setting"]
                    print(f"Final training fit and validation: {group}; {family}", flush=True)
                    selected.append(fit_selected(group, columns, setting, training, validation, args.output))
            # Choose the preferred comparison before test evaluation. The test
            # table describes all planned alternatives; it does not pick a winner.
            preferred = sorted([s for s in selected if s["family"] != "majority"], key=lambda s:
                (-s["validation_metrics"]["average_precision"],
                 -s["validation_metrics"]["f1_instability"], len(s["columns"]), s["group"], s["family"]))[0]
            selections = {"models": selected,
                "preferred_by_validation": {"group": preferred["group"], "family": preferred["family"]},
                "frozen_utc": datetime.now(timezone.utc).isoformat()}
            common.write_json(selection_path, selections)
            common.write_csv(args.output / "validation_metrics.csv", pd.DataFrame([
                {"group": s["group"], "family": s["family"], "threshold": s["threshold"],
                 **s["validation_metrics"]} for s in selected]))
        warning_count = sum(bool(d["convergence_warnings"]) for s in selections["models"] for d in s["fit_diagnostics"])
        common.write_json(args.output / "fit_quality.json", {
            "final_fits_with_convergence_warning": warning_count,
            "note": "Inspect warnings before interpreting an affected model as fully optimized; warnings and iterations are in selections.json and search files.",
        })
        if warning_count:
            print(f"Convergence warning in {warning_count} final fits; inspect fit_quality.json and selections.json.", flush=True)
        print("Frozen preferred model:", selections["preferred_by_validation"], flush=True)
        if args.development_only:
            print("Stopped before final test; omit --development-only to continue this frozen run.")
            return
        report_test(data, selections, args.output, args.bootstrap)
        print(f"\nThis invocation took {(perf_counter()-started)/60:.1f} minutes.")
        print("Saved:", args.output)
        print("These classifiers reproduce a finite COSMO-RS stability screen, not experimentally established miscibility.")


if __name__ == "__main__":
    main()
