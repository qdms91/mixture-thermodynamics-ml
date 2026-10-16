"""Learn excess free energy from composition, then recover total mixing energy.

Run from cosmo_rs_1: python train_free_energy_models.py --workers 8
The existing train_pool / validation / test assignments are preserved.
"""

import os

# Independent fits run in separate processes. Each process uses one numerical
# thread, so eight workers do not each try to launch another eight threads.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import joblib
from joblib import Parallel, delayed, parallel_config
import matplotlib
matplotlib.use("Agg")  # Save figures without opening windows.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.constants import R
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


PROJECT = Path(__file__).resolve().parent
MOLECULES = ["water", "methanol", "acetone", "acetonitrile", "benzene"]
COMPOSITION = [f"x_{name}" for name in MOLECULES]
FEATURES = COMPOSITION[:4]  # x_benzene = 1 - sum(the other four fractions).
TARGET = "g_excess_kJ_mol"
TEMPERATURE = 298.15


def ideal_free_energy(frame):
    """Calculate the known ideal term from composition, in kJ/mol of mixture."""
    x = frame[COMPOSITION].to_numpy(dtype=float)
    return R * TEMPERATURE / 1000 * np.sum(x * np.log(x), axis=1)


def load_data(path):
    """Check the table and its original split before fitting anything."""
    data = pd.read_csv(path)
    required = ["recipe_id", "block_id", "split", "temperature_K", TARGET,
                "g_ideal_kJ_mol", "g_mix_kJ_mol", *COMPOSITION]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if data[required].isna().any().any():
        raise ValueError("Required columns contain missing values.")
    numeric = ["temperature_K", TARGET, "g_ideal_kJ_mol", "g_mix_kJ_mol", *COMPOSITION]
    if not np.isfinite(data[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Required numeric columns contain nonfinite values.")
    if data["recipe_id"].duplicated().any() or data.duplicated(COMPOSITION).any():
        raise ValueError("A recipe occurs more than once; inspect duplicates first.")
    if set(data["split"]) != {"train_pool", "validation", "test"}:
        raise ValueError("Expected nonempty train_pool, validation and test splits.")
    if data.groupby("block_id")["split"].nunique().max() != 1:
        raise ValueError("A composition block appears in more than one split.")
    x = data[COMPOSITION].to_numpy(dtype=float)
    if np.any(x < 0.15 - 1e-12) or np.any(x > 0.40 + 1e-12):
        raise ValueError("This experiment uses mole fractions between 0.15 and 0.40.")
    if not np.allclose(x.sum(axis=1), 1, rtol=0, atol=1e-12):
        raise ValueError("Mole fractions must sum to one.")
    if not np.allclose(data["temperature_K"], TEMPERATURE, rtol=0, atol=1e-10):
        raise ValueError("This experiment uses 298.15 K only.")
    ideal = ideal_free_energy(data)
    if not np.allclose(ideal, data["g_ideal_kJ_mol"], rtol=0, atol=1e-8):
        raise ValueError("The recorded ideal free energy does not match RT*sum(x*ln(x)).")
    if not np.allclose(ideal + data[TARGET], data["g_mix_kJ_mol"], rtol=0, atol=1e-8):
        raise ValueError("The recorded total does not equal ideal plus excess energy.")
    return data.sort_values("recipe_id").reset_index(drop=True)


def model_settings():
    """Define a small comparison before looking at validation or test errors."""
    settings = [{"id": "mean_excess", "family": "baseline"}]
    for degree in (2, 3, 4):
        for alpha in (1e-4, 1e-2, 1.0):
            settings.append({"id": f"ridge_d{degree}_a{alpha:g}", "family": "ridge",
                             "degree": degree, "alpha": alpha})
    for leaves in (7, 15, 31):
        for rate in (0.05, 0.1):
            settings.append({"id": f"boost_l{leaves}_r{rate:g}", "family": "boosting",
                             "max_leaf_nodes": leaves, "learning_rate": rate})
    return settings


def make_model(setting):
    """Construct one unfitted model. Only the four composition columns enter it."""
    if setting["family"] == "baseline":
        return DummyRegressor(strategy="mean")
    if setting["family"] == "ridge":
        # PolynomialFeatures adds powers and products of the mole fractions.
        # StandardScaler learns scales from the training subset only. Ridge
        # fits coefficients while penalizing large coefficients through alpha.
        return make_pipeline(
            PolynomialFeatures(degree=setting["degree"], include_bias=False),
            StandardScaler(),
            Ridge(alpha=setting["alpha"], solver="svd"),
        )
    # Boosting builds successive trees that improve the previous predictions.
    # Disabling automatic early stopping keeps our supplied validation split
    # as the only validation set; the number of trees is fixed in advance.
    return HistGradientBoostingRegressor(
        max_leaf_nodes=setting["max_leaf_nodes"],
        learning_rate=setting["learning_rate"],
        max_iter=300, min_samples_leaf=10, l2_regularization=0.0,
        early_stopping=False, random_state=88,
    )


def fit_candidate(setting, x_train, y_train, x_validation, y_validation, seed):
    """Fit on one subset and measure its training and validation errors."""
    model = make_model(setting)
    started = perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = perf_counter() - started
    predicted_train = model.predict(x_train)
    predicted_validation = model.predict(x_validation)
    return {
        "setting_id": setting["id"], "family": setting["family"],
        "training_recipes": len(y_train), "subset_seed": seed,
        "train_mae_kJ_mol": mean_absolute_error(y_train, predicted_train),
        "validation_mae_kJ_mol": mean_absolute_error(y_validation, predicted_validation),
        "validation_rmse_kJ_mol": float(np.sqrt(mean_squared_error(y_validation, predicted_validation))),
        "fit_seconds": fit_seconds,
    }


def select_settings(results):
    """At each training size, select settings by mean validation MAE over seeds."""
    grouped = results.groupby(["training_recipes", "family", "setting_id"], as_index=False).agg(
        repetitions=("subset_seed", "count"),
        train_mae_kJ_mol=("train_mae_kJ_mol", "mean"),
        validation_mae_kJ_mol=("validation_mae_kJ_mol", "mean"),
        validation_mae_std_kJ_mol=("validation_mae_kJ_mol", "std"),
    )
    # The full training pool is fitted once: repeating it would reuse the same
    # examples. Its missing standard deviation remains NaN, rather than zero.
    return grouped.sort_values(["validation_mae_kJ_mol", "setting_id"]).drop_duplicates(
        ["training_recipes", "family"]
    ).sort_values(["training_recipes", "family"]).reset_index(drop=True)


def predict_and_score(model, frame, family):
    """Recover total mixing energy and compare it with held-out COSMO-RS values."""
    predicted_excess = model.predict(frame[FEATURES].to_numpy(dtype=float))
    predicted_mix = ideal_free_energy(frame) + predicted_excess
    rows = []
    for property_name, actual, predicted in (
        ("g_excess", frame[TARGET], predicted_excess),
        ("g_mix", frame["g_mix_kJ_mol"], predicted_mix),
    ):
        rows.append({"model": family, "property": property_name,
                     "recipes": len(frame),
                     "mae_kJ_mol": mean_absolute_error(actual, predicted),
                     "rmse_kJ_mol": float(np.sqrt(mean_squared_error(actual, predicted))),
                     "max_abs_error_kJ_mol": float(np.max(np.abs(actual - predicted))),
                     "r2": r2_score(actual, predicted)})
    predictions = frame[["recipe_id", "block_id", "split", *COMPOSITION,
                         TARGET, "g_mix_kJ_mol"]].copy()
    predictions.insert(0, "model", family)
    predictions["predicted_g_excess_kJ_mol"] = predicted_excess
    predictions["predicted_g_mix_kJ_mol"] = predicted_mix
    predictions["error_g_mix_kJ_mol"] = predicted_mix - frame["g_mix_kJ_mol"].to_numpy()
    return rows, predictions


def plot_results(curves, predictions, directory):
    """Save learning curves and held-out test predictions without opening a GUI."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for family in ("baseline", "ridge", "boosting"):
        rows = curves.loc[curves["family"] == family]
        line, = ax.plot(rows["training_recipes"], rows["validation_mae_kJ_mol"], "o-", label=family)
        ax.plot(rows["training_recipes"], rows["train_mae_kJ_mol"], "--", color=line.get_color(), alpha=0.65)
    ax.set(xscale="log", yscale="log", xlabel="Number of training recipes",
           ylabel="MAE (kJ/mol)", title="Solid: validation error; dashed: training error")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(directory / "learning_curves.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, family in zip(axes, ("ridge", "boosting")):
        rows = predictions.loc[predictions["model"] == family]
        actual = rows["g_mix_kJ_mol"]
        predicted = rows["predicted_g_mix_kJ_mol"]
        lo, hi = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        ax.scatter(actual, predicted, s=8, alpha=0.3)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set(xlabel="COSMO-RS mixing free energy (kJ/mol)",
               ylabel="Predicted mixing free energy (kJ/mol)", title=family)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
    fig.suptitle("Test recipes: excluded from training and model selection")
    fig.tight_layout()
    fig.savefig(directory / "test_predictions.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "results/ml/thermo_seed88/dataset.csv")
    parser.add_argument("--output", type=Path, help="New output directory; default: a timestamped run")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 300, 1000, 3000, 10000],
                        help="Training subset sizes; the complete training pool is always added")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 88, 123],
                        help="Seeds for nested training subsets, shared by every model")
    args = parser.parse_args()
    if args.workers < 1 or min(args.sizes) < 20 or min(args.seeds) < 0:
        parser.error("Use positive workers, training sizes >= 20 and nonnegative seeds.")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Subset seeds must be distinct.")

    started = perf_counter()
    data = load_data(args.data)
    train = data.loc[data["split"] == "train_pool"].reset_index(drop=True)
    validation = data.loc[data["split"] == "validation"].reset_index(drop=True)
    sizes = sorted({n for n in args.sizes if n < len(train)} | {len(train)})
    output = args.output or PROJECT / "results/ml/project1_regression" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)  # Preserve earlier experiment results.
    settings = model_settings()
    by_id = {setting["id"]: setting for setting in settings}

    metadata = {
        "data_file": str(args.data.resolve()),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "temperature_K": TEMPERATURE, "molecule_order": MOLECULES,
        "features": FEATURES, "target": TARGET, "implied_fraction": "x_benzene",
        "split_counts": {str(k): int(v) for k, v in data["split"].value_counts().items()},
        "training_sizes": sizes, "subset_seeds": args.seeds, "workers": args.workers,
        "settings": settings, "stability_filter": "none",
        "selection_rule": "Mean validation MAE; select separately by family and training size",
        "test_rule": "Evaluate only the full-training selections, after all selections are saved",
        "versions": {name: version(name) for name in
                     ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "matplotlib")},
    }
    # Keep the calculation provenance beside the ML settings when available.
    source_metadata = args.data.parent / "metadata.json"
    if source_metadata.is_file():
        metadata["calculation_metadata"] = json.loads(source_metadata.read_text())
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Dataset:", args.data.resolve(), flush=True)
    print("Split sizes:", metadata["split_counts"], flush=True)
    print("Inputs:", ", ".join(FEATURES), flush=True)
    print("Target:", TARGET, "(total = predicted excess + exact ideal)", flush=True)
    print(f"Training sizes: {sizes}; workers: {args.workers}", flush=True)
    print("Output:", output.resolve(), flush=True)

    # For each seed, the first 100 examples belong to the first 300, and so on.
    # Every model receives exactly the same subset at a given size and seed.
    orders = {seed: np.random.default_rng(seed).permutation(len(train)) for seed in args.seeds}
    pd.concat([pd.DataFrame({"subset_seed": seed, "rank": np.arange(1, len(train) + 1),
                            "recipe_id": train.iloc[order]["recipe_id"].to_numpy()})
               for seed, order in orders.items()], ignore_index=True).to_csv(
                   output / "training_order.csv", index=False)
    x_validation = validation[FEATURES].to_numpy(dtype=float)
    y_validation = validation[TARGET].to_numpy(dtype=float)
    records = []

    with parallel_config(backend="loky", n_jobs=args.workers, inner_max_num_threads=1):
        for size in sizes:
            seeds = args.seeds if size < len(train) else [args.seeds[0]]
            tasks = []
            for seed in seeds:
                indices = orders[seed][:size] if size < len(train) else np.arange(len(train))
                subset = train.iloc[indices]
                for setting in settings:
                    tasks.append(delayed(fit_candidate)(
                        setting, subset[FEATURES].to_numpy(dtype=float),
                        subset[TARGET].to_numpy(dtype=float), x_validation, y_validation, seed))
            print(f"\nTraining recipes: {size}; fitting {len(tasks)} candidates...", flush=True)
            records.extend(Parallel()(tasks))
            results = pd.DataFrame(records)
            results.to_csv(output / "validation_search.csv", index=False)
            curves = select_settings(results)
            print(curves.loc[curves["training_recipes"] == size,
                             ["family", "setting_id", "validation_mae_kJ_mol"]].to_string(index=False), flush=True)

    curves.to_csv(output / "learning_curves.csv", index=False)
    full = curves.loc[curves["training_recipes"] == len(train)]
    chosen = {row.family: by_id[row.setting_id] for row in full.itertuples()}
    winner = full.loc[full["family"] != "baseline"].sort_values("validation_mae_kJ_mol").iloc[0]["family"]
    (output / "selected_settings.json").write_text(json.dumps(
        {"selected_by_validation": chosen, "preferred_family_by_validation": winner}, indent=2) + "\n")

    # Selection is finished. Only now do we calculate test predictions. We keep
    # validation separate and refit each selected setting on train_pool only.
    # Test errors are reported for both families and the mean-excess baseline.
    print("\nSelections saved. Evaluating the reserved test recipes...", flush=True)
    test = data.loc[data["split"] == "test"].reset_index(drop=True)
    all_metrics, all_predictions = [], []
    for family, setting in chosen.items():
        model = make_model(setting)
        model.fit(train[FEATURES].to_numpy(dtype=float), train[TARGET].to_numpy(dtype=float))
        joblib.dump(model, output / f"{family}_model.joblib")
        for split_name, frame in (("validation", validation), ("test", test)):
            metrics, predictions = predict_and_score(model, frame, family)
            all_metrics.extend({"split": split_name, **row} for row in metrics)
            all_predictions.append(predictions)
    metrics = pd.DataFrame(all_metrics)
    predictions = pd.concat(all_predictions, ignore_index=True)
    for split_name in ("validation", "test"):
        metrics.loc[metrics["split"] == split_name].to_csv(output / f"{split_name}_metrics.csv", index=False)
        predictions.loc[predictions["split"] == split_name].to_csv(output / f"{split_name}_predictions.csv", index=False)
    plot_results(curves, predictions.loc[predictions["split"] == "test"], output)
    print("\nFinal test performance for total mixing free energy:")
    print(metrics.loc[(metrics["split"] == "test") & (metrics["property"] == "g_mix"),
                      ["model", "mae_kJ_mol", "rmse_kJ_mol", "max_abs_error_kJ_mol", "r2"]].to_string(index=False))
    print(f"\nPreferred family, selected using validation: {winner}")
    print(f"Finished in {perf_counter() - started:.1f} s. Saved: {output.resolve()}")
    print("These errors measure agreement with COSMO-RS, not accuracy against experiment.")


if __name__ == "__main__":
    main()
