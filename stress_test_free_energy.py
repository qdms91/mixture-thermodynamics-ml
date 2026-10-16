"""Test the saved free-energy models on fresh compositions without retraining.

Run beside mixture_thermodynamics.py and project_config.py.
Example: python stress_test_free_energy.py --model-run results/ml/project1_regression/RUN --workers 8
"""

import os

# Each COSMO-RS worker uses one numerical thread. The saved ML models are
# evaluated in the main process, also with one numerical thread.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import multiprocessing
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.constants import R
from scipy.special import xlogy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from project_config import MOLECULES, PROJECT, surface_paths


COMPOSITION = [f"x_{name}" for name in MOLECULES]
FEATURES = COMPOSITION[:4]
TEMPERATURE = 298.15
MODEL_NAMES = ("ridge", "boosting")
GROUP_NAMES = {
    "off_grid_in_domain": "Off-grid\ncontrols",
    "mild_ood": "Mild\nOOD",
    "broad_ood": "Broad\nOOD",
    "named_ood": "Named\nexamples",
    "water_benzene_binary": "Water–benzene\nbinary",
}
CRS = None


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_version(name):
    """Source checkouts may be importable without installed package metadata."""
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable (source checkout)"


def inside_region(x, lower, upper):
    return bool(np.all(x >= lower - 1e-12) and np.all(x <= upper + 1e-12))


def make_recipes(samples, seed, original_compositions):
    """Choose all recipes before predicting or calculating their energies."""
    rng = np.random.default_rng(seed)
    used = {tuple(np.round(x, 12)) for x in original_compositions}
    rows = []

    def add(group, x, identifier):
        x = np.asarray(x, dtype=float)
        x = x / x.sum()
        key = tuple(np.round(x, 12))
        if key in used:
            return False
        used.add(key)
        violation = max(0.0, float(np.max(0.15 - x)), float(np.max(x - 0.40)))
        rows.append({"recipe_id": identifier, "test_group": group,
                     "max_bound_violation_mol_percent": 100 * violation,
                     **dict(zip(COMPOSITION, x))})
        return True

    # A Dirichlet draw supplies five positive fractions summing to one. Adding
    # the lower bound reserves that amount for each solvent. Rejection then
    # enforces the upper bound and removes the preceding, less extreme region.
    specifications = [
        ("off_grid_in_domain", 0.15, 0.40, None),
        ("mild_ood", 0.10, 0.45, (0.15, 0.40)),
        ("broad_ood", 0.01, 0.90, (0.10, 0.45)),
    ]
    for group, lower, upper, excluded_region in specifications:
        accepted = 0
        for _ in range(1000 * samples):
            x = lower + (1 - 5 * lower) * rng.dirichlet(np.ones(5))
            if x.max() > upper or (excluded_region and inside_region(x, *excluded_region)):
                continue
            accepted += add(group, x, f"{group}_{accepted + 1:04d}")
            if accepted == samples:
                break
        if accepted != samples:
            raise RuntimeError(f"Could not generate {samples} distinct recipes for {group}.")

    for identifier, x in [
        ("example_20_10_5_5_60", [0.20, 0.10, 0.05, 0.05, 0.60]),
        ("example_30_10_5_5_50", [0.30, 0.10, 0.05, 0.05, 0.50]),
    ]:
        if not add("named_ood", x, identifier):
            raise ValueError(f"Named recipe already present: {identifier}")

    # Include the full binary curve, its two pure limits and extra trace points.
    # Zero fractions are intentional: methanol, acetone and acetonitrile are absent.
    trace = np.array([1e-6, 1e-5, 1e-4, 0.001, 0.005])
    fractions = np.unique(np.concatenate([np.linspace(0, 1, 101), trace, 1 - trace]))
    for index, benzene in enumerate(fractions, 1):
        if not add("water_benzene_binary", [1 - benzene, 0, 0, 0, benzene], f"binary_{index:03d}"):
            raise ValueError("A binary recipe was already in the original table.")
    return pd.DataFrame(rows)


def ideal_free_energy(x):
    """Calculate the ideal term, including the limit x*ln(x) = 0 at x = 0."""
    return R * TEMPERATURE / 1000 * np.sum(xlogy(x, x), axis=1)


def initialize_worker(threshold, max_iterations):
    """Build one independent COSMO-RS calculation object per worker."""
    from mixture_thermodynamics import build_model
    global CRS
    CRS = build_model("calculated")
    CRS.par.cosmospace_conv_thresh = threshold
    CRS.par.cosmospace_max_iter = max_iterations


def calculate_reference(recipe):
    """Use the existing single-temperature function to obtain a new reference."""
    from mixture_thermodynamics import mixture_properties
    x = np.array([recipe[column] for column in COMPOSITION])
    started = perf_counter()
    try:
        _, properties = mixture_properties(CRS, x, TEMPERATURE)
        if not np.all(np.isfinite(list(properties.values()))):
            raise RuntimeError("Nonfinite reference energy.")
        return {"recipe_id": recipe["recipe_id"], "status": "ok", "error": "",
                **properties, "reference_seconds": perf_counter() - started}
    except Exception as error:
        return {"recipe_id": recipe["recipe_id"], "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "reference_seconds": perf_counter() - started}


def predict_fixed_models(models, recipes):
    """Make predictions before references are calculated, and time inference."""
    x = recipes[COMPOSITION].to_numpy(dtype=float)
    features = x[:, :4]
    predictions = recipes.copy()
    predictions["analytical_g_ideal_kJ_mol"] = ideal_free_energy(x)
    timings = []
    for name, model in models.items():
        excess = model.predict(features)
        if not np.all(np.isfinite(excess)):
            raise RuntimeError(f"{name} returned nonfinite predictions.")
        predictions[f"{name}_g_excess_kJ_mol"] = excess
        predictions[f"{name}_g_mix_kJ_mol"] = excess + ideal_free_energy(x)

        # Report batch throughput separately from single-recipe latency. Both
        # include ideal-energy reconstruction and exclude loading from disk.
        durations = []
        for _ in range(5):
            started = perf_counter()
            model.predict(features) + ideal_free_energy(x)
            durations.append(perf_counter() - started)
        timings.append({"method": name, "measurement": "batch_seconds_per_recipe",
                        "seconds": float(np.median(durations) / len(x))})
        durations = []
        for i in np.linspace(0, len(x) - 1, 31, dtype=int):
            started = perf_counter()
            model.predict(features[i:i + 1]) + ideal_free_energy(x[i:i + 1])
            durations.append(perf_counter() - started)
        timings.append({"method": name, "measurement": "median_single_recipe_seconds",
                        "seconds": float(np.median(durations))})
    return predictions, timings


def score_predictions(predictions):
    """Report errors separately for each domain, so easy cases cannot hide failures."""
    metrics = []
    for group, frame in predictions.groupby("test_group", sort=False):
        for name in MODEL_NAMES:
            for property_name in ("g_excess", "g_mix"):
                actual = frame[f"{property_name}_kJ_mol"]
                predicted = frame[f"{name}_{property_name}_kJ_mol"]
                errors = np.abs(predicted - actual)
                metrics.append({"test_group": group, "model": name, "property": property_name,
                                "recipes": len(frame),
                                "mae_kJ_mol": mean_absolute_error(actual, predicted),
                                "rmse_kJ_mol": float(np.sqrt(mean_squared_error(actual, predicted))),
                                "max_abs_error_kJ_mol": float(errors.max()),
                                "worst_recipe": frame.loc[errors.idxmax(), "recipe_id"],
                                "r2": r2_score(actual, predicted) if len(frame) >= 3 else np.nan})
    return pd.DataFrame(metrics)


def plot_results(predictions, metrics, output):
    """Show the binary curve and how prediction errors change between groups."""
    binary = predictions.loc[predictions["test_group"] == "water_benzene_binary"].sort_values("x_benzene")
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(binary["x_benzene"], binary["g_mix_kJ_mol"], "k-", label="COSMO-RS homogeneous liquid")
    for name in MODEL_NAMES:
        axes[0].plot(binary["x_benzene"], binary[f"{name}_g_mix_kJ_mol"], "--", label=name)
        axes[1].plot(binary["x_benzene"], binary[f"{name}_g_mix_kJ_mol"] - binary["g_mix_kJ_mol"], label=name)
    axes[0].set_ylabel("Mixing free energy (kJ/mol)")
    axes[1].set(xlabel="Benzene mole fraction (water = 1 − benzene)", ylabel="Prediction − reference (kJ/mol)")
    axes[1].axhline(0, color="gray", linewidth=0.8)
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle("Water–benzene: frozen models, extreme composition extrapolation")
    fig.tight_layout()
    fig.savefig(output / "binary_extrapolation.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    positions = np.arange(len(GROUP_NAMES))
    for index, name in enumerate(MODEL_NAMES):
        rows = metrics.loc[(metrics["model"] == name) & (metrics["property"] == "g_mix")].set_index("test_group").loc[list(GROUP_NAMES)]
        ax.bar(positions + (index - 0.5) * 0.35, rows["mae_kJ_mol"], width=0.35, label=name)
    ax.set_xticks(positions, list(GROUP_NAMES.values()))
    ax.set(yscale="log", ylabel="Mean absolute error (kJ/mol)", title="Separate scores for each test group")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "group_errors.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-run", type=Path, required=True, help="Existing regression run directory")
    parser.add_argument("--data", type=Path, default=PROJECT / "results/ml/thermo_seed88/dataset.csv")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--samples", type=int, default=200, help="Recipes in each of the three random groups")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, help="New directory; defaults to a timestamped extrapolation run")
    args = parser.parse_args()
    if args.workers < 1 or args.samples < 3 or args.seed < 0:
        parser.error("Use positive workers, samples >= 3 and a nonnegative seed.")

    trained = json.loads((args.model_run / "run_metadata.json").read_text())
    if list(MOLECULES) != ["water", "methanol", "acetone", "acetonitrile", "benzene"]:
        raise ValueError("The project molecule order differs from the trained model.")
    if (trained["features"] != FEATURES or trained["molecule_order"] != list(MOLECULES)
            or trained["target"] != "g_excess_kJ_mol" or trained["temperature_K"] != TEMPERATURE):
        raise ValueError("The saved model inputs, target or temperature do not match this experiment.")
    if version("scikit-learn") != trained["versions"]["scikit-learn"]:
        raise ValueError("Use the same scikit-learn version as the saved model: " + trained["versions"]["scikit-learn"])
    if sha256(args.data) != trained["data_sha256"]:
        raise ValueError("The supplied source table differs from the recorded training campaign.")
    calculation = trained["calculation_metadata"]
    if calculation["source"] != "calculated":
        raise ValueError("This experiment expects the project's calculated surfaces.")
    surfaces = surface_paths("calculated")
    for name, path in surfaces.items():
        if sha256(path) != calculation["sha256"][f"surface:{name}"]:
            raise ValueError(f"Surface changed since the training campaign: {name}")
    # Checking the property function keeps the reference calculations consistent
    # with the label-generating code. This check also detects edits to comments.
    if sha256(PROJECT / "mixture_thermodynamics.py") != calculation["sha256"]["mixture_thermodynamics.py"]:
        raise ValueError("mixture_thermodynamics.py changed since label generation; inspect the change before this benchmark.")

    original = pd.read_csv(args.data, usecols=COMPOSITION)[COMPOSITION].to_numpy(dtype=float)
    recipes = make_recipes(args.samples, args.seed, original)
    output = args.output or PROJECT / "results/ml/project1_extrapolation" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    model_paths = {name: args.model_run / f"{name}_model.joblib" for name in MODEL_NAMES}
    model_hashes = {name: sha256(path) for name, path in model_paths.items()}
    models = {name: joblib.load(path) for name, path in model_paths.items()}
    if any(model.n_features_in_ != len(FEATURES) for model in models.values()):
        raise ValueError("A model has an unexpected number of input features.")
    recipes.to_csv(output / "recipes.csv", index=False)
    predictions, timings = predict_fixed_models(models, recipes)
    predictions.to_csv(output / "predictions_before_cosmors.csv", index=False)
    metadata = {
        "model_run": str(args.model_run.resolve()), "model_sha256": model_hashes,
        "source_data_sha256": trained["data_sha256"],
        "script_sha256": sha256(Path(__file__)), "temperature_K": TEMPERATURE,
        "molecule_order": list(MOLECULES), "features": FEATURES,
        "seed": args.seed, "samples_per_random_group": args.samples, "workers": args.workers,
        "group_counts": {str(k): int(v) for k, v in recipes["test_group"].value_counts().items()},
        "fit_performed": False, "original_recipe_overlap": 0,
        "calculation_metadata": calculation,
        "versions": {name: package_version(name) for name in ("numpy", "scipy", "pandas", "scikit-learn", "joblib", "opencosmorspy")},
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Saved the recipes and frozen-model predictions before reference calculations.", flush=True)
    print(recipes.groupby("test_group", sort=False).size().to_string(), flush=True)
    print("Output:", output.resolve(), flush=True)

    started = perf_counter()
    references = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize_worker,
                             initargs=(calculation["cosmospace_convergence"], calculation["cosmospace_max_iter"])) as pool:
        for count, row in enumerate(pool.map(calculate_reference, recipes.to_dict("records")), 1):
            references.append(row)
            if count % 25 == 0 or count == len(recipes):
                pd.DataFrame(references).to_csv(output / "references.csv", index=False)
                print(f"COSMO-RS references: {count}/{len(recipes)}", flush=True)
    reference_wall_seconds = perf_counter() - started
    references = pd.DataFrame(references)
    failures = int((references["status"] != "ok").sum())
    if failures:
        raise RuntimeError(f"{failures} references failed; inspect references.csv. No aggregate scores were produced.")
    predictions = predictions.merge(references, on="recipe_id", validate="one_to_one")
    if not np.allclose(predictions["analytical_g_ideal_kJ_mol"], predictions["g_ideal_kJ_mol"], rtol=0, atol=1e-10):
        raise RuntimeError("Reference and analytical ideal terms disagree.")
    for name in MODEL_NAMES:
        predictions[f"{name}_error_g_mix_kJ_mol"] = predictions[f"{name}_g_mix_kJ_mol"] - predictions["g_mix_kJ_mol"]
        if sha256(model_paths[name]) != model_hashes[name]:
            raise RuntimeError(f"The source {name} model file changed during this run.")
    predictions.to_csv(output / "predictions.csv", index=False)
    metrics = score_predictions(predictions)
    metrics.to_csv(output / "metrics_by_group.csv", index=False)
    timings.append({"method": "cosmors", "measurement": "median_single_recipe_seconds",
                    "seconds": float(references["reference_seconds"].median())})
    timings.append({"method": "cosmors", "measurement": "reference_campaign_wall_seconds",
                    "seconds": reference_wall_seconds})
    pd.DataFrame(timings).to_csv(output / "timings.csv", index=False)
    plot_results(predictions, metrics, output)
    print("\nTotal mixing-energy prediction errors:")
    print(metrics.loc[metrics["property"] == "g_mix", ["test_group", "model", "recipes", "mae_kJ_mol", "max_abs_error_kJ_mol"]].to_string(index=False))
    print(f"\nReference campaign: {reference_wall_seconds:.1f} seconds. Saved: {output.resolve()}")
    print("The original models are unchanged. These are homogeneous-branch energy tests, not phase-stability tests.")


if __name__ == "__main__":
    main()
