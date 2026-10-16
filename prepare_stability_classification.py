"""Audit the completed campaign, define feature groups and freeze Project 2 splits."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.constants import R
from sklearn.model_selection import StratifiedGroupKFold
import sklearn

import project2_common as common


def audit(data, recipes, metadata):
    """Check the stored search evidence and thermodynamic identities."""
    if len(data) != metadata["total"] or len(recipes) != len(data):
        raise ValueError("The collection is incomplete or does not match its manifest.")
    if data.recipe_id.duplicated().any() or data.duplicated(common.X_COLUMNS).any():
        raise ValueError("Duplicate recipes were found.")
    left = data.sort_values("recipe_id").reset_index(drop=True)
    right = recipes.sort_values("recipe_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(left[right.columns], right, check_dtype=False)
    x = data[common.X_COLUMNS].to_numpy(float)
    if (not np.isfinite(x).all() or not np.allclose(x.sum(1), 1, atol=1e-12, rtol=0)
            or np.any(x < .15 - 1e-12) or np.any(x > .40 + 1e-12)):
        raise ValueError("Invalid composition or a recipe outside the intended domain.")
    protocol = metadata["calculation_metadata"]
    if (protocol["molecule_order"] != common.MOLECULES or protocol["starts"] != 40
            or not data.starts.eq(40).all() or not data.temperature_K.eq(298.15).all()):
        raise ValueError("Unexpected molecule order, temperature or search protocol.")
    if not set(data.label_status).issubset({"instability_found", "no_instability_found", "needs_review"}):
        raise ValueError("Unexpected label status.")
    usable = data.label_status.ne("needs_review")
    positive = data.label_status.eq("instability_found")
    negative = data.label_status.eq("no_instability_found")
    tolerance = protocol["tpd_tolerance"]
    if (not data.loc[positive, "instability_label"].eq(1).all()
            or not data.loc[negative, "instability_label"].eq(0).all()
            or not data.loc[positive, "lowest_tpd_over_RT"].lt(-tolerance).all()
            or not data.loc[positive, "split_change_kJ_mol"].lt(-1e-10).all()
            or not data.loc[negative, "lowest_tpd_over_RT"].ge(-tolerance).all()
            or not data.loc[negative, "nonconverged_searches"].eq(0).all()):
        raise ValueError("Stored labels disagree with the search evidence.")
    w = data.loc[positive, [f"best_trial_x_{n}" for n in common.MOLECULES]].to_numpy(float)
    remaining = data.loc[positive, [f"remaining_x_{n}" for n in common.MOLECULES]].to_numpy(float)
    fraction = data.loc[positive, "trial_fraction"].to_numpy(float)[:, None]
    if (not np.isfinite(w).all() or not np.isfinite(remaining).all()
            or np.any(w < 0) or np.any(remaining < 0)
            or not np.all((fraction > 0) & (fraction < 1))
            or not np.allclose(w.sum(1), 1, atol=1e-10)
            or not np.allclose(remaining.sum(1), 1, atol=1e-10)
            or not np.allclose(fraction*w + (1-fraction)*remaining, x[positive], atol=1e-10, rtol=0)):
        raise ValueError("A recorded positive label fails its mass-balance check.")
    numeric = [*common.THERMO, *common.ACTIVITY, "g_mix_kJ_mol", "g_ideal_kJ_mol",
               "h_step_change_kJ_mol", "feature_seconds", "stability_seconds"]
    if not np.isfinite(data.loc[usable, numeric].to_numpy(float)).all():
        raise ValueError("A usable recipe contains nonfinite features or timing diagnostics.")
    ideal = R * 298.15 / 1000 * np.sum(x * np.log(x), axis=1)
    excess = R * 298.15 / 1000 * np.sum(x * data[common.ACTIVITY].to_numpy(), axis=1)
    for observed, expected in [(data.g_ideal_kJ_mol, ideal), (data.g_excess_kJ_mol, excess),
        (data.g_mix_kJ_mol, data.g_ideal_kJ_mol + data.g_excess_kJ_mol),
        (data.g_mix_kJ_mol, data.h_mix_kJ_mol - 298.15 * data.s_mix_J_mol_K / 1000)]:
        if not np.allclose(observed[usable], np.asarray(expected)[usable], rtol=0, atol=1e-8):
            raise ValueError("Thermodynamic identities do not agree with the stored features.")
    if data.loc[usable, "h_step_change_kJ_mol"].max() > .005:
        raise ValueError("Temperature-step sensitivity exceeds 0.005 kJ/mol; inspect it first.")
    return usable


def assign_splits(data):
    """Keep test fold 0; select a validation fold using class counts only."""
    y = data.instability_label.to_numpy(int)
    blocks = data.block_id.to_numpy()
    development, test = next(StratifiedGroupKFold(4, shuffle=True, random_state=2026).split(data, y, blocks))
    # The originally suggested validation fold 0 has only three positives.
    # Choose among the same five grouped folds by closeness to 20% of each
    # development class. No model scores or thermodynamic values enter this rule.
    choices = []
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=2026)
    totals = np.bincount(y[development], minlength=2)
    for fold, (train, validation) in enumerate(splitter.split(data.iloc[development], y[development], blocks[development])):
        counts = np.bincount(y[development][validation], minlength=2)
        balance_loss = float(np.sum((counts / totals - .2) ** 2))
        choices.append((balance_loss, fold, train, validation))
    _, selected_fold, train, validation = min(choices, key=lambda v: (v[0], v[1]))
    result = data.copy()
    result["split"] = "train"
    result.loc[result.index[test], "split"] = "test"
    result.loc[result.index[development[validation]], "split"] = "validation"
    result["cv_fold"] = -1
    training = np.flatnonzero(result.split.eq("train"))
    inner = StratifiedGroupKFold(3, shuffle=True, random_state=2027)
    for fold, (_, held) in enumerate(inner.split(result.iloc[training], y[training], blocks[training])):
        result.loc[result.index[training[held]], "cv_fold"] = fold
    if result.groupby("block_id").split.nunique().max() != 1:
        raise ValueError("A block crosses the train/validation/test boundary.")
    if result.loc[training].groupby("block_id").cv_fold.nunique().max() != 1:
        raise ValueError("A training block crosses the inner cross-validation boundary.")
    counts = pd.crosstab(result.split, result.instability_label)
    if counts.min().min() < 15:
        raise ValueError(f"Insufficient minority coverage; inspect before fitting:\n{counts}")
    return result, selected_fold, [dict(fold=int(v[1]), class_balance_loss=v[0]) for v in choices]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=common.DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=common.DEFAULT_PREPARED)
    args = parser.parse_args()
    source_files = ["dataset.csv", "recipes.csv", "metadata.json"]
    inputs = {name: common.sha256(args.source / name) for name in source_files}
    code = {p.name: common.sha256(p) for p in [Path(__file__), Path(common.__file__)]}
    if args.output.exists():
        _, _, old = common.load_prepared(args.output)
        if old["source_sha256"] != inputs or old["code_sha256"] != code:
            raise ValueError("Inputs changed. Preserve the existing frozen preparation; inspect before making a new version.")
        print("Reusing the existing frozen preparation:", args.output)
        print(pd.read_csv(args.output / "split_counts.csv").to_string(index=False))
        return
    metadata = json.loads((args.source / "metadata.json").read_text())
    if inputs["recipes.csv"] != metadata["manifest_sha256"]:
        raise ValueError("The recipe manifest does not match the campaign hash.")
    data = pd.read_csv(args.source / "dataset.csv").sort_values("recipe_number").reset_index(drop=True)
    recipes = pd.read_csv(args.source / "recipes.csv")
    usable = audit(data, recipes, metadata)
    clean = common.add_descriptors(data.loc[usable].reset_index(drop=True))
    clean, selected_fold, candidates = assign_splits(clean)
    groups = common.feature_groups()
    columns = list(dict.fromkeys(c for group in groups.values() for c in group))
    kept = ["recipe_id", "recipe_number", "block_id", "split", "cv_fold", "sampling_group",
            "data_origin", "original_split", "instability_label", *common.X_COLUMNS,
            *columns, "feature_seconds", "stability_seconds"]
    clean = clean[list(dict.fromkeys(kept))]
    if not np.isfinite(clean[columns].to_numpy()).all():
        raise ValueError("Nonfinite derived predictors.")
    args.output.mkdir(parents=True, exist_ok=False)
    common.write_csv(args.output / "classification_data.csv", clean)
    common.write_csv(args.output / "frozen_split.csv", clean[["recipe_id", "block_id", "split", "cv_fold", "instability_label"]])
    common.write_csv(args.output / "excluded_needs_review.csv", data.loc[~usable])
    counts = clean.groupby(["split", "instability_label"]).size().unstack(fill_value=0)
    counts.columns = ["no_instability_found", "instability_found"]
    counts["total"] = counts.sum(axis=1)
    counts["positive_fraction"] = counts.instability_found / counts.total
    common.write_csv(args.output / "split_counts.csv", counts.reset_index())
    common.write_json(args.output / "feature_groups.json", groups)
    common.write_json(args.output / "descriptor_constants.json", {
        "molecule_order": common.MOLECULES, "elements": common.ELEMENTS,
        "atoms": common.ATOMS.tolist(), "atomic_mass_g_mol": common.ATOMIC_MASS.tolist(),
        "molar_mass_g_mol": common.MOLAR_MASS.tolist(),
        "pure_density_g_mL": common.DENSITY.tolist(), "density_temperature_K": 298.15,
        "density_sources": common.DENSITY_SOURCES,
        "density_definition": "sum(x*M) / sum(x*M/rho); additive pure-liquid volumes, not real mixture density",
    })
    output_files = [p for p in args.output.iterdir() if p.is_file()]
    common.write_json(args.output / "preparation.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "sklearn_version": sklearn.__version__,
        "source_sha256": inputs, "code_sha256": code,
        "output_sha256": {p.name: common.sha256(p) for p in output_files},
        "source_metadata": metadata, "selected_validation_fold": int(selected_fold),
        "validation_fold_balance": candidates, "test_fold": 0, "split_seed": 2026,
        "inner_folds": 3, "inner_seed": 2027,
        "label_definition": "1: detected instability with negative explicit split; 0: none found by the recorded 40-start search",
        "feature_policy": "Only explicit feed-feature whitelists; no TPD, split, convergence, timing or recipe identifiers as predictors",
    })
    print(counts.to_string())
    print("Validation fold selected by class counts:", selected_fold)
    print("Feature groups:", {name: len(value) for name, value in groups.items()})
    print("Frozen preparation:", args.output)


if __name__ == "__main__":
    main()
