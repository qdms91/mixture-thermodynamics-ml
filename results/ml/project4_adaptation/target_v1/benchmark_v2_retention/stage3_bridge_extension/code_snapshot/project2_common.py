"""Shared features, metrics and file checks for the stability classifiers."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, auc, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

MOLECULES = ["water", "methanol", "acetone", "acetonitrile", "benzene"]
X_COLUMNS = [f"x_{name}" for name in MOLECULES]
COMPOSITION = X_COLUMNS[:4]  # Benzene is 1 minus the sum of these four fractions.
THERMO = ["g_excess_kJ_mol", "h_mix_kJ_mol", "s_mix_J_mol_K"]
ACTIVITY = [f"ln_gamma_{name}" for name in MOLECULES]
PROJECT = Path(__file__).resolve().parent
DEFAULT_SOURCE = PROJECT / "results/ml/project2_stability/total2000_sample2026_search88_starts40"
DEFAULT_PREPARED = PROJECT / "results/ml/project2_classification/prepared_seed2026"

# Rows follow MOLECULES; columns are H, C, N, O atoms per molecule.
ATOMS = np.array([[2, 0, 0, 1], [4, 1, 0, 1], [6, 3, 0, 1],
                  [3, 2, 1, 0], [6, 6, 0, 0]], dtype=float)
ELEMENTS = ["H", "C", "N", "O"]
ATOMIC_MASS = np.array([1.008, 12.011, 14.007, 15.999])
MOLAR_MASS = ATOMS @ ATOMIC_MASS

# Approximate pure-liquid densities near atmospheric pressure at 298.15 K.
# These constants define a volume-additive descriptor. They are not measured
# densities of our five-solvent mixtures, and do not include excess volume.
DENSITY = np.array([0.9970, 0.7866, 0.7845, 0.7767, 0.8736])  # g/mL
DENSITY_SOURCES = {
    "water_methanol": "https://doi.org/10.1002/cite.202000058 (Table 2)",
    "acetone": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5637464/ (solvent-property table)",
    "acetonitrile": "https://doi.org/10.3389/fphy.2023.1208382 (Table 2; 0.78 bar)",
    "benzene": "https://doi.org/10.1021/je900468u (Table 1)",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """Replace a result file only after its complete JSON has been written."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    data.to_csv(temporary, index=False)
    temporary.replace(path)


def add_descriptors(data):
    """Derive molecular descriptors from the feed composition, without labels."""
    result = data.copy()
    x = result[X_COLUMNS].to_numpy(float)
    result["mean_molar_mass_g_mol"] = x @ MOLAR_MASS
    result["mean_atom_count"] = x @ ATOMS.sum(axis=1)
    for j, element in enumerate(ELEMENTS):
        result[f"atom_moles_{element}_per_mol_mixture"] = x @ ATOMS[:, j]
    for j, name in enumerate(MOLECULES):
        result[f"atom_moles_from_{name}_per_mol_mixture"] = x[:, j] * ATOMS[j].sum()
    result["volume_additive_molar_volume_mL_mol"] = x @ (MOLAR_MASS / DENSITY)
    result["volume_additive_density_g_mL"] = (
        result["mean_molar_mass_g_mol"] / result["volume_additive_molar_volume_mL_mol"]
    )
    return result


def feature_groups():
    """Use explicit lists: search diagnostics can never enter the predictors."""
    molecular = ["mean_molar_mass_g_mol", "mean_atom_count"]
    molecular += [f"atom_moles_{e}_per_mol_mixture" for e in ELEMENTS]
    molecular += [f"atom_moles_from_{n}_per_mol_mixture" for n in MOLECULES]
    molecular += ["volume_additive_molar_volume_mL_mol", "volume_additive_density_g_mL"]
    return {
        "a_composition": COMPOSITION,
        "b_molecular": COMPOSITION + molecular,
        "c_all": COMPOSITION + molecular + THERMO + ACTIVITY,
        "d_thermodynamics": COMPOSITION + THERMO,
        "e_activity": COMPOSITION + ACTIVITY,
    }


def metrics(y, score, threshold):
    """Class 1 is a detected instability. Scores need not be probabilities."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    predicted = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    precision, recall, _ = precision_recall_curve(y, score)
    two_classes = len(np.unique(y)) == 2
    return {
        "recipes": len(y), "positives": int(y.sum()), "prevalence": float(y.mean()),
        "precision_instability": float(precision_score(y, predicted, zero_division=0)),
        "recall_instability": float(recall_score(y, predicted, zero_division=0)),
        "f1_instability": float(f1_score(y, predicted, zero_division=0)),
        "precision_no_instability": float(precision_score(y, predicted, pos_label=0, zero_division=0)),
        "recall_no_instability": float(recall_score(y, predicted, pos_label=0, zero_division=0)),
        "f1_no_instability": float(f1_score(y, predicted, pos_label=0, zero_division=0)),
        "f1_macro": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "average_precision": float(average_precision_score(y, score)) if y.sum() else None,
        "pr_auc_trapezoid": float(auc(recall, precision)) if y.sum() else None,
        "roc_auc": float(roc_auc_score(y, score)) if two_classes else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)) if two_classes else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def choose_threshold(y, score):
    """Maximize validation F1; ties prefer recall, then the lower threshold."""
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    eligible = np.flatnonzero(np.isclose(f1, f1.max(), atol=1e-12, rtol=0))
    index = sorted(eligible, key=lambda i: (-recall[i], thresholds[i]))[0]
    return float(thresholds[index])


def load_prepared(directory):
    """Require the frozen split, features and data to match their saved hashes."""
    directory = Path(directory)
    metadata = json.loads((directory / "preparation.json").read_text())
    for filename, expected in metadata["output_sha256"].items():
        if sha256(directory / filename) != expected:
            raise ValueError(f"Prepared file changed: {filename}. Do not edit a frozen dataset.")
    data = pd.read_csv(directory / "classification_data.csv")
    groups = json.loads((directory / "feature_groups.json").read_text())
    if groups != feature_groups():
        raise ValueError("Feature definitions changed after the split was frozen.")
    if data.groupby("block_id")["split"].nunique().max() != 1:
        raise ValueError("A composition block crosses dataset splits.")
    return data, groups, metadata
