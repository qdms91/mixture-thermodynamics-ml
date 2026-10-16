"""Learning and recipe-selection functions for the active-learning benchmark."""

import inspect
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

import project2_common as common

STRATEGIES = ["random", "space_filling", "uncertainty_diversity"]
ROOT = Path(__file__).resolve().parent
DEFAULT_PREPARED = ROOT / "results/ml/project2_classification/prepared_seed2026"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def derived_seed(seed, *parts):
    """Independent, reproducible random streams; no dependence on worker order."""
    return int(np.random.SeedSequence([seed, *parts]).generate_state(1)[0])


def load_data(directory):
    data, _, metadata = common.load_prepared(directory)
    require(data.recipe_id.is_unique, "Prepared data contain duplicate recipe IDs.")
    require(set(data.split) == {"train", "validation", "test"}, "Unexpected split names.")
    require(data.instability_label.isin([0, 1]).all(), "Expected labels 0 and 1.")
    x = data[common.X_COLUMNS].to_numpy(float)
    require(np.isfinite(x).all() and (x >= 0).all()
            and np.allclose(x.sum(axis=1), 1, rtol=0, atol=1e-10), "Invalid compositions.")
    for role, rows in data.groupby("split"):
        require(rows.instability_label.nunique() == 2, f"The {role} split lacks a class.")
    require("sample_weight" in inspect.signature(MLPClassifier.fit).parameters,
            "This script needs scikit-learn >= 1.7 for weighted MLP fitting.")
    return data, metadata


class LabelOracle:
    """Reveal recorded answers only when their training-pool positions are queried.

    An actual prospective oracle would run COSMO-RS. This retrospective oracle
    looks up completed calculations. Selection functions never receive it.
    """

    def __init__(self, labels):
        self._answers = np.asarray(labels, dtype=int).copy()
        self.queried = set()

    def reveal(self, positions):
        positions = np.asarray(positions, dtype=int)
        require(positions.ndim == 1 and len(set(positions)) == len(positions),
                "Query positions must be a one-dimensional list without duplicates.")
        require(((positions >= 0) & (positions < len(self._answers))).all(),
                "Query outside the training pool.")
        self.queried.update(positions.tolist())
        return self._answers[positions].copy()


def initial_positions(pool_size, count, seed):
    """Choose the same label-blind initial sample for all strategies of a seed."""
    rng = np.random.default_rng(derived_seed(seed, 10))
    return rng.choice(pool_size, count, replace=False).tolist()


def fit_committee(x, y, seed, config):
    """Fit independent MLPs from scratch using only the currently revealed rows."""
    started = perf_counter()
    if len(np.unique(y)) == 1:
        # Do not inspect hidden labels to force a positive into the initial set.
        # If only one class is known, exploration continues with constant scores.
        return {"models": [], "constant": int(y[0])}, {
            "fit_seconds": perf_counter() - started, "warnings": [], "iterations": [],
            "constant_class": int(y[0]),
        }
    models, messages, iterations, attempts = [], [], [], []
    weight = compute_sample_weight("balanced", y)
    for member in range(config["committee"]):
        model = make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=tuple(config["layers"]), alpha=config["alpha"],
            solver="adam", activation="relu", learning_rate_init=.001,
            batch_size=min(64, len(y)), max_iter=config["max_epochs"],
            tol=1e-5, n_iter_no_change=40, early_stopping=False,
            random_state=derived_seed(seed, len(y), member, 20)))
        # The scaler and class weights use revealed training rows only. There
        # is no internal validation split and no reuse of Project 2 weights.
        member_iterations = 0
        for attempt in range(2):
            with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(x, y, mlpclassifier__sample_weight=weight)
            member_iterations += int(model[-1].n_iter_)
            attempts.append({"member": member, "attempt": attempt + 1,
                             "iterations": int(model[-1].n_iter_),
                             "warnings": [str(w.message) for w in caught]})
            unfinished = any(issubclass(w.category, ConvergenceWarning) for w in caught)
            if not unfinished or attempt == 1:
                messages.extend(str(w.message) for w in caught)
                break
            # One training-only continuation if the optimizer hits its cap.
            # These weights come from this same fit on the same revealed rows.
            # No weights are reused between acquisition rounds or from Project 2.
            model[-1].warm_start = True
            model[-1].max_iter = 2 * config["max_epochs"]
        iterations.append(member_iterations)
        models.append(model)
    return {"models": models, "constant": None}, {
        "fit_seconds": perf_counter() - started, "warnings": messages,
        "iterations": iterations, "optimization_attempts": attempts, "constant_class": None,
    }


def committee_scores(package, x):
    if package["constant"] is not None:
        return np.full((1, len(x)), float(package["constant"]))
    with threadpool_limits(limits=1):
        return np.stack([model.predict_proba(x)[:, 1] for model in package["models"]])


def binary_entropy(probability):
    p = np.clip(probability, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2)


def uncertainty_scores(member_scores):
    """Committee disagreement plus a smaller mean-score ambiguity term.

    These are acquisition heuristics, not calibrated physical probabilities.
    The disagreement is Jensen-Shannon divergence for Bernoulli predictions.
    """
    mean_entropy = binary_entropy(member_scores.mean(axis=0))
    disagreement = np.maximum(0, mean_entropy - binary_entropy(member_scores).mean(axis=0))
    return disagreement + .25 * mean_entropy


def farthest_first(x5, candidates, selected, count, rng):
    """Spread new recipes in five-component composition space, using no labels."""
    candidates = np.asarray(candidates, dtype=int).copy()
    rng.shuffle(candidates)  # Resolve geometric ties without favoring CSV order.
    require(count <= len(candidates), "Not enough unqueried candidates.")
    distance = cdist(x5[candidates], x5[selected], metric="sqeuclidean").min(axis=1)
    chosen = []
    for _ in range(count):
        position = int(np.argmax(distance))
        index = int(candidates[position])
        chosen.append(index)
        new_distance = ((x5[candidates] - x5[index]) ** 2).sum(axis=1)
        distance = np.minimum(distance, new_distance)
        distance[position] = -np.inf
    return chosen


def choose_batch(strategy, x5, selected, count, seed, member_scores=None):
    """Choose recipe positions without access to any reference labels or costs."""
    rng = np.random.default_rng(derived_seed(seed, len(selected), 30))
    remaining = np.setdiff1d(np.arange(len(x5)), np.asarray(selected))
    require(0 < count <= len(remaining), "Invalid acquisition batch size.")
    if strategy == "random":
        return rng.choice(remaining, count, replace=False).tolist()
    if strategy == "space_filling":
        return farthest_first(x5, remaining, selected, count, rng)
    require(strategy == "uncertainty_diversity", "Unknown acquisition strategy.")
    require(member_scores is not None and member_scores.shape[1] == len(x5),
            "Active selection needs a score for every pool recipe.")

    # With the default batch of 25, select 20 informative/diverse recipes plus
    # 5 random exploratory recipes. Exploration can discover confidently wrong
    # regions that pure uncertainty sampling would overlook.
    explore = max(1, int(np.ceil(.2 * count)))
    informed = count - explore
    score = uncertainty_scores(member_scores)
    order = rng.permutation(remaining)
    if np.ptp(score[remaining]) <= 1e-12:
        candidates = order  # Constant models have no useful uncertainty ranking.
    else:
        order = order[np.argsort(-score[order], kind="stable")]
        candidates = order[:min(len(order), max(informed, 5 * informed))]
    chosen = farthest_first(x5, candidates, selected, informed, rng) if informed else []
    remaining = np.setdiff1d(remaining, chosen)
    chosen.extend(rng.choice(remaining, explore, replace=False).tolist())
    return chosen
