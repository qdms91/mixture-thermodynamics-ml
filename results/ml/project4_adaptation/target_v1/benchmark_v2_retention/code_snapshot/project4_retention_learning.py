"""Feature-aware MLP adaptation and duration-based source replay.

Only supplied training rows enter these functions. Validation and test labels
are handled by the separate reporting command.
"""

from copy import deepcopy
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

import project2_common as common
from project3_common import derived_seed
import project4_learning as legacy


POLICIES = ['random20', 'slowest20', 'time_class_matched20']
require = legacy.require


def features(rows, columns):
    """Compute only the frozen composition/molecular feature definitions."""
    allowed = common.feature_groups()
    require(columns in [allowed['a_composition'], allowed['b_molecular']],
            'Unexpected feature list: only composition or molecular inputs are allowed.')
    x = rows[common.X_COLUMNS].to_numpy(float)
    require(np.isfinite(x).all() and (x >= 0).all() and
            np.allclose(x.sum(axis=1), 1., atol=1e-10, rtol=0), 'Invalid mole fractions.')
    values = common.add_descriptors(rows[common.X_COLUMNS])[columns].to_numpy(float)
    require(np.isfinite(values).all(), 'Nonfinite molecular descriptor.')
    return values


def adapt_source(package, columns):
    """Preserve source predictions while exposing networks and fixed scaling.

    Four-input models use the same coordinates as benchmark_v1. Molecular
    models keep their original source-fitted scalers. Scratch uses the same
    transforms as its paired transferred network; scalers are never refitted.
    """
    require(package.get('constant') is None, 'A constant classifier cannot be fine-tuned.')
    require(package.get('columns') == columns, 'Source feature order changed.')
    networks, centers, scales = [], [], []
    require(len(package['models']) == 3, 'Expected the original three-member committee.')
    for pipeline in package['models']:
        require(len(pipeline.steps) == 2, 'Expected a scaler followed by an MLP.')
        scaler, original = pipeline.steps[0][1], pipeline.steps[-1][1]
        require(isinstance(original, MLPClassifier) and
                tuple(original.hidden_layer_sizes) == (128, 64), 'Unexpected source architecture.')
        require(original.n_features_in_ == len(columns), 'Wrong number of source inputs.')
        require(np.array_equal(original.classes_, [0, 1]), 'Expected binary classes [0, 1].')
        center = np.full(4, .2) if len(columns) == 4 else scaler.mean_.copy()
        scale = np.full(4, .2) if len(columns) == 4 else scaler.scale_.copy()
        network = deepcopy(original)
        old_weight = network.coefs_[0].copy()
        network.coefs_[0] = (scale / scaler.scale_)[:, None] * old_weight
        network.intercepts_[0] += ((center - scaler.mean_) / scaler.scale_) @ old_weight
        networks.append(network); centers.append(center); scales.append(scale)
    return {'format': 'retention_mlp_v1', 'models': networks, 'columns': columns,
            'centers': centers, 'scales': scales}


def scores(package, rows):
    values = features(rows, package['columns'])
    with threadpool_limits(limits=1):
        return np.stack([model.predict_proba((values - center) / scale)[:, 1]
                         for model, center, scale in zip(package['models'],
                                                        package['centers'], package['scales'])])


def check_source_predictions(original, adapted, rows):
    values = features(rows, adapted['columns'])
    with threadpool_limits(limits=1):
        before = np.stack([m.predict_proba(values)[:, 1] for m in original['models']])
    after = scores(adapted, rows)
    error = float(np.max(np.abs(before - after)))
    require(np.allclose(before, after, rtol=1e-9, atol=1e-10),
            f'Source-coordinate conversion changed predictions: {error}')
    return error


def eligible_memory(source_training, original_training_ids, source, cutoff):
    """Correctness is assessed only on that source model's training examples."""
    require(source_training.split.eq('train').all(), 'Replay candidates include held-out rows.')
    require(source_training.recipe_id.is_unique, 'Duplicate source training ID.')
    require(len(set(original_training_ids)) == len(original_training_ids), 'Duplicate checkpoint training ID.')
    indexed = source_training.set_index('recipe_id', drop=False)
    require(set(original_training_ids) <= set(indexed.index),
            'A checkpoint training ID is missing or belongs to a held-out split.')
    rows = indexed.loc[original_training_ids].reset_index(drop=True).copy()
    rows['source_score'] = scores(source, rows).mean(axis=0)
    rows['source_cutoff'] = float(cutoff)
    rows['source_prediction'] = (rows.source_score >= cutoff).astype(int)
    rows['source_correct'] = rows.source_prediction.eq(rows.instability_label)
    rows['eligible'] = rows.source_correct
    return rows


def choose_memories(candidates, seed, size=20):
    """Select all policies before adaptation; deterministic ties use recipe IDs."""
    require(size == 20, 'This protocol uses twenty replay examples per run.')
    eligible = candidates.loc[candidates.eligible].sort_values('recipe_id').copy()
    require(len(eligible) >= size, 'Fewer than twenty correctly classified source-training examples.')
    require(np.isfinite(eligible.stability_seconds).all() and (eligible.stability_seconds > 0).all(),
            'Eligible reference timings must be finite and positive; inspect timing provenance.')
    slow = eligible.sort_values(['stability_seconds', 'recipe_id'], ascending=[False, True]).head(size)
    rng = np.random.default_rng(derived_seed(seed, 812))
    random = eligible.iloc[rng.choice(len(eligible), size, replace=False)]
    matched = []
    rng = np.random.default_rng(derived_seed(seed, 813))
    for label in [0, 1]:
        count = int(slow.instability_label.eq(label).sum())
        pool = eligible.loc[eligible.instability_label == label]
        require(len(pool) >= count, 'Insufficient class count for matched replay.')
        if count:
            matched.append(pool.iloc[rng.choice(len(pool), count, replace=False)])
    result = {'random20': random, 'slowest20': slow,
              'time_class_matched20': pd.concat(matched, ignore_index=True)}
    # Selection order is not training order. Canonical ordering prevents an
    # unintended correlation between input order, duration and optimizer steps.
    return {name: rows.sort_values('recipe_id').reset_index(drop=True)
            for name, rows in result.items()}


def training_weights(y_target, y_memory, replay_share=.2):
    """Use only revealed labels; allocate 80:20 of supplied sample-weight mass.

    Total weight equals the target count, preserving its global regularization
    denominator. Shuffled minibatches have varying local source proportions;
    80:20 describes the full supplied weight vector, not every minibatch.
    """
    n = len(y_target)
    require(n > 0 and 0 < replay_share < 1, 'Invalid target size or replay weight.')
    y = np.concatenate([y_target, y_memory]).astype(int)
    require(np.isin(y, [0, 1]).all(), 'Training labels must be 0 or 1.')
    weight = compute_sample_weight('balanced', y)
    if len(y_memory):
        weight[:n] *= (1 - replay_share) * n / weight[:n].sum()
        weight[n:] *= replay_share * n / weight[n:].sum()
    return weight


def fit_committee(source, target_rows, memory_rows, transfer, seed, config):
    """Start each fit from its original source weights or fresh random weights."""
    require(target_rows.split.eq('train').all(), 'Target fit received held-out rows.')
    require(memory_rows.empty or memory_rows.split.eq('train').all(), 'Replay fit received held-out rows.')
    y_target = target_rows.instability_label.to_numpy(int)
    y_memory = memory_rows.instability_label.to_numpy(int)
    # Delegate this case unchanged: it is also how the reusable P3 baselines
    # were fitted. The feature adapter already gives their common coordinates.
    if memory_rows.empty and len(source['columns']) == 4:
        fitted = legacy.fit_committee(target_rows[common.X_COLUMNS].to_numpy(float), y_target,
                                      source['models'], transfer, seed, config)
        return {**source, **fitted, 'replay_size': 0, 'replay_weight_share': 0.}

    started = perf_counter()
    rows = pd.concat([target_rows, memory_rows], ignore_index=True)
    values = features(rows, source['columns'])
    y = np.concatenate([y_target, y_memory])
    weight = training_weights(y_target, y_memory, config['replay_share'])
    models, diagnostics = [], []
    for member, original in enumerate(source['models']):
        member_seed = derived_seed(seed, len(y_target), member, 401)
        model = deepcopy(original) if transfer else MLPClassifier(
            hidden_layer_sizes=(128, 64), activation='relu')
        model = legacy.reset_training(model, member_seed, len(y_target), config)
        x = (values - source['centers'][member]) / source['scales'][member]
        best, stale, best_weights = np.inf, 0, None
        with threadpool_limits(limits=1):
            for epoch in range(1, config['epochs'] + 1):
                model.partial_fit(x, y, classes=np.array([0, 1]), sample_weight=weight)
                require(np.isfinite(model.loss_), 'Nonfinite MLP training loss.')
                improvement = best - model.loss_
                if model.loss_ < best:
                    best = float(model.loss_)
                    best_weights = deepcopy(model.coefs_), deepcopy(model.intercepts_)
                stale = 0 if improvement > config['tol'] else stale + 1
                if epoch >= config['min_epochs'] and stale >= config['patience']:
                    break
        model.coefs_, model.intercepts_ = best_weights
        models.append(model)
        diagnostics.append({'epochs': epoch, 'epoch_cap_reached': epoch == config['epochs'],
                            'best_training_loss': best})
    return {**source, 'models': models, 'diagnostics': diagnostics,
            'fit_seconds': perf_counter() - started, 'transfer': transfer,
            'replay_size': len(memory_rows),
            'replay_weight_share': float(weight[len(y_target):].sum() / weight.sum())}
