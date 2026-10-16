"""Target learning: same MLP architecture, different initial weights and queries."""

from copy import deepcopy
from time import perf_counter

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from project3_common import choose_batch, derived_seed, initial_positions


def require(condition, message):
    if not condition:
        raise ValueError(message)


def coordinates(x):
    # Fixed units require no fitted scaler or hidden target data. Benzene is
    # one minus the other four fractions. Both learning arms use these units.
    return (np.asarray(x, float)[:, :4] - .2) / .2


def source_networks(package):
    """Change input coordinates without changing any source predictions."""
    require(package.get('constant') is None, 'A constant source cannot be fine-tuned.')
    answer = []
    for pipeline in package['models']:
        scaler, original = pipeline.steps[0][1], pipeline.steps[-1][1]
        require(isinstance(original, MLPClassifier), 'Expected a source MLP.')
        require(tuple(original.hidden_layer_sizes) == (128, 64), 'Expected layers (128, 64).')
        require(original.n_features_in_ == 4, 'Use composition-only source models.')
        network = deepcopy(original)
        old_weight = network.coefs_[0].copy()
        network.coefs_[0] = (.2 / scaler.scale_)[:, None] * old_weight
        network.intercepts_[0] += ((.2 - scaler.mean_) / scaler.scale_) @ old_weight
        answer.append(network)
    require(len(answer) > 0, 'Empty source committee.')
    return answer


def reset_training(network, seed, n, config):
    """Keep weights, but reset Adam and the source training-loss history."""
    network.set_params(alpha=config['alpha'], learning_rate_init=config['learning_rate'],
                       batch_size=min(64, n), early_stopping=False, solver='adam',
                       random_state=np.random.RandomState(seed), warm_start=False,
                       tol=0., n_iter_no_change=config['epochs'] + 1)
    # These fitted attributes are version-sensitive. The command checks the
    # sklearn version, and the supplied tests verify prediction preservation.
    for name in ['_optimizer', '_random_state']:
        if hasattr(network, name):
            delattr(network, name)
    network.t_, network.n_iter_ = 0, 0
    network.loss_curve_, network.best_loss_ = [], np.inf
    network._no_improvement_count = 0
    network.validation_scores_, network.best_validation_score_ = None, None
    return network


def fit_committee(x, y, sources, transfer, seed, config):
    """Fit only revealed target labels; begin each round from its original weights."""
    started = perf_counter()
    inputs = coordinates(x)
    y = np.asarray(y, int)
    weight = compute_sample_weight('balanced', y)
    models, diagnostics = [], []
    for member, source in enumerate(sources):
        member_seed = derived_seed(seed, len(y), member, 401)
        if transfer:
            model = reset_training(deepcopy(source), member_seed, len(y), config)
        else:
            model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu')
            model = reset_training(model, member_seed, len(y), config)
        best, stale, best_weights = np.inf, 0, None
        with threadpool_limits(limits=1):
            for epoch in range(1, config['epochs'] + 1):
                # partial_fit preserves Adam between epochs. classes allows a
                # legitimate initial sample containing only negative labels.
                model.partial_fit(inputs, y, classes=np.array([0, 1]), sample_weight=weight)
                require(np.isfinite(model.loss_), 'Nonfinite neural-network training loss.')
                improvement = best - model.loss_
                if model.loss_ < best:
                    best = float(model.loss_)
                    best_weights = (deepcopy(model.coefs_), deepcopy(model.intercepts_))
                stale = 0 if improvement > config['tol'] else stale + 1
                if epoch >= config['min_epochs'] and stale >= config['patience']:
                    break
        model.coefs_, model.intercepts_ = best_weights
        models.append(model)
        diagnostics.append({'epochs': epoch, 'epoch_cap_reached': epoch == config['epochs'],
                            'best_training_loss': best})
    return {'models': models, 'diagnostics': diagnostics,
            'fit_seconds': perf_counter() - started, 'transfer': transfer}


def scores(package, x):
    with threadpool_limits(limits=1):
        return np.vstack([m.predict_proba(coordinates(x))[:, 1] for m in package['models']])


def select_next(strategy, x, selected, count, seed, package):
    # This function receives compositions and model scores. It never receives
    # unqueried labels, reference costs, validation answers or test answers.
    name = 'random' if strategy == 'random' else 'uncertainty_diversity'
    member_scores = None if name == 'random' else scores(package, x)
    return choose_batch(name, x, selected, count, seed, member_scores)
