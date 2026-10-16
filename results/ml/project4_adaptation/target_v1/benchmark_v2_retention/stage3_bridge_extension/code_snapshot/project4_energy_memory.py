"""Frozen Ridge predictions and twenty-example energy replay memories."""

from pathlib import Path
from time import perf_counter
import json

import joblib
import numpy as np
import pandas as pd
from scipy.constants import R
from scipy.special import xlogy
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from threadpoolctl import threadpool_limits

import project2_common as common
from project3_common import derived_seed
from project4_retention_learning import require

TEMPERATURE = 298.15
ENERGY = 'predicted_g_mix_kJ_mol'
PAIRS = {'highest_g20': 'highest_g_class_matched20',
         'lowest_g20': 'lowest_g_class_matched20',
         'energy_bridge20': 'bridge_class_matched20'}


def load_ridge(directory, timing_protocol):
    """Verify model meaning and surface provenance; no regression refitting."""
    directory = Path(directory)
    metadata = json.loads((directory / 'run_metadata.json').read_text())
    selected = json.loads((directory / 'selected_settings.json').read_text())
    require(metadata['molecule_order'] == common.MOLECULES and
            metadata['features'] == common.COMPOSITION and
            metadata['target'] == 'g_excess_kJ_mol' and
            metadata['temperature_K'] == TEMPERATURE,
            'Ridge inputs, target, solvent order or temperature differ.')
    require(metadata['versions']['scikit-learn'] == '1.8.0', 'Expected the saved sklearn 1.8.0 model.')
    require(timing_protocol['temperature_K'] == TEMPERATURE, 'Source temperature differs.')
    for name in common.MOLECULES:
        key = f'surface:{name}'
        require(metadata['calculation_metadata']['sha256'][key] == timing_protocol['sha256'][key],
                f'Thermodynamics and stability used different surfaces: {name}.')
    model = joblib.load(directory / 'ridge_model.joblib')
    require(model.n_features_in_ == 4, 'Ridge should take four explicit mole fractions.')
    require(isinstance(model.steps[-1][1], Ridge), 'Expected the selected polynomial Ridge pipeline.')
    polynomials = [obj for _, obj in model.steps if isinstance(obj, PolynomialFeatures)]
    choice = selected['selected_by_validation']['ridge']
    require(len(polynomials) == 1 and polynomials[0].degree == choice['degree'] and
            model.steps[-1][1].alpha == choice['alpha'], 'Ridge checkpoint differs from selected settings.')
    return model, metadata, selected


def predicted_energies(model, rows):
    """Add exact ideal mixing energy to Ridge's predicted excess energy.

    This is the homogeneous-liquid molar g_mix at 298.15 K. It ranks memories;
    it is not supplied as an additional classifier input or instability label.
    """
    x = rows[common.X_COLUMNS].to_numpy(float)
    require(np.isfinite(x).all() and (x >= .15-1e-12).all() and (x <= .4+1e-12).all()
            and np.allclose(x.sum(axis=1), 1, atol=1e-12, rtol=0),
            'Replay energy prediction expects original in-domain source compositions.')
    started = perf_counter()
    with threadpool_limits(limits=1):
        excess = np.asarray(model.predict(x[:, :4]), float)
    require(excess.shape == (len(rows),) and np.isfinite(excess).all(), 'Invalid Ridge predictions.')
    ideal = R * TEMPERATURE / 1000 * np.sum(xlogy(x, x), axis=1)
    answer = rows.copy()
    answer['predicted_g_excess_kJ_mol'] = excess
    answer['analytical_g_ideal_kJ_mol'] = ideal
    answer[ENERGY] = excess + ideal
    return answer, perf_counter() - started


def ranked_eligible(candidates):
    require(candidates.recipe_id.is_unique and candidates.split.eq('train').all(),
            'Replay candidates must be distinct source-training examples.')
    require(candidates.instability_label.isin([0, 1]).all(), 'Unresolved source labels.')
    require(candidates.eligible.isin([True, False]).all(), 'Invalid frozen eligibility flags.')
    ranked = candidates.loc[candidates.eligible].sort_values(
        [ENERGY, 'recipe_id'], ascending=[False, True], kind='stable').reset_index(drop=True).copy()
    require(len(ranked) >= 20 and np.isfinite(ranked[ENERGY]).all(),
            'At least twenty eligible examples with finite predicted energies are required.')
    ranked['energy_rank_descending'] = np.arange(1, len(ranked) + 1)
    return ranked


def bridge_memory(ranked):
    """Strict Daniel rule; never move the anchor to obtain a feasible memory."""
    negative = np.flatnonzero(ranked.instability_label.to_numpy() == 0)
    if not len(negative):
        return None, {'status': 'unavailable', 'reason': 'No eligible label-0 example.'}
    anchor = int(negative[0])
    info = {'first_negative_rank': anchor+1,
            'first_negative_recipe_id': str(ranked.iloc[anchor].recipe_id),
            'first_negative_g_mix_kJ_mol': float(ranked.iloc[anchor][ENERGY]),
            'preceding_positive_count': anchor, 'eligible_negative_count': int(len(negative))}
    if anchor < 10 or len(negative) < 10:
        return None, {**info, 'status': 'unavailable',
                      'reason': 'Need ten positives above the FIRST negative and ten negatives from it downward.'}
    positive_rows = ranked.iloc[anchor-10:anchor].copy()
    negative_rows = ranked.iloc[negative[:10]].copy()
    positive_rows['bridge_side'] = 'ten_preceding_positives'
    negative_rows['bridge_side'] = 'first_ten_negatives'
    memory = pd.concat([positive_rows, negative_rows], ignore_index=True)
    return memory, {**info, 'status': 'available', 'reason': '',
                    'skipped_intervening_positive_count': int(negative[9]-anchor+1-10)}


def choose_energy_memories(candidates, seed):
    """Three energy rules and a separate random class-matched control for each."""
    ranked = ranked_eligible(candidates)
    selections = {'highest_g20': ranked.head(20).copy(),
                  'lowest_g20': ranked.sort_values([ENERGY, 'recipe_id'], ascending=[True, True]).head(20).copy()}
    bridge, info = bridge_memory(ranked)
    availability = [{**info, 'policy': name} for name in ['energy_bridge20', 'bridge_class_matched20']]
    if bridge is not None:
        selections['energy_bridge20'] = bridge
    pool = ranked.sort_values('recipe_id').reset_index(drop=True)
    for policy, memory in list(selections.items()):
        # Counts, not energies, choose the random-control composition. Equal
        # requested class counts use the same seed convention across policies.
        positives = int(memory.instability_label.sum())
        rng = np.random.default_rng(derived_seed(seed, 931, positives))
        pieces = []
        for label, count in [(0, 20-positives), (1, positives)]:
            subset = pool.loc[pool.instability_label == label]
            require(len(subset) >= count, 'Class-matched control cannot be constructed.')
            if count:
                pieces.append(subset.iloc[rng.choice(len(subset), count, replace=False)])
        selections[PAIRS[policy]] = pd.concat(pieces, ignore_index=True)
    # Canonical order aligns fitting across selections that happen to choose
    # the same recipe set. Energy order is preserved in the rank column.
    return ({name: rows.sort_values('recipe_id').reset_index(drop=True)
             for name, rows in selections.items()}, availability, ranked)
