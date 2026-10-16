"""P3 extension: ten highest-ranked positives, then ten downstream negatives.

Only frozen source-training candidates enter this module. Energy ties are
resolved by recipe_id, exactly as in Stage 3. 'Downstream' refers to that order.
"""

import numpy as np
import pandas as pd

from project3_common import derived_seed
import project4_energy_memory as energy
from project4_retention_learning import require

POLICY = 'downstream_energy_bridge20'
CONTROL = 'downstream_bridge_class_matched20'
POLICIES = [POLICY, CONTROL]


def choose_memories(candidates, seed):
    ranked = energy.ranked_eligible(candidates)
    labels = ranked.instability_label.to_numpy(int)
    positives = np.flatnonzero(labels == 1)
    info = {'eligible_examples': len(ranked),
            'eligible_positives': int((labels == 1).sum()),
            'eligible_negatives': int((labels == 0).sum())}
    require(len(positives) >= 10,
            f'Seed {seed}: fewer than ten eligible positives; no fallback is allowed.')
    chosen_positive = positives[:10]
    anchor = int(chosen_positive[-1])
    downstream = np.flatnonzero((labels == 0) & (np.arange(len(ranked)) > anchor))
    require(len(downstream) >= 10,
            f'Seed {seed}: fewer than ten eligible negatives AFTER the tenth positive; no fallback.')
    chosen_negative = downstream[:10]
    pos = ranked.iloc[chosen_positive].copy()
    neg = ranked.iloc[chosen_negative].copy()
    pos['bridge_side'] = 'first_ten_positives'
    neg['bridge_side'] = 'first_ten_negatives_after_tenth_positive'
    memory = pd.concat([pos, neg], ignore_index=True)

    # Same random stream and canonical pool order as Stage 3's 10/10 control.
    # The control can include earlier negatives: it is sampled from the WHOLE
    # eligible source pool, not restricted to the downstream energy interval.
    pool = ranked.sort_values('recipe_id').reset_index(drop=True)
    rng = np.random.default_rng(derived_seed(seed, 931, 10))
    pieces = []
    for label in [0, 1]:
        subset = pool.loc[pool.instability_label == label]
        pieces.append(subset.iloc[rng.choice(len(subset), 10, replace=False)])
    control = pd.concat(pieces, ignore_index=True)
    selections = {POLICY: memory, CONTROL: control}
    for name, rows in selections.items():
        require(len(rows) == rows.recipe_id.nunique() == 20 and
                rows.instability_label.value_counts().to_dict() == {0: 10, 1: 10},
                f'Invalid 10/10 memory: {name}')
        selections[name] = rows.sort_values('recipe_id').reset_index(drop=True)
    require(pos.energy_rank_descending.max() < neg.energy_rank_descending.min(),
            'A selected negative is not downstream of the tenth positive.')
    return selections, {**info, 'status': 'available',
        'tenth_positive_rank': anchor + 1,
        'tenth_positive_recipe_id': str(ranked.iloc[anchor].recipe_id),
        'tenth_positive_g_mix_kJ_mol': float(ranked.iloc[anchor][energy.ENERGY]),
        'excluded_negatives_before_tenth_positive': int((labels[:anchor] == 0).sum()),
        'downstream_negatives_available': int(len(downstream)),
        'first_selected_negative_rank': int(chosen_negative[0] + 1),
        'last_selected_negative_rank': int(chosen_negative[-1] + 1),
        'skipped_positives_during_negative_scan': int(chosen_negative[-1] - anchor - 10),
        'equal_energy_across_boundary': bool(pos[energy.ENERGY].min() == neg[energy.ENERGY].max())}, ranked
