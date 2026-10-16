"""Tests for the post-Stage-3 P3 downstream bridge selector.

These tests exercise selection logic only. They do not claim scientific
performance and do not fit a model.
"""

import unittest

import numpy as np
import pandas as pd

import project4_bridge_memory as bridge


def candidate_rows():
    # Descending-energy ranks are deliberately interspersed at the top:
    # negatives at ranks 2, 4, ..., 20 must be excluded from the bridge even
    # though they occur before the tenth positive.
    labels = ([1, 0] * 10) + ([0] * 30)
    rows = pd.DataFrame({
        'recipe_id': [f'r{i:03d}' for i in range(len(labels))],
        'split': 'train',
        'instability_label': labels,
        'eligible': True,
        'predicted_g_mix_kJ_mol': np.arange(len(labels), 0, -1, dtype=float),
    })
    return rows.sample(frac=1.0, random_state=17).reset_index(drop=True)


class DownstreamBridgeTests(unittest.TestCase):
    def test_first_ten_positives_then_downstream_negatives(self):
        selections, info, ranked = bridge.choose_memories(candidate_rows(), seed=11)
        memory = selections[bridge.POLICY]
        self.assertEqual(len(memory), 20)
        self.assertEqual(memory.instability_label.value_counts().to_dict(), {0: 10, 1: 10})
        positives = memory.loc[memory.instability_label == 1, 'energy_rank_descending']
        negatives = memory.loc[memory.instability_label == 0, 'energy_rank_descending']
        self.assertEqual(sorted(positives.tolist()), list(range(1, 20, 2)))
        self.assertTrue((negatives >= 20).all())
        self.assertEqual(info['tenth_positive_rank'], 19)
        self.assertEqual(info['excluded_negatives_before_tenth_positive'], 9)
        self.assertEqual(info['first_selected_negative_rank'], 20)

    def test_control_is_random_ten_ten_and_uses_same_pool(self):
        selections_a, _, _ = bridge.choose_memories(candidate_rows(), seed=11)
        selections_b, _, _ = bridge.choose_memories(candidate_rows(), seed=11)
        control = selections_a[bridge.CONTROL]
        self.assertEqual(control.recipe_id.tolist(), selections_b[bridge.CONTROL].recipe_id.tolist())
        self.assertEqual(control.instability_label.value_counts().to_dict(), {0: 10, 1: 10})
        self.assertTrue(set(control.recipe_id) <= set(candidate_rows().recipe_id))

    def test_ineligible_rows_are_excluded(self):
        rows = candidate_rows()
        rows.loc[rows.recipe_id == 'r000', 'eligible'] = False
        rows.loc[rows.recipe_id == 'r025', 'instability_label'] = 1
        # Removing the first positive shifts the tenth-positive anchor, but the
        # selector must still return exactly ten eligible positives.
        selections, _, _ = bridge.choose_memories(rows, seed=11)
        memory = selections[bridge.POLICY]
        self.assertEqual(memory.instability_label.value_counts().to_dict(), {0: 10, 1: 10})
        self.assertNotIn('r000', memory.recipe_id.tolist())

    def test_too_few_positives_is_explicitly_rejected(self):
        rows = candidate_rows()
        rows.loc[rows.instability_label == 1, 'instability_label'] = 0
        with self.assertRaises(ValueError):
            bridge.choose_memories(rows, seed=11)


if __name__ == '__main__':
    unittest.main()
