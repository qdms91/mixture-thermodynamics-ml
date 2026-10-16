"""Fast scientific-invariant checks; synthetic examples, no reference jobs.

Run from the project root: python -m unittest test_project4_retention -v
"""

import os
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'

from copy import deepcopy
import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import project2_common as common
import project4_learning as legacy
import project4_retention_learning as learning


class RetentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(901)
        cls.rows = pd.DataFrame(rng.dirichlet(np.ones(5), 80), columns=common.X_COLUMNS)
        cls.rows['recipe_id'] = [f'r{i:03d}' for i in range(80)]
        cls.rows['block_id'] = cls.rows.recipe_id
        cls.rows['split'] = 'train'
        cls.rows['instability_label'] = (cls.rows.x_water > .18).astype(int)
        cls.rows['stability_seconds'] = np.arange(80) + 1.
        cls.packages = {}
        for columns in [common.COMPOSITION, common.feature_groups()['b_molecular']]:
            values = learning.features(cls.rows, columns)
            models = []
            for seed in range(3):
                model = make_pipeline(StandardScaler(), MLPClassifier(
                    hidden_layer_sizes=(128, 64), max_iter=3, random_state=seed))
                with warnings.catch_warnings(), threadpool_limits(limits=1):
                    warnings.simplefilter('ignore')
                    model.fit(values, cls.rows.instability_label)
                models.append(model)
            cls.packages[len(columns)] = {'columns': columns, 'models': models, 'constant': None}
        cls.config = {'epochs': 3, 'min_epochs': 3, 'patience': 60, 'tol': 1e-5,
                      'alpha': .01, 'learning_rate': .001, 'replay_share': .2}

    def test_affine_conversion_preserves_scores_including_endpoints(self):
        probe = pd.concat([self.rows[common.X_COLUMNS], pd.DataFrame(np.eye(5), columns=common.X_COLUMNS)])
        for original in self.packages.values():
            adapted = learning.adapt_source(original, original['columns'])
            self.assertLess(learning.check_source_predictions(original, adapted, probe), 1e-12)

    def test_duration_selection_and_class_matched_control(self):
        data = self.rows.copy()
        data['eligible'] = True
        data.loc[79, 'eligible'] = False  # The very slowest is ineligible.
        data['instability_label'] = (np.arange(80) % 3 == 0).astype(int)
        a = learning.choose_memories(data, 11)
        b = learning.choose_memories(data.sample(frac=1, random_state=7), 11)
        for policy in a:
            self.assertEqual(a[policy].recipe_id.tolist(), b[policy].recipe_id.tolist())
            self.assertEqual(len(a[policy]), 20)
            self.assertEqual(a[policy].recipe_id.nunique(), 20)
        self.assertEqual(set(a['slowest20'].recipe_id), set(data.iloc[59:79].recipe_id))
        self.assertEqual(a['slowest20'].instability_label.sum(),
                         a['time_class_matched20'].instability_label.sum())

    def test_eligibility_rejects_heldout_and_unknown_training_ids(self):
        original = self.packages[4]
        source = learning.adapt_source(original, original['columns'])
        corrupted = self.rows.copy(); corrupted.loc[0, 'split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'held-out'):
            learning.eligible_memory(corrupted, corrupted.recipe_id.tolist(), source, .5)
        with self.assertRaisesRegex(ValueError, 'missing'):
            learning.eligible_memory(self.rows, ['not_in_training'], source, .5)

    def test_only_features_enter_network(self):
        changed = self.rows.copy()
        changed['instability_label'] = 1 - changed.instability_label
        changed['stability_seconds'] = 999999
        changed['g_mix_kJ_mol'] = -999
        for package in self.packages.values():
            np.testing.assert_array_equal(learning.features(self.rows, package['columns']),
                                          learning.features(changed, package['columns']))

    def test_replay_weight_budget(self):
        y = np.array([0] * 51 + [1] * 9)
        memory = np.array([0] * 6 + [1] * 14)
        weights = learning.training_weights(y, memory)
        self.assertAlmostEqual(weights[:60].sum(), 48.)
        self.assertAlmostEqual(weights[60:].sum(), 12.)
        self.assertTrue((weights > 0).all())
        self.assertAlmostEqual(learning.training_weights(y, []).sum(), 60.)

    def test_no_replay_matches_legacy_fit_exactly(self):
        original = self.packages[4]
        source = learning.adapt_source(original, original['columns'])
        data = self.rows.iloc[:60]
        before = deepcopy(source['models'][0].coefs_[0])
        actual = learning.fit_committee(source, data, data.iloc[:0], True, 11, self.config)
        expected = legacy.fit_committee(data[common.X_COLUMNS].to_numpy(), data.instability_label,
                                        source['models'], True, 11, self.config)
        np.testing.assert_array_equal(learning.scores(actual, data),
                                      legacy.scores(expected, data[common.X_COLUMNS].to_numpy()))
        np.testing.assert_array_equal(before, source['models'][0].coefs_[0])

    def test_replay_fit_is_repeatable_and_heldout_rows_are_rejected(self):
        original = self.packages[17]
        source = learning.adapt_source(original, original['columns'])
        target, memory = self.rows.iloc[:60], self.rows.iloc[60:]
        for transfer in [True, False]:
            a = learning.fit_committee(source, target, memory, transfer, 22, self.config)
            b = learning.fit_committee(source, target, memory, transfer, 22, self.config)
            np.testing.assert_array_equal(learning.scores(a, target), learning.scores(b, target))
            self.assertAlmostEqual(a['replay_weight_share'], .2)
        wrong = memory.copy(); wrong['split'] = 'validation'
        with self.assertRaisesRegex(ValueError, 'held-out'):
            learning.fit_committee(source, target, wrong, True, 22, self.config)


if __name__ == '__main__':
    unittest.main()
