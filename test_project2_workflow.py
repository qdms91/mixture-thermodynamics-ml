"""Fast checks of descriptor physics, leakage exclusions and classifier behavior.

Run: python -m unittest test_project2_workflow.py
The artificial examples below are software checks, not research results.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

import project2_common as common
import train_stability_classifiers as training


class WorkflowChecks(unittest.TestCase):
    def test_descriptor_pure_limits(self):
        data = pd.DataFrame(np.eye(5), columns=common.X_COLUMNS)
        derived = common.add_descriptors(data)
        np.testing.assert_allclose(derived.mean_molar_mass_g_mol, common.MOLAR_MASS)
        np.testing.assert_allclose(derived.volume_additive_density_g_mL, common.DENSITY)
        np.testing.assert_allclose(derived.mean_atom_count, common.ATOMS.sum(1))

    def test_feature_lists_exclude_search_evidence(self):
        forbidden = ["tpd", "label", "trial", "remaining", "converged", "seconds", "recipe_id", "block", "split"]
        for columns in common.feature_groups().values():
            self.assertEqual(len(columns), len(set(columns)))
            self.assertFalse(any(word in name for name in columns for word in forbidden))

    def test_threshold_and_confusion(self):
        y = np.array([0, 0, 1, 1])
        score = np.array([.1, .3, .6, .9])
        threshold = common.choose_threshold(y, score)
        result = common.metrics(y, score, threshold)
        self.assertEqual((result['tn'], result['fp'], result['fn'], result['tp']), (2, 0, 0, 2))
        self.assertAlmostEqual(result['average_precision'], 1)

    def test_constant_score_ap_is_prevalence(self):
        result = common.metrics([0, 0, 0, 1], [0, 0, 0, 0], .5)
        self.assertAlmostEqual(result['average_precision'], .25)
        # A straight-line PR integral differs for this tied-score baseline.
        self.assertAlmostEqual(result['pr_auc_trapezoid'], .625)

    def test_all_estimators_and_training_only_scaler(self):
        x, y = make_classification(n_samples=150, n_features=4, n_informative=3,
                                  n_redundant=0, weights=[.8, .2], random_state=9)
        configurations = [dict(family='majority', weighted=False),
            dict(family='logistic', C=1., weighted=True),
            dict(family='svm', C=1., gamma='scale', weighted=True),
            dict(family='mlp', layers=[16, 8], alpha=.01, weighted=True)]
        for setting in configurations:
            estimator, _ = training.fit_estimator(setting, x[:100], y[:100], 9)
            np.testing.assert_allclose(estimator.steps[0][1].mean_, x[:100].mean(0))
            before = estimator.steps[0][1].mean_.copy()
            predicted = training.scores([estimator], x[100:], setting['family'])
            self.assertTrue(np.isfinite(predicted).all())
            np.testing.assert_array_equal(estimator.steps[0][1].mean_, before)


if __name__ == '__main__':
    unittest.main()
