"""Focused checks for binary restrictions, labels and challenge construction."""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from contextlib import redirect_stdout
import io

import numpy as np
import pandas as pd

import project2_common as common
import phase_stability
import stress_test_stability as experiment


class AnalyticMixture:
    """Ideal or regular-solution binary used to check the reference wrapper."""
    def __init__(self, interaction):
        self.interaction = interaction

    def mixture_properties(self, model, x, temperature):
        # A water/benzene feed cannot create methanol, acetone or acetonitrile.
        assert np.all(np.asarray(x)[1:4] == 0)
        active = np.flatnonzero(x)
        w = np.asarray(x)[active]
        lng = self.interaction * (1-w)**2
        return pd.DataFrame({'molecule': np.asarray(common.MOLECULES)[active],
                             'ln_gamma': lng}), {}


class ReferenceTests(unittest.TestCase):
    def test_binary_mass_balance_and_instability(self):
        old = experiment.MODEL, experiment.THERMO, experiment.STABILITY
        try:
            experiment.MODEL = object()
            experiment.STABILITY = phase_stability
            recipe = {'recipe_id': 'ood_0001', 'recipe_number': 1,
                      **dict(zip(common.X_COLUMNS, [.5, 0, 0, 0, .5]))}
            with tempfile.TemporaryDirectory() as directory:
                (Path(directory)/'searches').mkdir()
                for interaction, label in [(0., 0), (4., 1)]:
                    experiment.THERMO = AnalyticMixture(interaction)
                    result = experiment.calculate_reference(recipe, 16, 88, directory)
                    self.assertEqual(result['instability_label'], label)
                    if label:
                        w = np.array([result[f'best_trial_x_{n}'] for n in common.MOLECULES])
                        r = np.array([result[f'remaining_x_{n}'] for n in common.MOLECULES])
                        f = result['trial_fraction']
                        np.testing.assert_allclose(f*w+(1-f)*r, [.5, 0, 0, 0, .5], atol=1e-12)
                        self.assertLess(result['split_change_kJ_mol'], 0)
        finally:
            experiment.MODEL, experiment.THERMO, experiment.STABILITY = old

    def test_nonconvergence_does_not_become_a_negative_label(self):
        trials = pd.DataFrame({'converged': [False], 'lowest_tpd_over_RT': [0.],
                               'final_tpd_over_RT': [0.]})
        label, status, _ = experiment.classify_search(0., trials, None, 1e-6)
        self.assertIsNone(label)
        self.assertEqual(status, 'needs_review')

    def test_pure_component_is_separate_from_a_search(self):
        recipe = {'recipe_id': 'ood_0001', **dict(zip(common.X_COLUMNS, [1, 0, 0, 0, 0]))}
        result = experiment.calculate_reference(recipe, 40, 88, '.')
        self.assertEqual(result['label_status'], 'pure_component_check')
        self.assertEqual(result['starts'], 0)

    def test_one_class_metrics_do_not_claim_perfect_ranking(self):
        result = experiment.safe_metrics(np.ones(4, int), [.4, .1, .9, .6], .5)
        self.assertIsNone(result['average_precision'])
        self.assertEqual(result['fn'], 2)

    def test_report_excludes_unresolved_labels_and_keeps_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            references = pd.DataFrame({
                'recipe_id': ['a', 'b', 'c', 'd'],
                'test_group': ['broad_ood', 'broad_ood', 'broad_ood', 'binary_water_benzene'],
                'instability_label': [1, 0, None, 0],
                'label_status': ['instability_found', 'no_instability_found', 'needs_review', 'no_instability_found']})
            predictions = pd.DataFrame({'recipe_id': references.recipe_id,
                'model_id': 'p3_test', 'score': [.9, .1, .2, .7],
                'threshold': .5, 'predicted': [1, 0, 0, 1]})
            common.write_csv(directory/'reference_labels.csv', references)
            common.write_csv(directory/'frozen_predictions.csv', predictions)
            common.write_json(directory/'references_complete.json', {
                'reference_labels_sha256': common.sha256(directory/'reference_labels.csv')})
            common.write_json(directory/'models.json', [{'model_id': 'p3_test',
                'source_project': 3, 'source_training_labels': 200}])
            with patch.object(experiment, 'validate_run', return_value={}), redirect_stdout(io.StringIO()):
                experiment.report(SimpleNamespace(run_dir=directory))
            metrics = pd.read_csv(directory/'report/metrics_by_domain.csv').set_index('test_group')
            self.assertEqual(metrics.loc['broad_ood', 'recipes'], 2)
            self.assertEqual(metrics.loc['broad_ood', 'unresolved'], 1)
            self.assertEqual(metrics.loc['broad_ood', 'average_precision'], 1.)
            self.assertTrue(pd.isna(metrics.loc['binary_water_benzene', 'average_precision']))
            self.assertEqual(metrics.loc['binary_water_benzene', 'fp'], 1)


if __name__ == '__main__':
    unittest.main()
