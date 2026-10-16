"""Focused checks for source weights, target splits, and acquisition fairness."""
import unittest
from copy import deepcopy
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import project4_adaptation as app
import project4_learning as learn


class AdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(51)
        cls.x = rng.dirichlet(np.ones(5), 70)
        cls.y = (cls.x[:, 0] > .2).astype(int)
        cls.pipeline = make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(128, 64), solver='lbfgs', max_iter=500, random_state=12))
        cls.pipeline.fit(cls.x[:, :4], cls.y)
        cls.sources = learn.source_networks({'models': [cls.pipeline]})
        cls.config = {'epochs': 3, 'min_epochs': 3, 'patience': 4, 'tol': 1e-5,
                      'alpha': .01, 'learning_rate': .001, 'budgets': [12, 18], 'batch': 6}

    def test_source_coordinate_change(self):
        x = np.vstack([self.x, np.eye(5)])
        old = self.pipeline.predict_proba(x[:, :4])
        new = self.sources[0].predict_proba(learn.coordinates(x))
        np.testing.assert_allclose(old, new, atol=2e-14, rtol=1e-11)

    def test_fit_isolated_and_one_class_allowed(self):
        old = deepcopy(self.sources[0].coefs_)
        a = learn.fit_committee(self.x[:12], np.zeros(12, int), self.sources, True, 8, self.config)
        b = learn.fit_committee(self.x[:12], np.zeros(12, int), self.sources, False, 8, self.config)
        self.assertTrue(np.isfinite(learn.scores(a, self.x)).all())
        self.assertTrue(np.isfinite(learn.scores(b, self.x)).all())
        for left, right in zip(old, self.sources[0].coefs_):
            np.testing.assert_array_equal(left, right)

    def test_split_counts_and_identity(self):
        data = app.make_target(20261008)
        self.assertEqual(data.split.value_counts().to_dict(), {'train': 600, 'validation': 200, 'test': 200})
        self.assertEqual(data.groupby('block_id').split.nunique().max(), 1)
        self.assertEqual(data[app.common.X_COLUMNS].round(12).drop_duplicates().shape[0], 1000)
        np.testing.assert_allclose(data[app.common.X_COLUMNS].sum(axis=1), 1, atol=1e-14)
        self.assertEqual(len(data[data.components == 2].test_group.unique()), 10)
        self.assertTrue((data[app.common.X_COLUMNS].to_numpy() >= 0).all())
        pd.testing.assert_frame_equal(data, app.make_target(20261008))

    def test_paths_resume_random_pair_and_matched_control(self):
        pool = pd.DataFrame(self.x, columns=app.common.X_COLUMNS)
        pool['recipe_id'] = [f'id{i}' for i in range(len(pool))]
        pool['instability_label'] = self.y
        pool['stability_seconds'] = 1.
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            scratch = app.path_run(directory, 'scratch_random', 5, pool, self.sources, self.config)
            transfer = app.path_run(directory, 'transfer_random', 5, pool, self.sources, self.config)
            self.assertEqual([r['positions'] for r in scratch], [r['positions'] for r in transfer])
            replay = app.path_run(directory, 'scratch_random', 5, pool, self.sources, self.config)
            self.assertEqual([r['model_sha256'] for r in scratch], [r['model_sha256'] for r in replay])
            active = app.path_run(directory, 'transfer_active', 5, pool, self.sources, self.config)
            matched = app.path_run(directory, 'scratch_matched_transfer_active', 5, pool, self.sources,
                                   self.config, directory / 'seed_5/transfer_active')
            self.assertEqual([r['positions'] for r in active], [r['positions'] for r in matched])

    def test_hidden_answers_do_not_change_next_batch(self):
        chosen = list(range(12))
        a = learn.fit_committee(self.x[chosen], self.y[chosen], self.sources, True, 5, self.config)
        changed = self.y.copy(); changed[12:] = 1 - changed[12:]
        b = learn.fit_committee(self.x[chosen], changed[chosen], self.sources, True, 5, self.config)
        self.assertEqual(learn.select_next('active', self.x, chosen, 6, 5, a),
                         learn.select_next('active', self.x, chosen, 6, 5, b))


if __name__ == '__main__':
    unittest.main()
