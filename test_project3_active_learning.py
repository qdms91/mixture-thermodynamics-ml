"""Focused checks for query isolation, paired initialization and preprocessing.

Run: python -m unittest test_project3_active_learning.py
These tests use synthetic data and do not change the project's prepared data.
"""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import project3_common as al
import project2_common as common
from run_active_learning import run_trajectory


class ActiveLearningChecks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(51)
        self.x5 = rng.dirichlet(np.ones(5), size=80)
        self.selected = al.initial_positions(len(self.x5), 20, 11)
        self.labels = np.array([0, 1] * 40)
        self.config = {"committee": 2, "layers": [8], "alpha": .01, "max_epochs": 15}

    def test_hidden_answers_cannot_change_first_batch(self):
        # Change every unqueried answer. Fit data, scaler, member predictions
        # and the next acquisition must remain identical, since those answers
        # have not been requested. This checks actual fitting, not just a signature.
        other = self.labels.copy()
        remaining = np.setdiff1d(np.arange(len(other)), self.selected)
        other[remaining] = 1 - other[remaining]
        models, oracles = [], []
        for labels in [self.labels, other]:
            oracle = al.LabelOracle(labels)
            y = oracle.reveal(self.selected)
            fitted, _ = al.fit_committee(self.x5[self.selected, :4], y, 11, self.config)
            models.append(al.committee_scores(fitted, self.x5[:, :4]))
            oracles.append(oracle)
            expected_mean = self.x5[self.selected, :4].mean(axis=0)
            np.testing.assert_allclose(fitted["models"][0][0].mean_, expected_mean)
        np.testing.assert_array_equal(models[0], models[1])
        for strategy in al.STRATEGIES:
            a = al.choose_batch(strategy, self.x5, self.selected, 10, 11, models[0])
            b = al.choose_batch(strategy, self.x5, self.selected, 10, 11, models[1])
            self.assertEqual(a, b)
            self.assertEqual(len(set(a)), 10)
            self.assertFalse(set(a).intersection(self.selected))
        for oracle in oracles:
            self.assertEqual(oracle.queried, set(self.selected))

    def test_initial_samples_are_label_blind_and_repeatable(self):
        a = al.initial_positions(80, 20, 11)
        self.assertEqual(a, al.initial_positions(80, 20, 11))
        self.assertNotEqual(a, al.initial_positions(80, 20, 22))
        self.assertEqual(len(set(a)), 20)

    def test_last_batch_can_exhaust_pool_without_duplicates(self):
        selected = list(range(73))
        constant = np.zeros((2, 80))
        for strategy in al.STRATEGIES:
            chosen = al.choose_batch(strategy, self.x5, selected, 7, 11, constant)
            self.assertEqual(set(chosen), set(range(73, 80)))

    def test_one_class_fallback_does_not_inspect_hidden_labels(self):
        package, diagnostics = al.fit_committee(self.x5[:10, :4], np.zeros(10, int), 11, self.config)
        self.assertEqual(diagnostics["constant_class"], 0)
        self.assertEqual(len(package["models"]), 0)
        scores = al.committee_scores(package, self.x5[:, :4])
        self.assertTrue((scores == 0).all())
        batch = al.choose_batch("uncertainty_diversity", self.x5, list(range(10)), 5, 11, scores)
        self.assertEqual(len(set(batch)), 5)
        self.assertFalse(set(batch).intersection(range(10)))

    def test_interrupted_path_resumes_with_the_same_queries_and_predictions(self):
        # Interrupt after the first committed batch, then compare its resumed
        # result with uninterrupted execution. Runtime counters may differ;
        # selected recipes and model predictions must be identical.
        pool = pd.DataFrame(self.x5, columns=common.X_COLUMNS)
        pool["recipe_id"] = [f"recipe_{i}" for i in range(len(pool))]
        pool["block_id"] = np.arange(len(pool))
        pool["split"] = "train"
        heldout = pool.iloc[:8].copy().assign(split="validation")
        heldout["recipe_id"] = [f"heldout_{i}" for i in range(len(heldout))]
        config = {**self.config, "initial": 20, "budgets": [20, 40], "batch": 10}
        original_fit = al.fit_committee
        calls = 0

        def interrupt_second_fit(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt
            return original_fit(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            straight, resumed = Path(temp) / "straight", Path(temp) / "resumed"
            arguments = (pool, self.labels, np.ones(len(pool)), heldout,
                         config, "uncertainty_diversity", 11)
            run_trajectory(*arguments, straight)
            with patch.object(al, "fit_committee", side_effect=interrupt_second_fit):
                with self.assertRaises(KeyboardInterrupt):
                    run_trajectory(*arguments, resumed)
            self.assertFalse(json.loads((resumed / "state.json").read_text())["complete"])
            run_trajectory(*arguments, resumed)
            for filename in ["queries.csv", "predictions_0040.csv"]:
                pd.testing.assert_frame_equal(pd.read_csv(straight / filename),
                                              pd.read_csv(resumed / filename))
            state = json.loads((resumed / "state.json").read_text())
            self.assertTrue(state["complete"])
            self.assertEqual([c["training_labels"] for c in state["checkpoints"]], [20, 40])


if __name__ == "__main__":
    unittest.main()
