import ast
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aews.presentation.explainability import OUTCOMES, OutcomeExplanations

ROOT = Path(__file__).resolve().parents[1]


class ExplanationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(12)
        cls.frame = pd.DataFrame({"grade": rng.normal(size=100), "activity": rng.normal(size=100)})
        cls.frame.loc[::9, "activity"] = np.nan
        labels = np.tile(OUTCOMES, 25)
        cls.pipeline = Pipeline([
            ("imputer", SimpleImputer(add_indicator=True, keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=100)),
        ]).fit(cls.frame.iloc[:60], labels[:60])
        cls.model = CalibratedClassifierCV(cls.pipeline, cv="prefit").fit(cls.frame.iloc[60:80], labels[60:80])
        cls.explainer = OutcomeExplanations(cls.model, cls.frame.iloc[:60])

    def test_all_four_outcomes_get_global_feature_views(self):
        view = self.explainer.training_importance(self.frame.iloc[:60])
        self.assertEqual(set(view.outcome), set(OUTCOMES))
        self.assertEqual(len(view), 8)
        self.assertTrue((view.mean_absolute_score_contribution >= 0).all())
        self.assertTrue(np.isfinite(view.mean_absolute_score_contribution).all())

    def test_global_values_equal_grouped_training_score_formula(self):
        training = self.frame.iloc[:60]
        actual = self.explainer.training_importance(training).set_index(["outcome", "feature"])
        centered = self.explainer.training_z - self.explainer.reference
        for outcome in OUTCOMES:
            class_index = self.explainer.classes.index(outcome)
            ungrouped = centered * self.explainer.coefficients[class_index]
            for feature_index, feature in enumerate(self.explainer.features):
                positions = [j for j, group in enumerate(self.explainer.groups) if group == feature_index]
                expected = np.abs(ungrouped[:, positions].sum(axis=1)).mean()
                self.assertAlmostEqual(actual.loc[(outcome, feature), "mean_absolute_score_contribution"], expected)

    def test_contributions_reconstruct_each_underlying_score(self):
        rows = self.frame.iloc[80:]
        scores = self.pipeline.decision_function(rows)
        for outcome in OUTCOMES:
            terms = self.explainer.contributions(rows, outcome)
            index = self.explainer.classes.index(outcome)
            np.testing.assert_allclose(terms.sum(axis=1) + self.explainer.baseline_scores[index], scores[:, index])

    def test_missing_indicators_group_into_original_feature(self):
        terms = self.explainer.contributions(self.frame.iloc[80:], "withdrawn")
        self.assertEqual(list(terms), ["grade", "activity"])
        self.assertTrue(np.isfinite(terms.to_numpy()).all())

    def test_current_data_does_not_change_training_reference(self):
        before = self.explainer.reference.copy()
        changed = self.frame.iloc[80:].copy()
        changed["grade"] = 9999
        self.explainer.contributions(changed, "pass")
        np.testing.assert_array_equal(before, self.explainer.reference)

    def test_every_result_has_predicted_class_drivers(self):
        current = self.frame.iloc[80:]
        probabilities = self.model.predict_proba(current)
        predicted = pd.DataFrame({"p_" + c: probabilities[:, i] for i, c in enumerate(self.model.classes_)})
        predicted["predicted_class"] = self.model.classes_[probabilities.argmax(axis=1)]
        drivers = self.explainer.result_drivers(current, predicted)
        self.assertEqual(len(drivers), len(current))
        self.assertTrue(all(len(row) == 2 for row in drivers))
        predicted.loc[0, "p_pass"] += .1
        with self.assertRaises(AssertionError):
            self.explainer.result_drivers(current, predicted)

    def test_unknown_outcome_is_rejected(self):
        with self.assertRaises(ValueError):
            self.explainer.contributions(self.frame.iloc[80:], "dropout")
