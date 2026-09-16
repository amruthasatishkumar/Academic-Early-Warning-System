import copy
import json
import unittest
from pathlib import Path
import types
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

from aews.configuration import load_configuration
from aews.core.features import CLASSES, FEATURES, labels, weekly_features
from aews.core.simulation import DAY, generate_course
from aews.execution import development_fit, split_features
from aews.model.modeling import predictors, score_current
from aews.presentation.explainability import fitted_linear_pipeline
from aews.presentation.reader import read_publication
from aews.runtime import PURPOSE

ROOT = Path(__file__).resolve().parents[1]


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source, cls.config, cls.contract = load_configuration(ROOT / "config")
        cls.config["logistic_C"] = [0.3]
        cls.config["class_weight_candidates"] = [None]
        frames = []
        for term in range(1, 6):
            for course in (1, 2):
                batch = generate_course(cls.source, cls.contract, term, course)
                features = weekly_features(batch, 4)
                if term < 5:
                    features = features.merge(labels(batch, batch.start + 28 * DAY), on="enrollment_id",
                                              validate="one_to_one")
                else:
                    features["label"] = None
                    features["label_status"] = "current_no_truth"
                    features["label_available_at"] = pd.NaT
                frames.append(features)
        cls.features = pd.concat(frames, ignore_index=True)
        cls.result = development_fit(cls.features, cls.config, "unit-test-code")
        cls.history, cls.current = split_features(cls.features)
        cls.training = cls.history.loc[cls.history.term.isin([1, 2]) & cls.history.label.notna()]
        cls.stamp = pd.Timestamp("2026-01-01", tz="UTC")

    def test_four_labels_and_unknowns_are_not_coerced(self):
        self.assertEqual(sorted(self.training.label.unique()), CLASSES)
        self.assertGreater(self.history.label.isna().sum(), 0)
        self.assertEqual(self.result.split_manifest["train_rows"], len(self.training))
        self.assertFalse(any("label" in column for column in self.current))

    def test_training_only_transformations(self):
        pipeline = fitted_linear_pipeline(self.result.model)
        np.testing.assert_allclose(pipeline.named_steps["imputer"].statistics_,
                                   predictors(self.training).median().fillna(0).to_numpy())
        imputed = pipeline.named_steps["imputer"].transform(predictors(self.training))
        np.testing.assert_allclose(pipeline.named_steps["scaler"].mean_, imputed.mean(axis=0))
        split = self.result.split_manifest
        self.assertFalse(set(split["calibration_enrollment_ids"]) & set(split["decision_enrollment_ids"]))
        self.assertFalse(set(self.training.enrollment_id) & set(split["calibration_enrollment_ids"]))

    def test_exposed_diagnostic_never_changes_selection(self):
        changed = self.features.copy()
        mask = changed.term.isin([4, 5])
        changed.loc[mask, FEATURES] = 9999.
        changed.loc[mask, "label"] = "withdrawn"
        result = development_fit(changed, self.config, "unit-test-code")
        self.assertEqual(self.result.version, result.version)
        self.assertEqual(self.result.selection, result.selection)
        self.assertFalse(result.selection["fresh_independent_holdout_available"])

    def test_scoring_idempotence_probability_keys_and_exact_evidence(self):
        predictions, evidence = score_current(self.result, self.current, self.stamp)
        again, again_evidence = score_current(self.result, self.current, self.stamp)
        pd.testing.assert_frame_equal(predictions, again)
        pd.testing.assert_frame_equal(evidence, again_evidence)
        np.testing.assert_allclose(predictions[["p_" + c for c in CLASSES]].sum(axis=1), 1)
        self.assertFalse(predictions.prediction_id.duplicated().any())
        raw = fitted_linear_pipeline(self.result.model).decision_function(predictors(self.current))
        pipeline_classes = list(self.result.model.classes_)
        for index, record in enumerate(evidence.explanation_json):
            item = json.loads(record)
            total = item["baseline_score"] + sum(item["contributions"].values())
            self.assertAlmostEqual(total, item["full_score"])
            self.assertAlmostEqual(total, raw[index, pipeline_classes.index(item["outcome"])])
        self.assertFalse(self.result.registration_eligible)
        self.assertEqual(len(evidence), len(self.current))
        self.assertTrue(self.result.model_name.startswith("logistic_"))

    def test_current_labels_and_wrong_terms_are_rejected(self):
        contaminated = self.current.assign(label=None)
        with self.assertRaisesRegex(ValueError, "outcome columns"):
            score_current(self.result, contaminated, self.stamp)
        with self.assertRaisesRegex(ValueError, "current term"):
            score_current(self.result, self.current.assign(term=4), self.stamp)
        changed = copy.deepcopy(self.config)
        changed["registration_approved"] = True
        with self.assertRaisesRegex(ValueError, "registration"):
            development_fit(self.features, changed, "unit-test-code")

    def test_read_only_publication_reproduces_model_and_training_reference(self):
        predictions, evidence = score_current(self.result, self.current, self.stamp, "local-test-run")
        frames = {
            "silver.historical_weekly_features": self.history,
            "silver.current_weekly_features": self.current,
            "gold.student_course_predictions": predictions,
            "gold.prediction_explanations": evidence,
            "gold.model_evaluation": pd.DataFrame({"metric": [1]}),
            "gold.data_quality_summary": pd.DataFrame({"assertion": ["passed"]}),
        }
        root = "Files/model_artifacts/aews-test/publications/test"
        paths = {key: root + "/tables/" + key.replace(".", "/") for key in frames}
        manifest = {
            "purpose": PURPOSE, "publication_state": "complete", "source_profile": "full", "registered": False,
            "model_version": self.result.version, "mlflow_run_id": "local-test-run",
            "code_sha256": "test-code", "train_data_sha256": self.result.train_fingerprint,
            "source_dataset_version": "test-source", "publication_archive": root,
            "schemas": {"silver": "silver", "gold": "gold"}, "table_paths": paths,
            "table_counts": {key: len(value) for key, value in frames.items()},
            "current_predictions": len(predictions), "workspace_name": self.source["intended_workspace"],
            "lakehouse_name": self.source["intended_lakehouse"],
        }
        class Frame:
            def __init__(self, frame):
                self.frame = frame

            def count(self):
                return len(self.frame)

            def toPandas(self):
                return self.frame.copy()
        by_path = {paths[name]: Frame(frame) for name, frame in frames.items()}
        spark = types.SimpleNamespace(read=types.SimpleNamespace(
            format=lambda _: types.SimpleNamespace(load=lambda path: by_path[path])))
        notebookutils = types.SimpleNamespace(fs=types.SimpleNamespace(
            exists=lambda _: True, head=lambda *_: json.dumps(manifest)))
        mlflow = types.ModuleType("mlflow")
        mlflow.__path__ = []
        mlflow.sklearn = types.ModuleType("mlflow.sklearn")
        mlflow.sklearn.load_model = lambda _: self.result.model
        with patch.dict(sys.modules, {"notebookutils": notebookutils, "mlflow": mlflow, "mlflow.sklearn": mlflow.sklearn}), \
             patch("aews.presentation.reader.validate_session", return_value=(None, {})):
            view = read_publication(spark, self.source)
        self.assertEqual(len(view["predictions"]), len(self.current))
        self.assertTrue(view["predictions"].top_predicted_class_drivers.notna().all())
