import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import pandas as pd

from aews.configuration import load_configuration
from aews.data_bundle import FILENAME, load_bundle, safe_member_name, sha256_file, validate_feature_frames
from aews.execution import finish_run, code_hash
from aews.fabric import prepare_input

ROOT = Path(__file__).resolve().parents[1]


class DataBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source, cls.model_config, cls.contract = load_configuration(ROOT / "config")
        cls.path = ROOT / "sample-data" / FILENAME
        cls.info = json.loads((ROOT / "sample-data" / "bundle-info.json").read_text())
        cls.bundle = load_bundle(cls.path, cls.info, cls.source, cls.contract)

    def test_bundle_has_source_data_and_current_features_without_labels(self):
        self.assertEqual(self.bundle.summary["observed_source_counts"]["submissions"], 60000)
        self.assertEqual(self.bundle.summary["observed_source_counts"]["users"], 408)
        self.assertEqual(len(self.bundle.features.loc[self.bundle.features.term == 5]), 583)
        self.assertTrue(self.bundle.features.loc[self.bundle.features.term == 5, "label"].isna().all())
        self.assertFalse(any("model.pkl" in m["path"] for m in self.bundle.manifest["members"]))

    def test_upload_path_does_not_generate_records(self):
        with tempfile.TemporaryDirectory() as temporary, patch("aews.fabric.prepare", side_effect=AssertionError("No generation")):
            destination = Path(temporary) / "sources"
            features, summary = prepare_input(self.source, self.contract, destination, self.bundle)
            pd.testing.assert_frame_equal(features, self.bundle.features)
            self.assertEqual(summary["profile"], "small")
            self.assertTrue((destination / "canvas" / "submissions" / "part-01.parquet").is_file())
            with self.assertRaises(FileExistsError):
                self.bundle.extract_sources(destination)

    def test_included_features_train_score_and_explain_without_generation(self):
        with patch("aews.fabric.prepare", side_effect=AssertionError("No generation")):
            result, summary, tables, importance = finish_run(
                self.bundle.features, copy.deepcopy(self.bundle.summary), self.model_config,
                code_hash(ROOT), pd.Timestamp("2026-01-01", tz="UTC"))
        self.assertEqual(len(tables["gold.student_course_predictions"]), 583)
        self.assertEqual(len(tables["gold.prediction_explanations"]), 583)
        self.assertEqual(len(importance), 64)
        self.assertFalse(summary["registered"])

    def test_wrong_outer_checksum_rejected(self):
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            load_bundle(self.path, dict(self.info, zip_sha256="0" * 64), self.source, self.contract)

    def test_changed_configuration_rejected(self):
        with self.assertRaisesRegex(ValueError, "profile/seed"):
            load_bundle(self.path, self.info, dict(self.source, seed=1), self.contract)

    def test_future_label_columns_rejected(self):
        historical = self.bundle.features.loc[self.bundle.features.term < 5].copy()
        current = self.bundle.features.loc[self.bundle.features.term == 5].copy()
        with self.assertRaisesRegex(ValueError, "current feature columns"):
            validate_feature_frames(historical, current)

    def test_unsafe_archive_paths_rejected(self):
        for path in ("../model.pkl", "/tmp/file", "C:/file", "nested\\file", "a/../b", "a//b", "./file"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                safe_member_name(path)

    def test_unlisted_member_rejected_even_with_updated_outer_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / FILENAME
            with zipfile.ZipFile(self.path) as original, zipfile.ZipFile(path, "w") as altered:
                for item in original.infolist():
                    altered.writestr(item, original.read(item.filename))
                altered.writestr("unlisted.txt", "unexpected")
            info = dict(self.info, zip_sha256=sha256_file(path), zip_bytes=path.stat().st_size)
            with self.assertRaisesRegex(ValueError, "unlisted"):
                load_bundle(path, info, self.source, self.contract)

    def test_member_tampering_rejected_even_with_updated_outer_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / FILENAME
            with zipfile.ZipFile(self.path) as original, zipfile.ZipFile(path, "w") as altered:
                for item in original.infolist():
                    content = original.read(item.filename)
                    if item.filename == "summary.json":
                        content = content.replace(b'"seed": 271828', b'"seed": 271829')
                    altered.writestr(item, content)
            info = dict(self.info, zip_sha256=sha256_file(path), zip_bytes=path.stat().st_size)
            with self.assertRaisesRegex(ValueError, "member checksum"):
                load_bundle(path, info, self.source, self.contract)


if __name__ == "__main__":
    unittest.main()
