import base64
import copy
import fnmatch
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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

    def test_every_parquet_timestamp_is_spark_compatible(self):
        timestamp_count = 0
        with zipfile.ZipFile(self.path) as archive:
            for name in archive.namelist():
                if not name.endswith(".parquet"):
                    continue
                parquet = pq.ParquetFile(io.BytesIO(archive.read(name)))
                schema = parquet.schema
                for i in range(len(schema)):
                    logical = str(schema.column(i).logical_type).lower()
                    self.assertNotIn("nanoseconds", logical, f"{name}:{schema.column(i).path}")
                    if "timestamp(" in logical:
                        timestamp_count += 1
                        self.assertIn("microseconds", logical)
                metadata = parquet.metadata.metadata or {}
                if b"ARROW:schema" in metadata:
                    embedded = pa.ipc.read_schema(pa.BufferReader(base64.b64decode(metadata[b"ARROW:schema"])))
                    for field in embedded:
                        if pa.types.is_timestamp(field.type):
                            self.assertEqual(field.type.unit, "us", f"{name}:{field.name}:embedded Arrow")
                if b"pandas" in metadata:
                    for column in json.loads(metadata[b"pandas"])["columns"]:
                        field = parquet.schema_arrow.field(column["field_name"])
                        if pa.types.is_timestamp(field.type):
                            self.assertEqual(column["numpy_type"], "datetime64[us]", f"{name}:{field.name}:pandas")
        self.assertGreater(timestamp_count, 0)

    def test_readme_maps_every_parquet_file_to_its_destination(self):
        readme = (ROOT / "README.rst").read_text()
        rows = re.findall(
            r"^   \* - ``([^`\n]+\.parquet)``\n     - ``(\w+)``\n     - ``(\w+)``$",
            readme, flags=re.MULTILINE,
        )
        self.assertTrue(rows, "README must include the Parquet destination table.")
        features = {
            "features/current.parquet": "silver.current_weekly_features",
            "features/historical.parquet": "silver.historical_weekly_features",
        }
        expected = {
            item["path"]: item.get("table", features.get(item["path"]))
            for item in self.bundle.manifest["members"] if item["path"].endswith(".parquet")
        }
        covered = {}
        destinations = set()
        for pattern, schema, table in rows:
            target = f"{schema}.{table}"
            self.assertNotIn(target, destinations, f"Duplicate destination: {target}")
            destinations.add(target)
            matching = [name for name in expected if fnmatch.fnmatchcase(name, pattern.replace("\\", "/"))]
            self.assertTrue(matching, f"No shipped files match {pattern}")
            for name in matching:
                self.assertNotIn(name, covered, f"File mapped more than once: {name}")
                self.assertEqual(target, expected[name], f"Wrong destination for {name}")
                covered[name] = target
        self.assertEqual(covered, expected)
        self.assertIn(f"{len(expected)} Parquet files", readme)
        self.assertIn(f"**{len(destinations)} input tables**", readme)

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
