from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import zipfile

import pandas as pd
import pyarrow as pa

from aews.configuration import load_configuration
from aews.data_bundle import FILENAME, load_bundle, load_lakehouse_input
from aews.fabric import prepare_input, publish_sources, run_fabric, validate_uploaded_data
from aews.runtime import qualified
from aews.spark_tables import _arrow_to_spark
from aews.table_import import _delta_frame, create_input_tables, load_folder_tables, table_plan


ROOT = Path(__file__).resolve().parents[1]


class FolderInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source, cls.model, cls.contract = load_configuration(ROOT / "config")
        cls.info = json.loads((ROOT / "sample-data" / "bundle-info.json").read_text())
        cls.zip_path = ROOT / "sample-data" / FILENAME

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.files = Path(self.temporary.name) / "Files"
        self.folder = self.files / "data-fixed"
        with zipfile.ZipFile(self.zip_path) as archive:
            archive.extractall(self.folder)

    def read_folder(self):
        return load_lakehouse_input(self.files, "folder", "Files/data-fixed",
                                    self.info, self.source, self.contract)

    def test_folder_and_zip_have_identical_records_and_all_27_targets(self):
        folder = self.read_folder()
        zipped = load_bundle(self.zip_path, self.info, self.source, self.contract)
        pd.testing.assert_frame_equal(folder.features, zipped.features)
        self.assertEqual(folder.manifest["verified_input_kind"], "folder")
        self.assertNotIn("verified_zip_sha256", folder.manifest)
        self.assertEqual(folder.manifest["verified_manifest_sha256"], self.info["manifest_sha256"])
        plan = table_plan(folder, self.contract)
        self.assertEqual(len(plan), 27)
        self.assertEqual(plan["silver.current_weekly_features"].num_rows, 583)
        self.assertEqual(plan["silver.historical_weekly_features"].num_rows, 2336)
        self.assertEqual(plan["canvas.submissions"].num_rows, 60000)
        self.assertNotIn("label", plan["silver.current_weekly_features"].column_names)
        for name, table in folder.tables.items():
            self.assertTrue(table.equals(zipped.tables[name], check_metadata=False), name)

    def test_folder_prepares_training_without_zip_or_generation(self):
        folder = self.read_folder()
        with patch("aews.fabric.prepare", side_effect=AssertionError("Must not generate")):
            features, summary = prepare_input(
                self.source, self.contract, self.files / "immutable-sources", folder)
        pd.testing.assert_frame_equal(features, folder.features)
        self.assertEqual(summary["input_mode"], "folder_verified_synthetic_bundle")
        self.assertTrue((self.files / "immutable-sources" / "canvas" / "submissions" / "part-05.parquet").is_file())

    def test_training_validation_accepts_same_folder_path(self):
        with patch("aews.fabric.validate_session", return_value=(self.files, {})):
            result = validate_uploaded_data(
                None, self.source, self.model, self.contract, "Files/data-fixed", self.info, "folder")
        self.assertEqual(len(result.features), 2919)

    def test_folder_publish_selects_small_profile_and_same_verified_input(self):
        source = dict(self.source, mode="full")
        with patch("aews.fabric.runtime_preflight", return_value=(self.files, {})) as preflight, \
             patch("aews.fabric.prepare_input", side_effect=RuntimeError("stop before training")) as prepare, \
             patch.dict("sys.modules", {"mlflow": types.SimpleNamespace(autolog=Mock()),
                                        "notebookutils": types.SimpleNamespace(fs=Mock())}):
            with self.assertRaisesRegex(RuntimeError, "stop before training"):
                run_fabric(Mock(), source, self.model, self.contract, "test", True,
                           "folder", "Files/data-fixed", self.info)
        self.assertEqual(preflight.call_args.args[1]["mode"], "small")
        self.assertTrue(preflight.call_args.kwargs["uploaded"])
        self.assertEqual(prepare.call_args.args[3].manifest["verified_input_kind"], "folder")
        self.assertEqual(source["mode"], "full")

    def test_changed_manifest_is_rejected_even_with_self_consistent_new_members(self):
        manifest = self.folder / "manifest.json"
        content = json.loads(manifest.read_text())
        content["members"][0]["sha256"] = "0" * 64
        manifest.write_text(json.dumps(content))
        with self.assertRaisesRegex(ValueError, "manifest checksum mismatch"):
            self.read_folder()

    def test_changed_file_is_rejected_before_any_table_creation(self):
        path = self.folder / "features" / "current.parquet"
        raw = bytearray(path.read_bytes())
        raw[len(raw) // 2] ^= 1
        path.write_bytes(raw)
        with patch("aews.table_import.validate_session", return_value=(self.files, {})), \
             patch("aews.table_import.create_input_tables") as create:
            with self.assertRaisesRegex(ValueError, "member checksum mismatch"):
                load_folder_tables(None, self.source, self.contract, "Files/data-fixed", self.info)
        create.assert_not_called()

    def test_missing_part_and_extra_file_are_rejected(self):
        part = self.folder / "source_parquet" / "canvas" / "submissions" / "part-05.parquet"
        contents = part.read_bytes()
        part.unlink()
        with self.assertRaisesRegex(ValueError, "Missing, duplicated or unlisted"):
            self.read_folder()
        part.write_bytes(contents)
        (self.folder / "unexpected.txt").write_text("not part of the dataset")
        with self.assertRaisesRegex(ValueError, "Missing, duplicated or unlisted"):
            self.read_folder()

    def test_missing_wrongly_nested_and_escaping_paths_are_rejected(self):
        for path in ("Files/missing", "Files/data-fixed/features"):
            with self.subTest(path=path), self.assertRaises((FileNotFoundError, ValueError)):
                load_lakehouse_input(self.files, "folder", path, self.info, self.source, self.contract)
        for path in ("Files/../data", "Files//data", "Files/data\\other", "/tmp/data", "Files/"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                load_lakehouse_input(self.files, "folder", path, self.info, self.source, self.contract)

    def test_source_changes_after_validation_cannot_be_published(self):
        folder = self.read_folder()
        path = self.folder / "source_parquet" / "canvas" / "users" / "part-00.parquet"
        contents = bytearray(path.read_bytes())
        contents[len(contents) // 2] ^= 1
        path.write_bytes(contents)
        with self.assertRaisesRegex(ValueError, "source checksum mismatch"):
            folder.extract_sources(self.files / "staging")

    def test_empty_input_plan_is_rejected(self):
        bundle = self.read_folder()
        bundle.tables = {}
        with self.assertRaisesRegex(ValueError, "missing expected input tables"):
            table_plan(bundle, self.contract)


class FakeSpark:
    def __init__(self, existing=None, fail=None):
        self.tables = dict(existing or {})
        self.statements = []
        self.writes = []
        self.fail = fail
        self.catalog = types.SimpleNamespace(tableExists=lambda name: name in self.tables)

    def table(self, name):
        return self.tables[name]

    def sql(self, statement):
        self.statements.append(statement)


class FakeFrame:
    def __init__(self, spark, data):
        self.spark, self.data = spark, data
        self.write = self

    def format(self, value):
        assert value == "delta"
        return self

    def mode(self, value):
        assert value == "errorifexists"
        return self

    def saveAsTable(self, name):
        if name in self.spark.tables:
            raise RuntimeError("Concurrent table already exists")
        if name == self.spark.fail:
            raise RuntimeError("Write failed")
        self.spark.tables[name] = self
        self.spark.writes.append(name)


def equivalent(left, right, description):
    if not left.data.equals(right.data, check_metadata=False):
        raise RuntimeError("Row content mismatch: " + description)
    return right.data.num_rows


class TableCreationTests(unittest.TestCase):
    def setUp(self):
        self.config = {"intended_lakehouse": "academic_early_warning"}
        self.plan = {"canvas.users": pa.table({"id": [1, 2]}),
                     "silver.current_weekly_features": pa.table({"enrollment_id": [3]})}

    def load(self, spark):
        with patch("aews.table_import._delta_frame", side_effect=lambda s, a: FakeFrame(s, a)), \
             patch("aews.table_import.require_equivalent", side_effect=equivalent):
            return create_input_tables(spark, self.config, self.plan)

    def test_create_once_then_verify_and_skip_on_rerun(self):
        spark = FakeSpark()
        with redirect_stdout(io.StringIO()):
            created = self.load(spark)
            repeated = self.load(spark)
        self.assertEqual(len(spark.writes), 2)
        self.assertEqual([row["status"] for row in created], ["created", "created"])
        self.assertEqual([row["status"] for row in repeated], ["already_matches", "already_matches"])
        self.assertEqual([row["rows"] for row in repeated], [2, 1])

    def test_mismatching_existing_table_blocks_all_new_writes(self):
        name = qualified(self.config, "silver", "current_weekly_features")
        spark = FakeSpark()
        spark.tables[name] = FakeFrame(spark, pa.table({"enrollment_id": [999]}))
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "Row content mismatch"):
            self.load(spark)
        self.assertEqual(spark.writes, [])
        self.assertEqual(spark.statements, [])

    def test_partial_failure_is_visible_and_can_resume_without_overwriting(self):
        fail = qualified(self.config, "silver", "current_weekly_features")
        spark = FakeSpark(fail=fail)
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaisesRegex(RuntimeError, "Write failed"):
            self.load(spark)
        self.assertNotIn("tables_loaded", output.getvalue())
        self.assertEqual(len(spark.writes), 1)
        spark.fail = None
        with redirect_stdout(io.StringIO()):
            result = self.load(spark)
        self.assertEqual([row["status"] for row in result], ["already_matches", "created"])
        self.assertEqual(len(spark.writes), 2)

    def test_arrow_conversion_keeps_timestamps_and_nulls(self):
        spark = Mock()
        arrow = pa.table({"id": pa.array([1, None], type=pa.int64()),
                          "time": pa.array([pd.Timestamp("2026-01-01T00:00:00.123456Z"), None],
                                           type=pa.timestamp("us", tz="UTC"))})
        with patch("aews.spark_tables._spark_schema", return_value="explicit-schema"):
            _arrow_to_spark(spark, arrow)
        rows = spark.createDataFrame.call_args.args[0]
        self.assertEqual(rows[0], (1, pd.Timestamp("2026-01-01T00:00:00.123456Z").to_pydatetime()))
        self.assertEqual(rows[1], (None, None))
        self.assertEqual(spark.createDataFrame.call_args.kwargs["schema"], "explicit-schema")

    def test_import_uses_delta_reader_nullability_without_changing_source_types(self):
        arrow = pa.table({"id": [1], "values": [["a"]]})
        spark = Mock()
        with patch("aews.table_import._arrow_to_spark") as convert:
            _delta_frame(spark, arrow)
        convert.assert_called_once_with(spark, arrow, nullable=True)

    def test_model_publication_sources_also_match_delta_reader_nullability(self):
        arrow_schema = pa.schema([pa.field("id", pa.int64(), nullable=False)])
        spark = Mock()
        spark.table.return_value.count.return_value = 1
        with patch("aews.fabric.source_targets", return_value=[("canvas", "accounts", arrow_schema)]), \
             patch("aews.fabric._spark_schema", return_value="delta-nullable") as schema, \
             patch("aews.fabric.require_equivalent") as verify:
            publish_sources(spark, self.config, {}, "Files/data/example", {"accounts": 1})
        schema.assert_called_once_with(arrow_schema, nullable=True)
        spark.read.schema.assert_called_once_with("delta-nullable")
        self.assertEqual(spark.createDataFrame.call_args.kwargs["schema"], "delta-nullable")
        verify.assert_called_once_with(spark.createDataFrame.return_value, spark.table.return_value,
                                       "canvas.accounts")


if __name__ == "__main__":
    unittest.main()
