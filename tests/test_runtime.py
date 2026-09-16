from contextlib import ExitStack, redirect_stdout
from importlib.metadata import PackageNotFoundError
import io
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch

from aews.runtime import validate_session
from aews.table_import import load_folder_tables


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.config = {"intended_workspace": "Academic Early Warning System",
                       "intended_lakehouse": "academic_early_warning"}
        self.context = {
            "currentWorkspaceName": self.config["intended_workspace"],
            "defaultLakehouseName": self.config["intended_lakehouse"],
            "defaultLakehouseWorkspaceName": self.config["intended_workspace"],
        }
        self.spark = Mock(version="3.5.5")
        self.spark.sql.return_value.collect.return_value = [
            ("Academic Early Warning System.academic_early_warning.dbo",)]
        self.versions = {"numpy": "1.26.4", "pandas": "2.1.4",
                         "scikit-learn": "1.2.2", "pyarrow": "14.0.2",
                         "mlflow": "2.12.2", "delta-spark": "3.2.0"}
        self.active_session = self.spark
        self.module_version = "2.12.2"
        self.missing_module = None
        self.output = io.StringIO()

    def get_version(self, name):
        if name not in self.versions:
            raise PackageNotFoundError(name)
        return self.versions[name]

    def import_module(self, name):
        if name == self.missing_module:
            raise ModuleNotFoundError(name)
        return types.SimpleNamespace(__version__=self.module_version)

    def run_preflight(self, **kwargs):
        notebookutils = types.ModuleType("notebookutils")
        notebookutils.runtime = types.SimpleNamespace(context=self.context)
        pyspark = types.ModuleType("pyspark")
        pyspark.__path__ = []
        sql = types.ModuleType("pyspark.sql")
        sql.SparkSession = types.SimpleNamespace(getActiveSession=lambda: self.active_session)
        with ExitStack() as stack:
            stack.enter_context(patch.dict("sys.modules", {"notebookutils": notebookutils,
                                                           "pyspark": pyspark, "pyspark.sql": sql}))
            stack.enter_context(patch("aews.runtime.sys.version_info", (3, 11, 8)))
            versions = stack.enter_context(patch("aews.runtime.version", side_effect=self.get_version))
            imports = stack.enter_context(patch("aews.runtime.importlib.import_module", side_effect=self.import_module))
            stack.enter_context(patch("aews.runtime.Path.is_dir", return_value=True))
            stack.enter_context(redirect_stdout(self.output))
            result = validate_session(self.spark, self.config, **kwargs)
        return result, versions, imports

    def test_table_loader_needs_neither_mlflow_module_nor_distribution(self):
        del self.versions["mlflow"]
        self.missing_module = "mlflow"
        (root, observed), versions, imports = self.run_preflight(require_model=False)
        self.assertEqual(root, Path("/lakehouse/default/Files"))
        self.assertNotIn("mlflow", observed)
        self.assertNotIn(("mlflow",), [call.args for call in versions.call_args_list])
        self.assertFalse(any(call.args[0].startswith("mlflow") for call in imports.call_args_list))

    def test_managed_delta_without_package_metadata_checks_both_classes(self):
        del self.versions["delta-spark"]
        (_, observed), _, imports = self.run_preflight(require_model=False)
        self.assertIsNone(observed["delta"])
        self.assertEqual(observed["delta_version_source"], "unavailable")
        self.assertTrue(observed["delta_capability_verified"])
        self.assertIn("exact Delta version metadata is unavailable", self.output.getvalue())
        self.assertIn(("delta.tables",), [call.args for call in imports.call_args_list])
        loader = self.spark._jvm.java.lang.Thread.currentThread.return_value.getContextClassLoader.return_value
        loader.loadClass.assert_called_once_with("io.delta.tables.DeltaTable")

    def test_table_only_session_handles_both_missing_distributions_together(self):
        del self.versions["delta-spark"]
        del self.versions["mlflow"]
        self.missing_module = "mlflow"
        (_, observed), _, imports = self.run_preflight(require_model=False)
        self.assertIsNone(observed["delta"])
        self.assertNotIn("mlflow", observed)
        self.assertFalse(any(call.args[0].startswith("mlflow") for call in imports.call_args_list))
        self.spark.sql.assert_any_call("SHOW SCHEMAS")
        self.spark.sql.assert_any_call("SHOW TABLES IN `academic_early_warning`.`dbo`")

    def test_missing_delta_python_support_is_not_ignored(self):
        self.missing_module = "delta.tables"
        with self.assertRaisesRegex(ModuleNotFoundError, "delta.tables"):
            self.run_preflight(require_model=False)

    def test_missing_delta_jvm_support_is_not_ignored(self):
        loader = self.spark._jvm.java.lang.Thread.currentThread.return_value.getContextClassLoader.return_value
        loader.loadClass.side_effect = RuntimeError("Delta class unavailable")
        with self.assertRaisesRegex(RuntimeError, "Delta class unavailable"):
            self.run_preflight(require_model=False)

    def test_present_incompatible_delta_version_is_rejected(self):
        self.versions["delta-spark"] = "4.0.0"
        with self.assertRaisesRegex(RuntimeError, "Expected Delta 3.2"):
            self.run_preflight(require_model=False)

    def test_model_workflow_still_requires_matching_mlflow_version(self):
        self.versions["mlflow"] = "3.0.0"
        with self.assertRaisesRegex(RuntimeError, "Expected mlflow 2.12.2"):
            self.run_preflight()

    def test_model_workflow_checks_managed_mlflow_module_version(self):
        del self.versions["mlflow"]
        (_, observed), _, imports = self.run_preflight()
        self.assertEqual(observed["mlflow"], "2.12.2")
        self.assertIn(("mlflow.sklearn",), [call.args for call in imports.call_args_list])
        self.module_version = "3.0.0"
        with self.assertRaisesRegex(RuntimeError, "Expected mlflow 2.12.2"):
            self.run_preflight()

    def test_missing_or_unversioned_mlflow_still_blocks_model_workflow(self):
        del self.versions["mlflow"]
        self.missing_module = "mlflow"
        with self.assertRaises(ModuleNotFoundError):
            self.run_preflight()
        self.missing_module = None
        self.module_version = None
        with self.assertRaisesRegex(RuntimeError, "version is unavailable"):
            self.run_preflight()

    def test_missing_default_lakehouse_has_actionable_error_before_package_checks(self):
        del self.context["defaultLakehouseName"]
        self.versions.clear()
        with self.assertRaisesRegex(RuntimeError, "Set the intended Lakehouse as default.*stop/restart"):
            self.run_preflight(require_model=False)
        self.spark.sql.assert_not_called()

    def test_wrong_target_and_missing_session_still_block_loading(self):
        self.context["currentWorkspaceName"] = "Other workspace"
        with self.assertRaisesRegex(RuntimeError, "expected 'Academic Early Warning System', observed 'Other workspace'"):
            self.run_preflight(require_model=False)
        self.active_session = None
        with self.assertRaisesRegex(RuntimeError, "No active Spark session"):
            self.run_preflight(require_model=False)

    def test_catalog_and_runtime_guards_remain_enabled(self):
        self.spark.sql.return_value.collect.return_value = [("dbo",)]
        with self.assertRaisesRegex(RuntimeError, "exact workspace.Lakehouse.dbo"):
            self.run_preflight(require_model=False)
        self.versions["pyarrow"] = "1.0.0"
        with self.assertRaisesRegex(RuntimeError, "Expected pyarrow"):
            self.run_preflight(require_model=False)

    def test_import_explicitly_selects_table_only_preflight(self):
        with patch("aews.table_import.validate_session", side_effect=RuntimeError("stop before data")) as preflight:
            with self.assertRaisesRegex(RuntimeError, "stop before data"):
                load_folder_tables(self.spark, self.config, {}, "Files/data", {})
        preflight.assert_called_once_with(self.spark, self.config, require_model=False)


if __name__ == "__main__":
    unittest.main()
