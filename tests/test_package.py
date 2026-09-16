import ast
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

from aews.configuration import load_configuration
from aews.fabric import archive_existing, run_fabric, runtime_preflight, source_targets
from aews.runtime import TABLES, qualified, validate_manifest
from aews.storage import write_once_json
from tools.build_notebooks import build_notebooks
from tools.scan_package import scan, scan_text

ROOT = Path(__file__).resolve().parents[1]


class MemoryFS:
    def __init__(self):
        self.files = {}

    def exists(self, path):
        return path in self.files

    def head(self, path, size):
        return self.files[path][:size]

    def mkdirs(self, path):
        return True

    def put(self, path, value, overwrite=False):
        if path in self.files and not overwrite:
            return False
        self.files[path] = value
        return True


class PackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source, cls.model, cls.contract = load_configuration(ROOT / "config")
        cls.work = ROOT / "outputs" / "test-package"
        if cls.work.exists():
            raise RuntimeError("Test directory already exists; remove it after verifying ownership.")
        cls.work.mkdir(parents=True)
        cls.notebooks = build_notebooks(output_dir=cls.work / "notebooks")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work)

    def test_guard_precedes_cloud_import_or_session_use(self):
        with self.assertRaisesRegex(ValueError, "Acknowledge"):
            run_fabric(None, self.source, self.model, self.contract, "test")
        with self.assertRaisesRegex(ValueError, "Only full"):
            runtime_preflight(None, self.source, self.model, self.contract, True)

    def test_stable_qualified_names(self):
        self.assertEqual(qualified(self.source, "gold", "student_course_predictions"),
                         "`academic_early_warning`.`gold`.`student_course_predictions`")
        namespaces = set(namespace for namespace, _, _ in source_targets(self.contract))
        self.assertEqual(namespaces, {"canvas", "canvas_logs", "sim_history", "sim_support"})

    def test_automatic_logging_is_disabled_before_generation_or_training(self):
        import types
        from unittest.mock import Mock
        source = copy.deepcopy(self.source)
        source["mode"] = "full"
        events = []
        automatic_logging = Mock(side_effect=lambda **kwargs: events.append(("autolog", kwargs)))
        def stop_at_generation(*args, **kwargs):
            events.append(("prepare", {}))
            raise RuntimeError("stop before data generation")
        with patch.dict(sys.modules, {
            "notebookutils": types.SimpleNamespace(fs=MemoryFS()),
            "mlflow": types.SimpleNamespace(autolog=automatic_logging),
        }), patch("aews.fabric.runtime_preflight", return_value=(self.work / "logging-order", {})), \
             patch("aews.fabric.prepare", side_effect=stop_at_generation):
            with self.assertRaisesRegex(RuntimeError, "stop before data generation"):
                run_fabric(unittest.mock.MagicMock(), source, self.model, self.contract, "test", True)
        self.assertEqual(events, [("autolog", {"disable": True}), ("prepare", {})])

    def test_notebook_syntax_clean_outputs_and_isolated_imports(self):
        for path in self.notebooks:
            notebook = json.loads(path.read_text())
            for cell in notebook["cells"]:
                ast.parse("".join(cell["source"]))
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
            bundle = "".join(notebook["cells"][1]["source"])
            check = bundle + "\nassert 'aews.presentation.explainability' in sys.modules\n"
            if path.name.startswith("01"):
                check += "from aews.fabric import run_fabric\nassert callable(run_fabric)\n"
            completed = subprocess.run([sys.executable, "-I", "-c", "import sys; exec(sys.stdin.read())"],
                                       input=check, text=True, capture_output=True)
            if completed.returncode:
                self.fail(completed.stderr)

    def test_exploration_contains_no_mutations(self):
        text = self.notebooks[1].read_text()
        for forbidden in (".fit(", ".saveAsTable(", ".write.", ".register_model(", ".put(", ".mkdirs(", ".conf.set("):
            self.assertNotIn(forbidden, text)
        self.assertIn("training_importance", text)
        self.assertIn("result_drivers", text)
        notebook = json.loads(self.notebooks[0].read_text())
        settings = "".join(notebook["cells"][2]["source"])
        self.assertIn("ALLOW_DEMO_TABLE_REPLACEMENT = False", settings)
        self.assertIn("RUN_MODE = 'validate'", settings)
        self.assertIn("INPUT_MODE = 'uploaded'", settings)
        self.assertIn("BUNDLE_INFO", settings)
        info = json.loads((ROOT / "sample-data" / "bundle-info.json").read_text())
        self.assertIn(info["zip_sha256"], settings)

    def test_quickstart_and_notebook_warn_against_manual_table_loading(self):
        for path in (ROOT / "README.rst", ROOT / "docs" / "replication-guide.rst"):
            content = path.read_text()
            self.assertIn("Do not use Load to Tables for the replication Quickstart.", content)
            self.assertIn("Upload files", content)
            self.assertIn("Files/uploads", content)
        dataset_readme = (ROOT / "sample-data" / "README.txt").read_text()
        self.assertIn("REPLICATION QUICKSTART: DO NOT USE LOAD TO TABLES.", dataset_readme)
        notebook = json.loads(self.notebooks[0].read_text())
        introduction = "".join(notebook["cells"][0]["source"])
        self.assertIn("Do not use Load to Tables", introduction)
        self.assertIn("keep the inner simulated-data ZIP compressed", introduction)
        self.assertIn("This notebook creates tables", introduction)

    def test_config_names_propagate_to_both_notebooks(self):
        directory = self.work / "configuration"
        directory.mkdir()
        source = copy.deepcopy(self.source)
        source.update(intended_workspace="Example Academic Workspace", intended_lakehouse="example_academic",
                      experiment_name="Example Academic Training")
        for name, value in (("simulation.json", source), ("modeling.json", self.model),
                            ("canvas-source-contract.json", self.contract)):
            (directory / name).write_text(json.dumps(value), encoding="ascii")
        paths = build_notebooks(config_dir=directory, output_dir=self.work / "configured")
        for path in paths:
            notebook = json.loads(path.read_text())
            values = {}
            exec("".join(notebook["cells"][2]["source"]), values)
            self.assertEqual(values["SOURCE_CONFIG"]["intended_workspace"], source["intended_workspace"])
            self.assertEqual(values["SOURCE_CONFIG"]["intended_lakehouse"], source["intended_lakehouse"])
            self.assertEqual(values["SOURCE_CONFIG"]["experiment_name"], source["experiment_name"])

    def test_explanation_controls_prefill_from_the_same_prediction(self):
        import pandas as pd
        records = pd.DataFrame({
            "enrollment_id": [200, 100], "course_id": [2, 1], "predicted_class": ["fail", "pass"],
            "p_excellent": [0.1, 0.1], "p_pass": [0.1, 0.7], "p_fail": [0.7, 0.1],
            "p_withdrawn": [0.1, 0.1], "top_predicted_class_drivers": ["[]", "[]"],
        })
        notebook = json.loads(self.notebooks[1].read_text())
        code = "".join(notebook["cells"][5]["source"])
        values = {"VIEW": {"predictions": records}, "display": lambda data: None}
        output = io.StringIO()
        with redirect_stdout(output):
            exec(code, values)
        self.assertEqual(values["ENROLLMENT_TO_EXPLAIN"], 100)
        self.assertEqual(values["OUTCOME_TO_EXPLAIN"], "pass")
        self.assertIn("ENROLLMENT_TO_EXPLAIN = 100", output.getvalue())
        self.assertIn("OUTCOME_TO_EXPLAIN = 'pass'", output.getvalue())
        alternate = code.replace(
            "ENROLLMENT_TO_EXPLAIN = int(available_predictions.iloc[0].enrollment_id)",
            "ENROLLMENT_TO_EXPLAIN = 200",
        )
        with redirect_stdout(io.StringIO()):
            exec(alternate, values)
        self.assertEqual(values["OUTCOME_TO_EXPLAIN"], "fail")

    def test_empty_prediction_controls_fail_explicitly(self):
        import pandas as pd
        notebook = json.loads(self.notebooks[1].read_text())
        code = "".join(notebook["cells"][5]["source"])
        values = {"VIEW": {"predictions": pd.DataFrame({"enrollment_id": []})}, "display": lambda data: None}
        with self.assertRaisesRegex(ValueError, "No current predictions"):
            exec(code, values)

    def test_invalid_outcome_cannot_silently_fall_back(self):
        import pandas as pd
        notebook = json.loads(self.notebooks[1].read_text())
        code = "".join(notebook["cells"][6]["source"])
        rows = pd.DataFrame({"enrollment_id": [100], "predicted_class": ["pass"]})
        values = {"VIEW": {"current": rows, "predictions": rows}, "ENROLLMENT_TO_EXPLAIN": 100,
                  "OUTCOME_TO_EXPLAIN": "", "OUTCOMES": ("excellent", "pass", "fail", "withdrawn")}
        with self.assertRaisesRegex(ValueError, "OUTCOME_TO_EXPLAIN must be"):
            exec(code, values)

    def test_write_once_preserves_and_rejects_conflicts(self):
        fs = MemoryFS()
        write_once_json(fs, "Files/model_artifacts/example/selection.json", {"selected": "logistic"})
        original = dict(fs.files)
        write_once_json(fs, "Files/model_artifacts/example/selection.json", {"selected": "logistic"})
        self.assertEqual(fs.files, original)
        with self.assertRaisesRegex(RuntimeError, "conflicting"):
            write_once_json(fs, "Files/model_artifacts/example/selection.json", {"selected": "different"})
        self.assertEqual(fs.files, original)
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            validate_manifest({})

    def test_every_existing_named_table_is_archived(self):
        class Catalog:
            def tableExists(self, name):
                return True

        class Spark:
            catalog = Catalog()

            def table(self, name):
                return name
        fs = MemoryFS()
        with patch("aews.fabric.archive_delta") as archived:
            archive_existing(Spark(), fs, self.source, self.contract, "Files/_archive/test")
        self.assertEqual(archived.call_count, len(source_targets(self.contract)) + len(TABLES))
        self.assertTrue(all(call.args[-1].startswith("Files/_archive/test/tables/") for call in archived.call_args_list))
        self.assertIn("Files/_archive/test/preservation.json", fs.files)

    def test_failed_archive_prevents_first_table_replacement(self):
        source = copy.deepcopy(self.source)
        source["mode"] = "full"
        import types
        import pandas as pd
        result = types.SimpleNamespace(version="aews-test", selection={"selected": "logistic"})
        notebookutils = types.SimpleNamespace(fs=MemoryFS())
        tables = {"gold.student_course_predictions": pd.DataFrame({"prediction_id": ["test"]})}
        spark = unittest.mock.MagicMock()
        with patch.dict(sys.modules, {"notebookutils": notebookutils}), \
             patch("aews.fabric.runtime_preflight", return_value=(self.work / "cloud-test", {})), \
             patch("aews.fabric.prepare", return_value=(pd.DataFrame(), {})), \
             patch("aews.fabric.finish_run", return_value=(result, {}, tables, pd.DataFrame())), \
             patch("aews.fabric.log_model", return_value="local-test-run"), \
             patch("aews.fabric.archive_existing", side_effect=RuntimeError("archive verification failed")), \
             patch("aews.fabric.publish_sources") as publish:
            with self.assertRaisesRegex(RuntimeError, "archive verification"):
                run_fabric(spark, source, self.model, self.contract, "test", True)
        publish.assert_not_called()
        self.assertFalse(any("current.json" in name for name in notebookutils.fs.files))

    def test_private_identifier_scanner(self):
        address = "person" + "@" + "example.org"
        identifier = "-".join(["12345678", "1234", "1234", "1234", "123456789012"])
        self.assertIn("email address", scan_text(address))
        self.assertIn("literal resource identifier", scan_text(identifier))
        _, findings = scan(ROOT)
        self.assertEqual(findings, [])
