"""Read-only Fabric runtime and publication checks. No cloud session is created."""
import importlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

PURPOSE = "academic_early_warning_current_publication"
CURRENT_MANIFEST = "Files/model_artifacts/current.json"
SCHEMAS = ("canvas", "canvas_logs", "sim_history", "sim_support", "silver", "gold")
TABLES = ("silver.historical_weekly_features", "silver.current_weekly_features",
          "gold.student_course_predictions", "gold.prediction_explanations",
          "gold.model_evaluation", "gold.data_quality_summary")


def qualified(config, schema, table=None):
    if schema not in SCHEMAS and schema != "dbo":
        raise ValueError("Unexpected schema.")
    pieces = [config["intended_lakehouse"], schema] + ([table] if table else [])
    if any(not p or any(c in p for c in "`\r\n/\\") for p in pieces):
        raise ValueError("Unsafe catalog name.")
    return ".".join("`" + piece + "`" for piece in pieces)


def _mlflow_version():
    try:
        return version("mlflow")
    except PackageNotFoundError:
        module = importlib.import_module("mlflow")
        value = getattr(module, "__version__", None)
        if not isinstance(value, str) or not value:
            raise RuntimeError("MLflow is required for modeling, but its version is unavailable.")
        return value


def _delta_version(spark):
    importlib.import_module("delta.tables")
    spark._jvm.java.lang.Thread.currentThread().getContextClassLoader().loadClass("io.delta.tables.DeltaTable")
    try:
        value = version("delta-spark")
    except PackageNotFoundError:
        print("Fabric-provided Delta Python and Spark classes are available; exact Delta version metadata is unavailable.")
        return None
    if not value.startswith("3.2."):
        raise RuntimeError(f"Expected Delta 3.2, observed {value}.")
    return value


def validate_session(spark, config, *, require_model=True):
    if spark is None:
        raise RuntimeError("Attach a Lakehouse and use the existing Fabric Spark session.")
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("This starter targets Python 3.11 in Fabric Runtime 1.3.")
    if not spark.version.startswith("3.5."):
        raise RuntimeError("This starter targets Spark 3.5 in Fabric Runtime 1.3.")
    import notebookutils
    from pyspark.sql import SparkSession
    if SparkSession.getActiveSession() is None:
        raise RuntimeError("No active Spark session; this starter never creates one.")
    context = dict(notebookutils.runtime.context)
    if not context.get("defaultLakehouseName") or not context.get("defaultLakehouseWorkspaceName"):
        raise RuntimeError(
            "The running session has no default Lakehouse. Set the intended Lakehouse as default, "
            "stop/restart the notebook session, then rerun all cells.")
    for key, field in (("currentWorkspaceName", "intended_workspace"),
                       ("defaultLakehouseName", "intended_lakehouse"),
                       ("defaultLakehouseWorkspaceName", "intended_workspace")):
        if context.get(key) != config[field]:
            raise RuntimeError(
                f"{key} is not the configured target: expected {config[field]!r}, observed {context.get(key)!r}. "
                "Open the notebook in the intended workspace, right-click its Lakehouse -> Set as default lakehouse, "
                "then stop/restart the notebook session and rerun all cells.")
    expected = {"numpy": "1.26.4", "pandas": "2.1.4", "scikit-learn": "1.2.2", "pyarrow": "14.0.2"}
    observed = {name: version(name) for name in expected}
    if require_model:
        expected["mlflow"] = "2.12.2"
        observed["mlflow"] = _mlflow_version()
    for name, required in expected.items():
        if observed[name] != required:
            raise RuntimeError(f"Expected {name} {required}, observed {observed[name]}; validate compatibility before changing the guard.")
    if require_model:
        importlib.import_module("mlflow.sklearn")
    delta_version = _delta_version(spark)
    expected_schema = f"{config['intended_workspace']}.{config['intended_lakehouse']}.dbo"
    namespaces = spark.sql("SHOW SCHEMAS").collect()
    if not any(str(value).casefold() == expected_schema.casefold() for row in namespaces for value in row):
        raise RuntimeError("The exact workspace.Lakehouse.dbo namespace is not present.")
    spark.sql("SHOW TABLES IN " + qualified(config, "dbo")).collect()
    root = Path("/") / "lakehouse" / "default" / "Files"
    if not root.is_dir():
        raise RuntimeError("Default Lakehouse Files mount unavailable.")
    return root, {**observed, "python": ".".join(map(str, sys.version_info[:3])),
                  "spark": spark.version, "delta": delta_version,
                  "delta_capability_verified": True,
                  "delta_version_source": "package_metadata" if delta_version is not None else "unavailable"}


def validate_manifest(manifest):
    if manifest.get("purpose") != PURPOSE or manifest.get("publication_state") != "complete":
        raise RuntimeError("Publication is absent, incomplete or unrelated.")
    if manifest.get("source_profile") not in ("small", "full") or manifest.get("registered") is not False:
        raise RuntimeError("Exploration requires a recognized synthetic unregistered publication.")
    required = ("model_version", "mlflow_run_id", "code_sha256", "source_dataset_version",
                "train_data_sha256", "publication_archive")
    if any(not isinstance(manifest.get(key), str) or not manifest[key] for key in required):
        raise RuntimeError("Publication lacks immutable provenance.")
    if manifest.get("schemas") != {"silver": "silver", "gold": "gold"}:
        raise RuntimeError("Publication schemas do not match.")
    if set(manifest.get("table_paths", {})) != set(TABLES):
        raise RuntimeError("Publication table snapshots are incomplete.")
    root = manifest["publication_archive"]
    if not root.startswith("Files/model_artifacts/") or ".." in root or "\\" in root:
        raise RuntimeError("Unsafe publication archive path.")
    for table, path in manifest["table_paths"].items():
        if path != root + "/tables/" + table.replace(".", "/"):
            raise RuntimeError("Table path is outside its immutable publication archive.")
    if int(manifest.get("current_predictions", 0)) <= 0:
        raise RuntimeError("Publication has no predictions.")
    return manifest
