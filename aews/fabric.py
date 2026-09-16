"""Explicit generated or uploaded-data publication. Importing performs no cloud I/O."""
import hashlib
import inspect
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from aews.configuration import validate_configuration
from aews.data_bundle import load_bundle
from aews.core.contract import dumps
from aews.core.execution import ParquetSourceSink, SUPPORT_SCHEMA, arrow_schema, prepare
from aews.core.features import feature_contract
from aews.execution import finish_run
from aews.model.modeling import predictors
from aews.runtime import CURRENT_MANIFEST, PURPOSE, SCHEMAS, TABLES, qualified, validate_manifest, validate_session
from aews.spark_tables import _pandas_to_spark, _spark_schema
from aews.storage import archive_delta, canonical_json, read_json, require_equivalent, write_once_json


def runtime_preflight(spark, source_config, model_config, contract, allow_replacement=False, uploaded=False):
    if allow_replacement is not True:
        raise ValueError("Acknowledge replacement of the named synthetic tables before publication.")
    validate_configuration(source_config, model_config, contract)
    if not uploaded and source_config["mode"] != "full":
        raise ValueError("Only full profile may publish stable cloud tables; validate small locally.")
    return validate_session(spark, source_config)


def validate_uploaded_data(spark, source_config, model_config, contract, bundle_path, bundle_info):
    validate_configuration(source_config, model_config, contract)
    root, versions = validate_session(spark, source_config)
    if bundle_path != bundle_info["expected_lakehouse_path"]:
        raise ValueError("Upload the included archive to the exact documented Files path.")
    if not bundle_path.startswith("Files/uploads/") or ".." in bundle_path or "\\" in bundle_path:
        raise ValueError("Invalid uploaded dataset location.")
    return load_bundle(root / bundle_path.removeprefix("Files/"), bundle_info, source_config, contract)


def prepare_input(source_config, contract, staging, bundle=None):
    if bundle is not None:
        bundle.extract_sources(staging)
        return bundle.features.copy(), dict(bundle.summary)
    return prepare(source_config, contract, ParquetSourceSink(staging, contract))


def source_targets(contract):
    targets = []
    for table, definition in contract["tables"].items():
        targets.append((definition["namespace"], table, arrow_schema(table, contract)))
        if table not in ("accounts", "users", "enrollment_terms"):
            targets.append(("sim_history", table + "_revisions", arrow_schema(table, contract, history=True)))
    return targets + [("sim_support", table, schema) for table, schema in SUPPORT_SCHEMA.items()]


def archive_existing(spark, fs, config, contract, archive_root):
    names = [(namespace, table) for namespace, table, _ in source_targets(contract)]
    names += [tuple(name.split(".")) for name in TABLES]
    archived = []
    if fs.exists(CURRENT_MANIFEST):
        previous = validate_manifest(read_json(fs, CURRENT_MANIFEST))
        write_once_json(fs, archive_root + "/previous_current.json", previous)
    for namespace, table in names:
        target = qualified(config, namespace, table)
        if spark.catalog.tableExists(target):
            path = archive_root + "/tables/" + namespace + "/" + table
            archive_delta(spark, fs, spark.table(target), path)
            archived.append(namespace + "." + table)
    write_once_json(fs, archive_root + "/preservation.json", {"archived_tables": archived})


def publish_sources(spark, config, contract, source_path, expected_counts):
    for namespace, table, schema in source_targets(contract):
        data = spark.read.schema(_spark_schema(schema)).parquet(source_path + "/" + namespace + "/" + table)
        data = spark.createDataFrame(data.rdd, schema=_spark_schema(schema))
        target = qualified(config, namespace, table)
        data.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(target)
        actual = spark.table(target)
        require_equivalent(data, actual, namespace + "." + table)
        expected = expected_counts.get(table)
        if expected is not None and actual.count() != expected:
            raise RuntimeError("Published source count mismatch: " + target)


def log_model(result, source, model_config, features, summary, versions):
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
    if "pyfunc_predict_fn" not in inspect.signature(mlflow.sklearn.log_model).parameters:
        raise RuntimeError("MLflow cannot log the required probability-output contract.")
    mlflow.autolog(disable=True)
    mlflow.set_experiment(source["experiment_name"])
    with mlflow.start_run(run_name=result.version) as run:
        mlflow.set_tags({
            "synthetic": "true", "workflow": model_config["workflow_version"],
            "model_status": "development_candidate_not_accepted",
            "term4_status": "previously_exposed_diagnostic_not_independent",
        })
        mlflow.log_params({
            "selected_model": result.model_name, "model_version": result.version,
            "source_seed": source["seed"], "source_profile": source["mode"],
            "train_terms": "1,2", "calibration_and_decision_term": 3,
            "train_data_sha256": result.train_fingerprint, "code_sha256": summary["code_sha256"],
        })
        for row in result.evaluations:
            prefix = f"{row['split']}.{row['model_name']}.{row['slice']}"
            for name, value in row["metrics"].items():
                if isinstance(value, (int, float)) and np.isfinite(value):
                    mlflow.log_metric(prefix + "." + name, float(value))
        for name, value in (
            ("model-selection.json", result.selection), ("split-manifest.json", result.split_manifest),
            ("evaluation.json", result.evaluations), ("quality.json", summary),
            ("feature-contract.json", feature_contract()), ("runtime.json", versions),
            ("configuration.json", {"source": source, "model": model_config}),
            ("registration.json", {"registered": False, "reason": result.registration_reason}),
        ):
            mlflow.log_dict(json.loads(dumps(value)), name)
        example = predictors(features.loc[(features.term == 1) & (features.cutoff_week == 4)]).head(5)
        mlflow.sklearn.log_model(
            result.model, artifact_path="model", input_example=example,
            signature=infer_signature(example, result.model.predict_proba(example)), pyfunc_predict_fn="predict_proba")
        return run.info.run_id


def run_fabric(spark, source_config, model_config, contract, code_fingerprint, allow_replacement=False,
               input_mode="generate", bundle_path=None, bundle_info=None):
    if input_mode not in ("generate", "uploaded"):
        raise ValueError("INPUT_MODE must be uploaded or generate.")
    source_config = dict(source_config)
    if input_mode == "uploaded":
        source_config["mode"] = "small"
    root, versions = runtime_preflight(spark, source_config, model_config, contract, allow_replacement,
                                      uploaded=input_mode == "uploaded")
    import notebookutils
    import mlflow
    mlflow.autolog(disable=True)
    fs = notebookutils.fs
    bundle = None
    if input_mode == "uploaded":
        if bundle_info is None or bundle_path != bundle_info["expected_lakehouse_path"]:
            raise ValueError("Use the included data ZIP and documented upload path.")
        bundle = load_bundle(root / bundle_path.removeprefix("Files/"), bundle_info, source_config, contract)
    publication_id = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S%f") + "-" + uuid.uuid4().hex[:12]
    source_id = hashlib.sha256(dumps({
        "config": source_config, "code": code_fingerprint, "contract": contract,
    }).encode()).hexdigest()[:20]
    staging_relative = Path("data") / source_id / publication_id / "source_parquet"
    archive_root = "Files/_archive/" + publication_id
    source_path = (Path("Files") / staging_relative).as_posix()
    staging = root / staging_relative
    if staging.exists():
        raise RuntimeError("Source publication location already exists; refusing overwrite.")
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    spark.conf.set("spark.sql.shuffle.partitions", "8")
    features, base_summary = prepare_input(source_config, contract, staging, bundle)
    stamp = pd.Timestamp.now(tz="UTC")
    result, summary, tables, importance = finish_run(
        features, base_summary, model_config, code_fingerprint, stamp)
    summary["source_dataset_version"] = source_id
    run_id = log_model(result, source_config, model_config, features, summary, versions)
    tables["gold.student_course_predictions"]["mlflow_run_id"] = run_id
    publication_root = f"Files/model_artifacts/{result.version}/publications/{publication_id}"
    write_once_json(fs, publication_root + "/model-selection.json", json.loads(dumps(result.selection)))
    write_once_json(fs, publication_root + "/runtime.json", versions)
    write_once_json(fs, publication_root + "/training-contributions.json", importance.to_dict("records"))
    for schema in SCHEMAS:
        spark.sql("CREATE SCHEMA IF NOT EXISTS " + qualified(source_config, schema))
    # Preserve every named existing table before the first replacement, not after it.
    archive_existing(spark, fs, source_config, contract, archive_root)
    snapshots, paths = {}, {}
    for table, data in tables.items():
        path = publication_root + "/tables/" + table.replace(".", "/")
        snapshots[table] = archive_delta(spark, fs, _pandas_to_spark(spark, data), path)
        paths[table] = path
    publish_sources(spark, source_config, contract, source_path, base_summary["observed_source_counts"])
    for name, frame in snapshots.items():
        schema, table = name.split(".")
        target = qualified(source_config, schema, table)
        frame.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(target)
        require_equivalent(frame, spark.table(target), name)
    manifest = {
        "purpose": PURPOSE, "publication_state": "complete", "source_profile": source_config["mode"],
        "input_mode": input_mode, "input_bundle_sha256": bundle_info["zip_sha256"] if bundle is not None else None,
        "model_version": result.version, "mlflow_run_id": run_id, "registered": False,
        "schemas": {"silver": "silver", "gold": "gold"}, "source_dataset_version": source_id,
        "source_data_path": source_path, "code_sha256": code_fingerprint,
        "train_data_sha256": result.train_fingerprint, "publication_archive": publication_root,
        "previous_archive": archive_root, "table_paths": paths,
        "current_predictions": len(tables["gold.student_course_predictions"]),
        "current_explanations": len(tables["gold.prediction_explanations"]),
        "current_features": len(tables["silver.current_weekly_features"]),
        "historical_features": len(tables["silver.historical_weekly_features"]),
        "table_counts": {name: len(data) for name, data in tables.items()},
        "published_at": stamp.isoformat(), "workspace_name": source_config["intended_workspace"],
        "lakehouse_name": source_config["intended_lakehouse"],
        "notice": "Synthetic demonstration. No independent institutional evidence or approved review policy.",
    }
    validate_manifest(manifest)
    write_once_json(fs, publication_root + "/publication.json", manifest)
    if not fs.put(CURRENT_MANIFEST, canonical_json(manifest), overwrite=True):
        raise RuntimeError("Could not update current publication pointer.")
    if read_json(fs, CURRENT_MANIFEST) != manifest:
        raise RuntimeError("Current publication pointer read-back mismatch.")
    print(dumps({"status": "publication_completed", "manifest": manifest, "observed": summary}))
    return {"summary": summary, "manifest": manifest}
