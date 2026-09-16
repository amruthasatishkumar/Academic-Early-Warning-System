"""Read-only inspection of one complete, immutable publication."""
import hashlib
import json

import numpy as np
import pandas as pd

from aews.runtime import CURRENT_MANIFEST, TABLES, validate_manifest, validate_session
from aews.presentation.explainability import OutcomeExplanations


def read_publication(spark, config):
    validate_session(spark, config)
    import notebookutils
    import mlflow.sklearn
    fs = notebookutils.fs
    if not fs.exists(CURRENT_MANIFEST):
        raise RuntimeError("No current publication. Complete the full build notebook first.")
    manifest = validate_manifest(json.loads(fs.head(CURRENT_MANIFEST, 16 * 1024 * 1024)))
    if manifest.get("workspace_name") != config["intended_workspace"] or manifest.get("lakehouse_name") != config["intended_lakehouse"]:
        raise RuntimeError("Manifest belongs to a different configured target.")
    frames = {}
    for name in TABLES:
        frame = spark.read.format("delta").load(manifest["table_paths"][name])
        count = frame.count()
        if count != manifest["table_counts"][name] or count > 100000:
            raise RuntimeError("Publication count mismatch or bounded driver read exceeded: " + name)
        frames[name] = frame.toPandas()
    current = frames["silver.current_weekly_features"].sort_values("enrollment_id").reset_index(drop=True)
    predictions = frames["gold.student_course_predictions"].sort_values("enrollment_id").reset_index(drop=True)
    if any(name in current for name in ("label", "label_status", "label_available_at")):
        raise RuntimeError("Current publication contains future outcome fields.")
    if predictions.prediction_id.duplicated().any() or current.enrollment_id.duplicated().any():
        raise RuntimeError("Duplicate current keys.")
    if len(current) != len(predictions) or not np.array_equal(current.enrollment_id, predictions.enrollment_id):
        raise RuntimeError("Current features and predictions are not aligned.")
    if not predictions.model_version.eq(manifest["model_version"]).all() or not predictions.mlflow_run_id.eq(manifest["mlflow_run_id"]).all():
        raise RuntimeError("Published prediction provenance mismatch.")
    historical = frames["silver.historical_weekly_features"].sort_values(["term", "course_id", "enrollment_id"])
    training = historical.loc[historical.term.isin([1, 2]) & historical.label.notna() & (historical.cutoff_week == 4)]
    # MLflow deserializes a model. Use only this trusted workspace's own publication.
    model = mlflow.sklearn.load_model("runs:/" + manifest["mlflow_run_id"] + "/model")
    explainer = OutcomeExplanations(model, training)
    fingerprint = hashlib.sha256(pd.util.hash_pandas_object(
        training[explainer.features + ["label", "enrollment_id"]], index=False).values.tobytes()).hexdigest()
    if fingerprint != manifest["train_data_sha256"]:
        raise RuntimeError("Training reference does not match the fitted model's provenance.")
    drivers = explainer.result_drivers(current, predictions)
    predictions["top_predicted_class_drivers"] = [json.dumps(value) for value in drivers]
    if json.loads(fs.head(CURRENT_MANIFEST, 16 * 1024 * 1024)) != manifest:
        raise RuntimeError("Publication changed during reading; rerun after the writer has finished.")
    return {"manifest": manifest, "frames": frames, "training": training, "current": current,
            "predictions": predictions, "explainer": explainer}
