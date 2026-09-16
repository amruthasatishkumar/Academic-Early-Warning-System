import hashlib
import json
from pathlib import Path

import pandas as pd

from aews.configuration import validate_configuration
from aews.core.contract import dumps
from aews.core.execution import ParquetSourceSink, evaluation_frame, prepare, quality_frame
from aews.core.features import feature_contract
from aews.model.modeling import fingerprint, metrics, probabilities, score_current, select_model


def code_hash(root):
    digest = hashlib.sha256()
    root = Path(root)
    for path in sorted((root / "aews").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def split_features(features):
    historical = features.loc[features.term < 5].copy()
    current = features.loc[features.term == 5].drop(columns=["label", "label_status", "label_available_at"])
    return historical, current.sort_values("enrollment_id").reset_index(drop=True)


def development_fit(features, model_config, code_fingerprint):
    known = features.loc[(features.cutoff_week == 4) & features.label.notna()]
    return select_model(known.loc[known.term.isin([1, 2])].copy(), known.loc[known.term == 3].copy(),
                        model_config, code_fingerprint)


def evaluate_diagnostic(result, historical):
    diagnostic = historical.loc[(historical.term == 4) & (historical.cutoff_week == 4) & historical.label.notna()]
    result.evaluations.append({
        "split": "previously_exposed_term4_diagnostic", "slice": "all", "model_name": result.model_name,
        "metrics": metrics(diagnostic.label, probabilities(result.model, diagnostic)),
    })


def finish_run(features, source_summary, model_config, fingerprint_, timestamp, run_id=None):
    historical, current = split_features(features)
    result = development_fit(features, model_config, fingerprint_)
    if model_config["reevaluate_term4"]:
        evaluate_diagnostic(result, historical)
    predictions, explanations = score_current(result, current, timestamp, run_id)
    repeated, repeated_explanations = score_current(result, current, timestamp, run_id)
    pd.testing.assert_frame_equal(predictions, repeated)
    pd.testing.assert_frame_equal(explanations, repeated_explanations)
    training = historical.loc[historical.term.isin([1, 2]) & historical.label.notna() & (historical.cutoff_week == 4)]
    if fingerprint(training) != result.train_fingerprint:
        raise RuntimeError("Training provenance mismatch.")
    result.explainer.result_drivers(current, predictions)
    summary = {
        **source_summary, "model_version": result.version, "workflow_version": model_config["workflow_version"],
        "selection": result.selection, "split": result.split_manifest, "registered": False,
        "code_sha256": fingerprint_, "train_data_sha256": result.train_fingerprint,
        "current_predictions": len(predictions), "current_explanations": len(explanations),
        "historical_features": len(historical), "current_features": len(current),
        "current_outcome_counts": predictions.predicted_class.value_counts().to_dict(),
        "term4_status": "Previously exposed diagnostic, not independent validation.",
    }
    summary["assertions"] += ["same_model_cutoff_scoring_idempotent", "exact_local_score_reconstruction",
                              "saved_model_reproduces_predictions", "training_reference_fingerprint_verified"]
    quality = quality_frame(summary)
    quality["model_version"] = result.version
    tables = {
        "silver.historical_weekly_features": historical,
        "silver.current_weekly_features": current,
        "gold.student_course_predictions": predictions,
        "gold.prediction_explanations": explanations,
        "gold.model_evaluation": evaluation_frame(result),
        "gold.data_quality_summary": quality,
    }
    return result, summary, tables, result.explainer.training_importance(training)


def run_local(root, source_config, model_config, contract, output, write_sources=True):
    validate_configuration(source_config, model_config, contract)
    root, output = Path(root).resolve(), Path(output).resolve()
    allowed = root / "outputs"
    if allowed not in output.parents:
        raise ValueError("Local runs must use a new child directory under outputs.")
    if output.exists():
        raise FileExistsError("Local run output already exists; choose a new run directory.")
    output.mkdir(parents=True)
    fingerprint_ = code_hash(root)
    sink = ParquetSourceSink(output / "data" / "source_parquet", contract) if write_sources else None
    features, source_summary = prepare(source_config, contract, sink)
    result, summary, tables, importance = finish_run(
        features, source_summary, model_config, fingerprint_, pd.Timestamp.now(tz="UTC"))
    for name, data in tables.items():
        schema, table = name.split(".")
        target = output / schema
        target.mkdir(exist_ok=True)
        data.to_parquet(target / (table + ".parquet"), index=False)
        pd.testing.assert_frame_equal(pd.read_parquet(target / (table + ".parquet")), data)
    artifacts = output / "model_artifacts" / result.version
    artifacts.mkdir(parents=True)
    importance.to_csv(artifacts / "training_contributions.csv", index=False)
    for name, value in (
        ("observed-summary.json", summary), ("feature-contract.json", feature_contract()),
        ("model-selection.json", result.selection), ("split-manifest.json", result.split_manifest),
        ("configuration.json", {"source": source_config, "model": model_config}),
    ):
        (artifacts / name).write_text(dumps(value) + "\n", encoding="ascii")
    (output / "validation-summary.json").write_text(dumps(summary) + "\n", encoding="ascii")
    print(dumps({"status": "local_validation_passed", "profile": source_config["mode"],
                 "observed_source_counts": summary["observed_source_counts"],
                 "current_predictions": len(tables["gold.student_course_predictions"]),
                 "model_version": result.version}))
    return result, summary
