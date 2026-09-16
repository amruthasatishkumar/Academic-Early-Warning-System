from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, log_loss
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aews.core.contract import dumps
from aews.core.features import CLASSES, FEATURES
from aews.presentation.explainability import OutcomeExplanations

WORKFLOW_VERSION = "four-outcome-calibrated-1"


@dataclass
class ModelResult:
    model: object
    model_name: str
    version: str
    evaluations: list
    train_fingerprint: str
    split_manifest: dict
    selection: dict
    explainer: object
    registration_eligible: bool = False
    registration_reason: str = "Synthetic development candidate; no independent institutional evidence or acceptance."


def predictors(data):
    result = data.loc[:, FEATURES].astype(float)
    if np.isinf(result.to_numpy()).any():
        raise ValueError("Infinite predictor.")
    return result


def probabilities(model, data):
    classes = list(model.classes_)
    if sorted(classes) != CLASSES:
        raise ValueError("Model must contain exactly four outcomes.")
    values = model.predict_proba(predictors(data))[:, [classes.index(c) for c in CLASSES]]
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Invalid predicted probabilities.")
    np.testing.assert_allclose(values.sum(axis=1), 1., atol=1e-10)
    return values


def fingerprint(data):
    columns = FEATURES + ["label", "enrollment_id"]
    return hashlib.sha256(pd.util.hash_pandas_object(data[columns], index=False).values.tobytes()).hexdigest()


def pipeline(estimator):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("classifier", estimator),
    ])


def metrics(labels, values):
    if sorted(pd.Series(labels).unique()) != CLASSES:
        raise ValueError("Evaluation requires all four classes; do not manufacture missing outcomes.")
    labels = np.asarray(labels)
    predicted = np.asarray(CLASSES)[values.argmax(axis=1)]
    per_class = {}
    for index, outcome in enumerate(CLASSES):
        target, probability = labels == outcome, values[:, index]
        groups = np.minimum((probability * 10).astype(int), 9)
        bins = [{"lower": i / 10, "count": int((groups == i).sum()),
                 "mean_probability": float(probability[groups == i].mean()),
                 "observed_rate": float(target[groups == i].mean())}
                for i in range(10) if (groups == i).any()]
        per_class[outcome] = {
            "prevalence": float(target.mean()),
            "average_precision": float(average_precision_score(target, probability)),
            "brier": float(np.mean((probability - target) ** 2)),
            "calibration_bins": bins,
        }
    report = classification_report(labels, predicted, labels=CLASSES, output_dict=True, zero_division=0)
    return {
        "rows": len(labels), "macro_f1": float(report["macro avg"]["f1-score"]),
        "macro_ovr_average_precision": float(np.mean([p["average_precision"] for p in per_class.values()])),
        "multiclass_log_loss": float(log_loss(labels, values, labels=CLASSES)),
        "per_class": per_class, "classification_report": report,
        "confusion_matrix": confusion_matrix(labels, predicted, labels=CLASSES).tolist(),
        "confusion_class_order": CLASSES,
        "scope": "Synthetic development evaluation only; not institutional evidence.",
    }


def split_validation(validation, config):
    if set(validation.term) != {3} or set(validation.cutoff_week) != {4}:
        raise ValueError("Calibration/decision data must be term three, week four.")
    if validation.duplicated("enrollment_id").any():
        raise ValueError("Duplicate validation enrollment.")
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=config["calibration_fraction_of_term3"],
                                     random_state=config["seed"])
    left, right = next(splitter.split(validation, validation.label))
    calibration, decision = validation.iloc[left].copy(), validation.iloc[right].copy()
    if set(calibration.enrollment_id) & set(decision.enrollment_id):
        raise ValueError("Calibration/decision enrollment overlap.")
    for data in (calibration, decision):
        if sorted(data.label.unique()) != CLASSES:
            raise ValueError("Missing class in calibration/decision; do not regenerate or change seed.")
    return calibration, decision


def select_model(training, validation, config, code_fingerprint):
    if config.get("registration_approved") is not False:
        raise ValueError("Accepted-model registration is disabled.")
    if config["workflow_version"] != WORKFLOW_VERSION:
        raise ValueError("Unexpected workflow.")
    if set(training.term) != {1, 2} or set(training.cutoff_week) != {4}:
        raise ValueError("Training must use terms one and two, week four only.")
    if training.label.isna().any() or sorted(training.label.unique()) != CLASSES:
        raise ValueError("Training requires known examples of all four classes.")
    if training.duplicated("enrollment_id").any() or set(training.enrollment_id) & set(validation.enrollment_id):
        raise ValueError("Duplicate or overlapping training enrollment.")
    if len(training) + len(validation) > config["max_training_rows"]:
        raise ValueError("Modeling driver row bound exceeded.")
    calibration, decision = split_validation(validation, config)
    candidates = {"prevalence_baseline": pipeline(DummyClassifier(strategy="prior"))}
    for weight in config["class_weight_candidates"]:
        for c in config["logistic_C"]:
            candidates[f"logistic_{weight or 'unweighted'}_C{c:g}"] = pipeline(
                LogisticRegression(C=c, class_weight=weight, max_iter=500, solver="lbfgs",
                                   random_state=config["seed"]))
    fitted, measured, evaluations = {}, {}, []
    for name, candidate in candidates.items():
        candidate.fit(predictors(training), training.label)
        if name == "prevalence_baseline":
            model = candidate
        else:
            # Runtime 1.3 uses sklearn 1.2's prefit calibration API.
            model = CalibratedClassifierCV(candidate, method="sigmoid", cv="prefit")
            model.fit(predictors(calibration), calibration.label)
        fitted[name] = model
        measured[name] = metrics(decision.label, probabilities(model, decision))
        evaluations.append({"split": "term3_decision", "model_name": name, "slice": "all",
                            "metrics": measured[name]})
    eligible = [name for name in fitted if name.startswith("logistic_")]
    selected = max(eligible, key=lambda n: (measured[n]["macro_ovr_average_precision"],
                                          -measured[n]["multiclass_log_loss"]))
    values = probabilities(fitted[selected], decision)
    for name, mask in (
        ("returning_students", decision.user_id.isin(training.user_id)),
        ("unseen_students", ~decision.user_id.isin(training.user_id)),
    ):
        subset = decision.loc[mask]
        scores = (metrics(subset.label, values[mask.to_numpy()])
                  if sorted(subset.label.unique()) == CLASSES
                  else {"rows": len(subset), "not_evaluable": "At least one class is absent."})
        evaluations.append({"split": "term3_decision", "model_name": selected, "slice": name, "metrics": scores})
    split = {
        "train_terms": [1, 2], "validation_term": 3,
        "train_rows": len(training), "calibration_rows": len(calibration), "decision_rows": len(decision),
        "calibration_enrollment_ids": sorted(map(int, calibration.enrollment_id)),
        "decision_enrollment_ids": sorted(map(int, decision.enrollment_id)),
        "calibration_fingerprint": fingerprint(calibration), "decision_fingerprint": fingerprint(decision),
        "calibration_label_counts": calibration.label.value_counts().to_dict(),
        "decision_label_counts": decision.label.value_counts().to_dict(),
    }
    selection = {
        "selected_model": selected, "criterion": config["selection_criterion"],
        "eligible_models": eligible, "validation_metrics": measured[selected],
        "term4_used_for_selection": False, "fresh_independent_holdout_available": False,
        "term4_status": "previously_exposed_diagnostic_not_independent",
    }
    train_hash = fingerprint(training)
    version = "aews-" + hashlib.sha256(dumps({
        "code": code_fingerprint, "train": train_hash, "split": split, "config": config,
        "selected": selected, "workflow": WORKFLOW_VERSION,
        "libraries": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }).encode()).hexdigest()[:20]
    return ModelResult(fitted[selected], selected, version, evaluations, train_hash, split, selection,
                       OutcomeExplanations(fitted[selected], training))


def score_current(result, current, timestamp, run_id=None):
    if set(current.term) != {5} or set(current.cutoff_week) != {4}:
        raise ValueError("Scoring requires current term five, week four only.")
    if any(column in current for column in ("label", "label_status", "label_available_at")):
        raise ValueError("Current scoring must not contain outcome columns.")
    if current.duplicated(["enrollment_id", "as_of_time"]).any():
        raise ValueError("Duplicate current feature key.")
    current = current.sort_values("enrollment_id").reset_index(drop=True)
    values = probabilities(result.model, current)
    metadata = ["enrollment_id", "user_id", "course_id", "term", "cutoff_week", "as_of_time",
                "observation_start", "observation_end", "feature_version", "data_quality_flag"]
    predictions = current.loc[:, metadata].copy()
    predictions["prediction_id"] = [
        hashlib.sha256(f"{result.version}:{row.enrollment_id}:{row.as_of_time.isoformat()}".encode()).hexdigest()
        for row in current.itertuples()]
    for index, outcome in enumerate(CLASSES):
        predictions["p_" + outcome] = values[:, index]
    predictions["predicted_class"] = np.asarray(CLASSES)[values.argmax(axis=1)]
    predictions["model_version"] = result.version
    predictions["mlflow_run_id"] = run_id
    predictions["scored_at"] = timestamp
    predictions["model_status"] = "synthetic_candidate_not_accepted"
    predictions["synthetic"] = True
    terms = {outcome: result.explainer.contributions(current, outcome) for outcome in CLASSES}
    payloads = []
    for index, row in predictions.iterrows():
        outcome = row.predicted_class
        values_ = terms[outcome].iloc[index]
        baseline = float(result.explainer.baseline_scores[result.explainer.classes.index(outcome)])
        payloads.append(dumps({
            "outcome": outcome, "reference": "Mean transformed training population",
            "units": "Underlying multinomial linear class-score units before sigmoid calibration",
            "baseline_score": baseline, "contributions": values_.to_dict(),
            "full_score": float(baseline + values_.sum()),
            "notice": "Not probability-point changes, percentage contributions, SHAP or causal effects.",
        }))
    explanations = predictions[["prediction_id", "model_version", "enrollment_id", "as_of_time"]].copy()
    explanations["predicted_class"] = predictions.predicted_class
    explanations["evidence_kind"] = "training_centered_linear_score_contributions"
    explanations["explanation_json"] = payloads
    explanations["synthetic"] = True
    return predictions, explanations
