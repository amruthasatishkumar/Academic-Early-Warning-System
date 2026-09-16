import json
import re
from pathlib import Path

from aews.core.simulation import validate_config


def load_configuration(directory):
    directory = Path(directory)
    values = [json.loads((directory / name).read_text(encoding="utf-8"))
              for name in ("simulation.json", "modeling.json", "canvas-source-contract.json")]
    validate_configuration(*values)
    return tuple(values)


def validate_configuration(source, model, contract):
    validate_config(source)
    if source["mode"] not in ("small", "full") or source["current_week"] != 4:
        raise ValueError("This starter supports small/full profiles and current week four only.")
    for field in ("intended_workspace", "intended_lakehouse", "experiment_name"):
        value = source.get(field)
        if not isinstance(value, str) or not value.strip() or any(c in value for c in "`\r\n/\\"):
            raise ValueError(f"Missing or unsafe configuration name: {field}")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", source["intended_lakehouse"]):
        raise ValueError("Use an alphanumeric/underscore Lakehouse name.")
    if model.get("registration_approved") is not False:
        raise ValueError("Accepted-model registration is not implemented.")
    if model["workflow_version"] != "four-outcome-calibrated-1":
        raise ValueError("Unsupported model workflow.")
    if model["calibration_method"] != "sigmoid" or not 0.2 <= model["calibration_fraction_of_term3"] <= 0.8:
        raise ValueError("Unsupported calibration protocol.")
    if not model["logistic_C"] or any(c <= 0 for c in model["logistic_C"]):
        raise ValueError("Logistic regularization values must be positive.")
    if not model["class_weight_candidates"] or any(c not in (None, "balanced") for c in model["class_weight_candidates"]):
        raise ValueError("Unsupported class weights.")
    if model["max_training_rows"] > 100000 or model["max_training_rows"] <= 0:
        raise ValueError("Model driver bound must be between one and 100000.")
    if set(item["namespace"] for item in contract["tables"].values()) != {"canvas", "canvas_logs"}:
        raise ValueError("Unexpected source namespaces.")
