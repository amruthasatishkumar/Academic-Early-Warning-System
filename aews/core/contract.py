import ipaddress
import json
import uuid

import numpy as np
import pandas as pd

HISTORY = ("effective_at", "recorded_at", "revision")


def frame(table, records, contract, history=False):
    specs = contract["tables"][table]["columns"]
    columns = list(specs) + (list(HISTORY) if history else [])
    result = pd.DataFrame.from_records(records, columns=columns)
    for name, spec in specs.items():
        kind = spec["type"]
        if kind in ("int64", "int32"):
            present = result[name].dropna()
            if any(isinstance(v, (float, np.floating)) and v != int(v) for v in present):
                raise ValueError(f"Fractional identifier: {table}.{name}")
            result[name] = result[name].astype("Int64" if kind == "int64" else "Int32")
        elif kind == "float64":
            result[name] = result[name].astype("float64")
        elif kind == "bool":
            result[name] = result[name].astype("boolean")
        elif kind == "datetime":
            result[name] = pd.to_datetime(result[name], utc=True).dt.floor("us")
        elif kind == "string":
            result[name] = result[name].astype("string")
    if history:
        for name in HISTORY[:2]:
            result[name] = pd.to_datetime(result[name], utc=True).dt.floor("us")
        result["revision"] = result["revision"].astype("int32")
    return result


def as_of(revisions, cutoff):
    available = revisions.loc[
        (revisions.effective_at <= cutoff) & (revisions.recorded_at <= cutoff)
    ]
    # Event order and availability are independent: a late-arriving old revision
    # must not displace an already-observed newer source revision.
    return available.sort_values(["id", "effective_at", "revision"]).drop_duplicates("id", keep="last").reset_index(drop=True)


def validate_table(table, data, contract, references=None, history=False):
    specs = contract["tables"][table]["columns"]
    required_columns = list(specs) + (list(HISTORY) if history else [])
    if list(data.columns) != required_columns:
        raise ValueError(f"Column projection mismatch: {table}")
    key = ["id", "revision"] if history else ["id"]
    if data.duplicated(key).any():
        raise ValueError(f"Duplicate key in {table}")
    for name, spec in specs.items():
        values = data[name]
        present = values.dropna()
        if not spec["nullable"] and values.isna().any():
            raise ValueError(f"Null mandatory field: {table}.{name}")
        kind = spec["type"]
        if kind.startswith("int") and str(values.dtype).lower() != kind:
            raise ValueError(f"Wrong integer width: {table}.{name}: {values.dtype}")
        if kind == "float64" and (str(values.dtype) != "float64" or not np.isfinite(present).all()):
            raise ValueError(f"Nonfinite/wrong numeric type: {table}.{name}")
        if kind == "bool" and str(values.dtype) != "boolean":
            raise ValueError(f"Wrong boolean type: {table}.{name}")
        if kind == "string" and str(values.dtype) != "string":
            raise ValueError(f"Wrong string type: {table}.{name}")
        if kind == "datetime" and not isinstance(values.dtype, pd.DatetimeTZDtype):
            raise ValueError(f"Missing UTC timestamp: {table}.{name}")
        if kind == "array<string>":
            if any(not isinstance(v, list) or any(not isinstance(x, str) for x in v) for v in present):
                raise ValueError(f"Invalid array: {table}.{name}")
            if spec["enum"] and any(not set(v) <= set(spec["enum"]) for v in present):
                raise ValueError(f"Invalid array enum: {table}.{name}")
        elif spec["enum"] and not present.isin(spec["enum"]).all():
            raise ValueError(f"Invalid enum: {table}.{name}")
        if spec["max_length"] and present.map(len).gt(spec["max_length"]).any():
            raise ValueError(f"String exceeds public maximum: {table}.{name}")
        if spec["format"] == "uuid":
            for value in present:
                if str(uuid.UUID(value)) != value:
                    raise ValueError(f"Noncanonical UUID in {table}")
        if spec["format"] == "ip":
            for value in present:
                ipaddress.ip_address(value)
        ref = spec["reference"]
        if references is not None and ref:
            # CD2 explicitly permits external web-log users; they are counted
            # and excluded from learner features, not manufactured as local users.
            if table == "web_logs" and name == "user_id":
                continue
            if ref not in references:
                raise ValueError(f"Missing reference target {ref}")
            if not present.isin(references[ref]["id"]).all():
                raise ValueError(f"Orphan reference: {table}.{name} -> {ref}")
    if history:
        if data[list(HISTORY)].isna().any().any():
            raise ValueError(f"Null revision metadata: {table}")
        if (data.recorded_at < data.effective_at).any():
            raise ValueError(f"Observation precedes event: {table}")
    return len(data)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def dumps(value):
    return json.dumps(json_safe(value), sort_keys=True, allow_nan=False)
