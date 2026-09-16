"""Parquet serialization compatible with the Fabric/Spark timestamp reader."""
import pandas as pd


def write_frame(data, path):
    # Spark cannot load Parquet TIMESTAMP(NANOS). Reject sub-microsecond data
    # rather than silently rounding observation times used by early features.
    data.to_parquet(path, index=False, engine="pyarrow", coerce_timestamps="us",
                    allow_truncated_timestamps=False)


def read_frame(path):
    return normalize_timestamp_units(pd.read_parquet(path, engine="pyarrow"))


def normalize_timestamp_units(data):
    data = data.copy()
    for column in data:
        if pd.api.types.is_datetime64_any_dtype(data[column].dtype):
            data[column] = data[column].dt.as_unit("ns")
    return data
