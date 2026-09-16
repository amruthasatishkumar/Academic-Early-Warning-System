"""Parquet serialization compatible with the Fabric/Spark timestamp reader."""
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def write_frame(data, path):
    table = pa.Table.from_pandas(data, preserve_index=False)
    timestamps = {field.name for field in table.schema if pa.types.is_timestamp(field.type)}
    schema = pa.schema([
        field.with_type(pa.timestamp("us", tz=field.type.tz)) if field.name in timestamps else field
        for field in table.schema
    ], metadata=table.schema.metadata)
    # Writer coercion alone leaves the embedded Arrow schema at nanoseconds.
    # A safe cast changes that schema too, rejecting any lossy conversion.
    table = table.cast(schema, safe=True)
    metadata = dict(table.schema.metadata or {})
    if b"pandas" in metadata:
        pandas_metadata = json.loads(metadata[b"pandas"])
        for column in pandas_metadata["columns"]:
            if column["field_name"] in timestamps:
                column["numpy_type"] = "datetime64[us]"
        metadata[b"pandas"] = json.dumps(pandas_metadata).encode("utf-8")
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, path, version="1.0", coerce_timestamps="us",
                   allow_truncated_timestamps=False)


def read_frame(path):
    return normalize_timestamp_units(pd.read_parquet(path, engine="pyarrow"))


def normalize_timestamp_units(data):
    data = data.copy()
    for column in data:
        if pd.api.types.is_datetime64_any_dtype(data[column].dtype):
            data[column] = data[column].dt.as_unit("ns")
    return data
