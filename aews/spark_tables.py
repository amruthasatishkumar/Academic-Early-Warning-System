import pandas as pd


def _spark_schema(arrow_schema):
    import pyarrow as pa
    from pyspark.sql.types import (ArrayType, BooleanType, DoubleType, IntegerType,
                                   LongType, StringType, StructField, StructType, TimestampType)
    fields = []
    for field in arrow_schema:
        t = field.type
        if pa.types.is_int64(t):
            spark_type = LongType()
        elif pa.types.is_int32(t):
            spark_type = IntegerType()
        elif pa.types.is_boolean(t):
            spark_type = BooleanType()
        elif pa.types.is_floating(t):
            spark_type = DoubleType()
        elif pa.types.is_timestamp(t):
            spark_type = TimestampType()
        elif pa.types.is_list(t):
            spark_type = ArrayType(StringType(), containsNull=False)
        elif pa.types.is_string(t) or pa.types.is_large_string(t):
            spark_type = StringType()
        else:
            raise TypeError(f"Unsupported explicit Arrow-to-Spark type: {t}")
        fields.append(StructField(field.name, spark_type, field.nullable))
    return StructType(fields)



def _pandas_to_spark(spark, data):
    import pyarrow as pa
    # Explicit schema preserves all-null columns (e.g. no registry version).
    fields = []
    string_names = {"registered_version", "mlflow_run_id", "label", "label_status", "model_status"}
    for name in data:
        dtype = data[name].dtype
        if pd.api.types.is_datetime64_any_dtype(dtype):
            typ = pa.timestamp("us", tz="UTC")
        elif pd.api.types.is_bool_dtype(dtype):
            typ = pa.bool_()
        elif pd.api.types.is_integer_dtype(dtype):
            typ = pa.int64()
        elif pd.api.types.is_float_dtype(dtype):
            typ = pa.float64()
        elif name in string_names or pd.api.types.is_string_dtype(dtype) or dtype == object:
            typ = pa.string()
        else:
            raise TypeError(f"Unsupported output type: {name} {dtype}")
        fields.append(pa.field(name, typ))
    schema = pa.schema(fields)
    copied = data.copy()
    for name in copied:
        if pd.api.types.is_datetime64_any_dtype(copied[name].dtype):
            copied[name] = pd.to_datetime(copied[name], utc=True).dt.floor("us")
    arrow = pa.Table.from_pandas(copied, schema=schema, preserve_index=False, safe=True)
    records = []
    for row in arrow.to_pylist():
        records.append(tuple(v.to_pydatetime() if isinstance(v, pd.Timestamp) else v for v in row.values()))
    return spark.createDataFrame(records, schema=_spark_schema(schema))


