"""Create the named input tables from a verified folder, without training or replacement."""
import json

from aews.data_bundle import load_lakehouse_input, source_schemas
from aews.runtime import qualified, validate_session
from aews.spark_tables import _arrow_to_spark
from aews.storage import require_equivalent


FEATURE_TARGETS = {
    "features/current.parquet": "silver.current_weekly_features",
    "features/historical.parquet": "silver.historical_weekly_features",
}


def table_plan(bundle, contract):
    expected = set(source_schemas(contract)) | set(FEATURE_TARGETS)
    if set(bundle.tables) != expected:
        raise ValueError("The verified dataset is missing expected input tables.")
    return {FEATURE_TARGETS.get(name, name): table for name, table in sorted(bundle.tables.items())}


def _delta_frame(spark, arrow):
    # Delta readers can expose nullable columns even for non-null source fields.
    # The source contract has already checked required values before any write.
    return _arrow_to_spark(spark, arrow, nullable=True)


def create_input_tables(spark, config, plan):
    results, existing = [], set()
    for name, arrow in plan.items():
        schema, table = name.split(".")
        target = qualified(config, schema, table)
        if spark.catalog.tableExists(target):
            print(f"Checking existing table {name}; mismatches stop the import without replacement.")
            count = require_equivalent(_delta_frame(spark, arrow), spark.table(target), target)
            results.append({"table": name, "status": "already_matches", "rows": count})
            existing.add(name)
    for schema in sorted({name.split(".")[0] for name in plan}):
        spark.sql("CREATE SCHEMA IF NOT EXISTS " + qualified(config, schema))
    for name, arrow in plan.items():
        if name in existing:
            continue
        schema, table = name.split(".")
        target = qualified(config, schema, table)
        print(f"Creating {name} from the verified input records.")
        frame = _delta_frame(spark, arrow)
        frame.write.format("delta").mode("errorifexists").saveAsTable(target)
        count = require_equivalent(frame, spark.table(target), target)
        results.append({"table": name, "status": "created", "rows": count})
        print(json.dumps(results[-1]))
    results.sort(key=lambda row: row["table"])
    print(json.dumps({"status": "tables_loaded", "tables": len(results), "training_performed": False}))
    return results


def load_folder_tables(spark, source_config, contract, source_folder, bundle_info):
    root, _ = validate_session(spark, source_config, require_model=False)
    bundle = load_lakehouse_input(root, "folder", source_folder, bundle_info, source_config, contract)
    plan = table_plan(bundle, contract)
    print(json.dumps({"status": "folder_verified", "input_tables": len(plan), "writes_performed": False}))
    return create_input_tables(spark, source_config, plan)
