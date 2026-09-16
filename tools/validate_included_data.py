"""Verify the included dataset and run its prepared features through the model locally."""
import copy
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from aews.configuration import load_configuration
from aews.data_bundle import FILENAME, load_bundle
from aews.execution import code_hash, finish_run
from aews.fabric import prepare_input


def main():
    source, model, contract = load_configuration(ROOT / "config")
    info = json.loads((ROOT / "sample-data" / "bundle-info.json").read_text())
    bundle = load_bundle(ROOT / "sample-data" / FILENAME, info, source, contract)
    with tempfile.TemporaryDirectory(prefix="aews-verify-") as temporary:
        features, summary = prepare_input(source, contract, Path(temporary) / "sources", bundle)
        result, report, tables, importance = finish_run(
            features, copy.deepcopy(summary), model, code_hash(ROOT), pd.Timestamp.now(tz="UTC"))
    print(json.dumps({"status": "included_dataset_verified", "data_generated": False,
                      "source_records": report["observed_source_counts"], "profile": report["profile"],
                      "predictions": len(tables["gold.student_course_predictions"]),
                      "explanations": len(tables["gold.prediction_explanations"]),
                      "global_feature_rows": len(importance), "registered": False,
                      "model_version": result.version}))


if __name__ == "__main__":
    main()
