import argparse
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aews.configuration import load_configuration
from aews.execution import code_hash

READ_MODULES = (
    "aews", "aews.runtime", "aews.presentation", "aews.presentation.explainability",
    "aews.presentation.reader",
)
BUILD_MODULES = READ_MODULES + (
    "aews.core", "aews.core.contract", "aews.core.simulation", "aews.core.features",
    "aews.core.execution", "aews.configuration", "aews.model", "aews.model.modeling",
    "aews.parquet", "aews.execution", "aews.data_bundle", "aews.spark_tables", "aews.storage", "aews.fabric",
)


def module_bundle(root, names):
    modules, packages = {}, []
    for name in names:
        path = root.joinpath(*name.split("."))
        if path.is_dir():
            packages.append(name)
            path = path / "__init__.py"
        else:
            path = path.with_suffix(".py")
        text = path.read_text(encoding="ascii")
        ast.parse(text)
        modules[name] = text
    return (
        "import sys\nimport types\n"
        "EMBEDDED_MODULES = " + repr(modules) + "\n"
        "EMBEDDED_PACKAGES = " + repr(packages) + "\n"
        "for name, source in EMBEDDED_MODULES.items():\n"
        "    module = types.ModuleType(name)\n"
        "    module.__file__ = '<embedded:' + name + '>'\n"
        "    module.__package__ = name if name in EMBEDDED_PACKAGES else name.rpartition('.')[0]\n"
        "    if name in EMBEDDED_PACKAGES:\n"
        "        module.__path__ = []\n"
        "    sys.modules[name] = module\n"
        "    if '.' in name:\n"
        "        parent, _, child = name.rpartition('.')\n"
        "        setattr(sys.modules[parent], child, module)\n"
        "    exec(compile(source, module.__file__, 'exec'), module.__dict__)\n"
        "print('Self-contained Academic Early Warning System source loaded.')\n"
    )


def code_cell(text, number):
    ast.parse(text)
    return {"cell_type": "code", "id": f"code-{number}", "execution_count": None,
            "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def build_notebooks(root=ROOT, config_dir=None, output_dir=None):
    root = Path(root)
    source, model, contract = load_configuration(config_dir or root / "config")
    bundle_info = json.loads((root / "sample-data" / "bundle-info.json").read_text(encoding="utf-8"))
    output_dir = Path(output_dir or root / "notebooks")
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = "import json\nSOURCE_CONFIG = json.loads(" + repr(json.dumps(source)) + ")\n"
    build_settings = settings + (
        "MODEL_CONFIG = json.loads(" + repr(json.dumps(model)) + ")\n"
        "SOURCE_CONTRACT = json.loads(" + repr(json.dumps(contract)) + ")\n"
        "CODE_FINGERPRINT = " + repr(code_hash(root)) + "\n"
        "BUNDLE_INFO = json.loads(" + repr(json.dumps(bundle_info)) + ")\n"
        "INPUT_MODE = 'uploaded'  # Included dataset; 'generate' builds the larger full dataset.\n"
        "UPLOAD_PATH = BUNDLE_INFO['expected_lakehouse_path']\n"
        "SOURCE_CONFIG['mode'] = 'small' if INPUT_MODE == 'uploaded' else 'full'\n"
        "RUN_MODE = 'validate'  # Change to 'publish' only after validation and scope review.\n"
        "ALLOW_DEMO_TABLE_REPLACEMENT = False\n"
    )
    build = [
        "# Academic Early Warning System: build and publish\n"
        "# Synthetic course outcomes: Excellent, Pass, Fail, Withdrawn.\n"
        "# START HERE: Do not use Load to Tables for the replication Quickstart.\n"
        "# Extract the outer repository ZIP, but keep the inner simulated-data ZIP compressed.\n"
        "# In the Lakehouse: Files -> uploads -> Upload files -> choose the inner ZIP file.\n"
        "# Do not select Parquet/CSV or a destination table name. This notebook creates tables.\n"
        "# Attach the configured schema-enabled Lakehouse in its configured workspace.\n"
        "# Runtime 1.3: Python 3.11, Spark 3.5, Delta 3.2. No package installs here.\n"
        "# Upload sample-data/academic-early-warning-simulated-data.zip into Lakehouse Files/uploads.\n"
        "# Prepared early features are included; uploaded mode does not regenerate source records.\n"
        "# Default is read-only environment/data validation. Publication requires two explicit edits.\n"
        "# Terms 1-2 train; separate term-3 folds calibrate and select one logistic model.\n"
        "# Term 4 is a previously exposed synthetic diagnostic, never independent validation.\n"
        "# No real Canvas adapter, accepted-model registration or automated student decisions.\n"
        "print('Upload academic-early-warning-simulated-data.zip unchanged to Files/uploads. Do not use Load to Tables.')\n",
        module_bundle(root, BUILD_MODULES),
        build_settings,
        "from aews.configuration import validate_configuration\n"
        "from aews.runtime import validate_session\n"
        "from aews.fabric import run_fabric, validate_uploaded_data\n"
        "validate_configuration(SOURCE_CONFIG, MODEL_CONFIG, SOURCE_CONTRACT)\n"
        "if RUN_MODE == 'validate':\n"
        "    _, runtime_versions = validate_session(spark, SOURCE_CONFIG)\n"
        "    if INPUT_MODE == 'uploaded':\n"
        "        DATASET = validate_uploaded_data(spark, SOURCE_CONFIG, MODEL_CONFIG, SOURCE_CONTRACT, UPLOAD_PATH, BUNDLE_INFO)\n"
        "        print({'dataset': 'verified', 'profile': 'small', 'prepared_feature_rows': len(DATASET.features),\n"
        "               'source_counts': DATASET.summary['observed_source_counts']})\n"
        "    elif INPUT_MODE != 'generate':\n"
        "        raise ValueError('INPUT_MODE must be uploaded or generate.')\n"
        "    print({'validation': 'passed', 'runtime': runtime_versions, 'writes_performed': False})\n"
        "elif RUN_MODE == 'publish':\n"
        "    PUBLICATION = run_fabric(spark, SOURCE_CONFIG, MODEL_CONFIG, SOURCE_CONTRACT,\n"
        "                             CODE_FINGERPRINT, ALLOW_DEMO_TABLE_REPLACEMENT,\n"
        "                             INPUT_MODE, UPLOAD_PATH, BUNDLE_INFO)\n"
        "else:\n"
        "    raise ValueError('RUN_MODE must be validate or publish.')\n",
        "if RUN_MODE == 'publish':\n"
        "    print(json.dumps(PUBLICATION['manifest'], indent=2))\n"
        "    print(json.dumps(PUBLICATION['summary']['observed_source_counts'], indent=2))\n"
        "# Review actual class-aware development metrics and source assertions.\n"
        "# Open the exploration notebook only after publication_completed appears.\n",
    ]
    explore = [
        "# Academic Early Warning System: explore and explain\n"
        "# READ ONLY. This notebook never fits, publishes, registers or contacts students.\n"
        "# It follows current.json to immutable uploaded-small or generated-full snapshots.\n"
        "# Use only your trusted workspace's own MLflow model artifacts.\n"
        "# Contributions are underlying linear class-score units before calibration,\n"
        "# not probability changes, percentages, feature significance or causal effects.\n",
        module_bundle(root, READ_MODULES),
        settings,
        "from aews.presentation.reader import read_publication\n"
        "VIEW = read_publication(spark, SOURCE_CONFIG)\n"
        "print('Dataset profile:', VIEW['manifest']['source_profile'])\n"
        "print(json.dumps(VIEW['manifest'], indent=2))\n"
        "display(VIEW['frames']['gold.data_quality_summary'])\n"
        "display(VIEW['training'].head(12))\n"
        "print('Terms 1-2 train. Term 3 development evaluation. Term 4 previously exposed, not independent.')\n",
        "from aews.presentation.explainability import OUTCOMES\n"
        "GLOBAL_CONTRIBUTIONS = VIEW['explainer'].training_importance(VIEW['training'])\n"
        "for outcome in OUTCOMES:\n"
        "    print(outcome.upper() + ': mean absolute training-centered linear-score contributions')\n"
        "    display(GLOBAL_CONTRIBUTIONS.loc[GLOBAL_CONTRIBUTIONS.outcome == outcome].sort_values(\n"
        "        'mean_absolute_score_contribution', ascending=False))\n"
        "import pandas as pd\n"
        "evaluation = VIEW['frames']['gold.model_evaluation']\n"
        "selected = evaluation.loc[evaluation.selected & (evaluation['split'] == 'term3_decision') &\n"
        "                          (evaluation['slice'] == 'all')]\n"
        "if len(selected) != 1:\n"
        "    raise ValueError('Expected exactly one selected development evaluation.')\n"
        "measured = json.loads(selected.iloc[0].metrics_json)\n"
        "display(pd.DataFrame([{'outcome': outcome,\n"
        "    **{key: measured['classification_report'][outcome][key] for key in ('precision', 'recall', 'f1-score', 'support')},\n"
        "    **{key: measured['per_class'][outcome][key] for key in ('prevalence', 'average_precision', 'brier')}}\n"
        "    for outcome in OUTCOMES]))\n"
        "display(pd.DataFrame(measured['confusion_matrix'], index=measured['confusion_class_order'],\n"
        "                     columns=measured['confusion_class_order']))\n"
        "print('Confusion matrix: actual rows, predicted columns. Full baseline and slice metrics remain in model_evaluation.')\n"
        "print('Synthetic development metrics only; no independent institutional accuracy claim.')\n",
        "# Enrollment ID comes from canvas.enrollments.id, carried into features and predictions.\n"
        "# It identifies one student's enrollment in one course, not just the student.\n"
        "# predicted_class comes from the model's highest probability across the four outcomes.\n"
        "# It is a prediction, NOT a known future course result.\n"
        "available_predictions = VIEW['predictions'].sort_values('enrollment_id')\n"
        "if available_predictions.empty:\n"
        "    raise ValueError('No current predictions are available; complete publication first.')\n"
        "display(available_predictions[['enrollment_id', 'course_id', 'predicted_class',\n"
        "    'p_excellent', 'p_pass', 'p_fail', 'p_withdrawn', 'top_predicted_class_drivers']].head(30))\n"
        "\n"
        "# Prefilled from real rows in THIS synthetic publication, so both small/full profiles work.\n"
        "# To choose another enrollment, replace the next expression with a displayed enrollment_id.\n"
        "ENROLLMENT_TO_EXPLAIN = int(available_predictions.iloc[0].enrollment_id)\n"
        "selected_prediction = available_predictions.loc[\n"
        "    available_predictions.enrollment_id == ENROLLMENT_TO_EXPLAIN]\n"
        "if len(selected_prediction) != 1:\n"
        "    raise ValueError('Choose one enrollment_id from the current predictions table.')\n"
        "\n"
        "# Automatically prefill the selected enrollment's actual predicted outcome.\n"
        "# Or replace this expression with 'excellent', 'pass', 'fail', or 'withdrawn'.\n"
        "# Choosing another outcome explains THAT class's score; it does not change the prediction.\n"
        "OUTCOME_TO_EXPLAIN = str(selected_prediction.iloc[0].predicted_class)\n"
        "print(f'ENROLLMENT_TO_EXPLAIN = {ENROLLMENT_TO_EXPLAIN}')\n"
        "print(f'OUTCOME_TO_EXPLAIN = {OUTCOME_TO_EXPLAIN!r}')\n"
        "print('Run the next cell to explain these populated values. No training or prediction is changed.')\n",
        "row = VIEW['current'].loc[VIEW['current'].enrollment_id == ENROLLMENT_TO_EXPLAIN]\n"
        "prediction = VIEW['predictions'].loc[VIEW['predictions'].enrollment_id == ENROLLMENT_TO_EXPLAIN]\n"
        "if len(row) != 1 or len(prediction) != 1:\n"
        "    raise ValueError('Choose one published enrollment.')\n"
        "if OUTCOME_TO_EXPLAIN not in OUTCOMES:\n"
        "    raise ValueError('OUTCOME_TO_EXPLAIN must be excellent, pass, fail, or withdrawn.')\n"
        "outcome = OUTCOME_TO_EXPLAIN\n"
        "contributions = VIEW['explainer'].contributions(row, outcome)\n"
        "baseline = float(VIEW['explainer'].baseline_scores[VIEW['explainer'].classes.index(outcome)])\n"
        "display(prediction)\n"
        "display(contributions.T.rename(columns={row.index[0]: 'signed_score_contribution'}))\n"
        "print({'outcome': outcome, 'training_reference_score': baseline,\n"
        "       'contribution_sum': float(contributions.sum(axis=1).iloc[0]),\n"
        "       'underlying_score': baseline + float(contributions.sum(axis=1).iloc[0])})\n"
        "print('All terms reconstruct the underlying score. Top-three drivers alone do not.')\n",
    ]
    paths = []
    for filename, cells in (("01_build_and_publish.ipynb", build), ("02_explore_and_explain.ipynb", explore)):
        notebook = {
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {"display_name": "Synapse PySpark", "language": "python", "name": "synapse_pyspark"},
                "language_info": {"name": "python", "version": "3.11"},
                "description": "Academic Early Warning System: self-contained synthetic replication starter.",
            },
            "cells": [code_cell(text, index) for index, text in enumerate(cells, start=1)],
        }
        path = output_dir / filename
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=True) + "\n", encoding="ascii")
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Build two self-contained importable Fabric notebooks without cloud access.")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "config")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "notebooks")
    args = parser.parse_args()
    for path in build_notebooks(config_dir=args.config_dir, output_dir=args.output_dir):
        print("Built " + str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path.name))


if __name__ == "__main__":
    main()
