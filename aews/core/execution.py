import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn

from aews.core.contract import HISTORY, dumps, validate_table
from aews.core.features import FEATURE_VERSION, FEATURES, feature_contract, labels, weekly_features
from aews.core.simulation import DAY, STATIC, dimensions, generate_course, validate_batch, validate_config


def arrow_schema(table, contract, history=False):
    mapping = {"int64": pa.int64(), "int32": pa.int32(), "float64": pa.float64(), "bool": pa.bool_(),
               "datetime": pa.timestamp("us", tz="UTC"), "string": pa.string(), "array<string>": pa.list_(pa.string())}
    fields = [pa.field(name, mapping[spec["type"]], nullable=spec["nullable"])
              for name, spec in contract["tables"][table]["columns"].items()]
    if history:
        fields += [pa.field("effective_at", pa.timestamp("us", tz="UTC"), nullable=False),
                   pa.field("recorded_at", pa.timestamp("us", tz="UTC"), nullable=False),
                   pa.field("revision", pa.int32(), nullable=False)]
    return pa.schema(fields)


SUPPORT_SCHEMA = {
    "course_withdrawal_events": pa.schema([
        ("id", pa.int64()), ("enrollment_id", pa.int64()), ("effective_at", pa.timestamp("us", tz="UTC")),
        ("recorded_at", pa.timestamp("us", tz="UTC")), ("event_type", pa.string())]),
    "gradebook_qualification": pa.schema([
        ("enrollment_id", pa.int64()), ("effective_at", pa.timestamp("us", tz="UTC")),
        ("recorded_at", pa.timestamp("us", tz="UTC")), ("qualified", pa.bool_()), ("reason", pa.string()),
        ("course_finished_at", pa.timestamp("us", tz="UTC"))]),
}


class ParquetSourceSink:
    """One bounded course at a time; at most one writer per table per term."""
    def __init__(self, root, contract):
        self.root, self.contract = Path(root), contract
        self.writers = {}
        self.term = None

    def write(self, namespace, table, data, schema):
        key = namespace, table
        path = self.root / namespace / table
        path.mkdir(parents=True, exist_ok=True)
        if key not in self.writers:
            self.writers[key] = pq.ParquetWriter(str(path / f"part-{self.term or 0:02d}.parquet"), schema, compression="snappy")
        arrow = pa.Table.from_pandas(data, schema=schema, preserve_index=False, safe=True)
        arrow.validate(full=True)
        self.writers[key].write_table(arrow, row_group_size=25000)

    def dimensions(self, dims):
        for table, data in dims.items():
            self.write("canvas", table, data, arrow_schema(table, self.contract))
        self.close()

    def batch(self, batch):
        if batch.term != self.term:
            self.close()
            self.term = batch.term
        for table, data in batch.revisions.items():
            namespace = self.contract["tables"][table]["namespace"]
            self.write(namespace, table, batch.source(table).drop(columns=list(HISTORY)), arrow_schema(table, self.contract))
            self.write("sim_history", table + "_revisions", data, arrow_schema(table, self.contract, history=True))
        self.write("sim_support", "course_withdrawal_events", batch.withdrawal_events, SUPPORT_SCHEMA["course_withdrawal_events"])
        self.write("sim_support", "gradebook_qualification", batch.qualification, SUPPORT_SCHEMA["gradebook_qualification"])

    def close(self):
        for writer in self.writers.values():
            writer.close()
        self.writers = {}


def environment_versions():
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "pyarrow": pa.__version__}


def prepare(config, contract, sink=None):
    size = validate_config(config)
    dims = dimensions(config, contract)
    for name, data in dims.items():
        validate_table(name, data, contract, dims)
    if sink is not None:
        sink.dimensions(dims)
    counts = Counter({k: len(v) for k, v in dims.items()})
    revisions = Counter()
    scenario_counts = Counter()
    frames, excluded_frames = [], []
    prior_max = {}
    seen_courses = set()
    try:
        for term in range(1, 6):
            term_counts = Counter()
            for course_number in range(1, size["courses_per_term"] + 1):
                batch = generate_course(config, contract, term, course_number)
                if batch.course_id in seen_courses:
                    raise ValueError("Duplicate course chunk")
                seen_courses.add(batch.course_id)
                batch_counts = validate_batch(batch, dims, contract)
                for table, amount in batch_counts.items():
                    counts[table] += amount
                    term_counts[table] += amount
                    history = batch.revisions[table]
                    revisions[table] += len(history)
                    # Disjoint monotone integer key ranges validate cross-chunk
                    # PK uniqueness without an unbounded Python set of IDs.
                    if table != "web_logs" and len(history):
                        lower, upper = int(history.id.min()), int(history.id.max())
                        if lower <= prior_max.get(table, -1):
                            raise ValueError(f"Overlapping cross-course PK range: {table}")
                        prior_max[table] = upper
                for week in config["cutoff_weeks"]:
                    if term == 5 and week > config["current_week"]:
                        continue
                    features = weekly_features(batch, week)
                    if term < 5:
                        outcome = labels(batch, batch.start + 7 * week * DAY)
                        features = features.merge(outcome, on="enrollment_id", how="left", validate="one_to_one")
                        excluded = features.loc[features.label.isna(), ["enrollment_id", "term", "cutoff_week", "label_status"]]
                        excluded_frames.append(excluded)
                    else:
                        # Current silver table will not contain label columns.
                        features["label"] = None
                        features["label_status"] = "current_term_no_truth_published"
                        features["label_available_at"] = pd.NaT
                    frames.append(features)
                logs = batch.source("web_logs")
                scenario_counts["anonymous_web_requests"] += int(logs.user_id.isna().sum())
                scenario_counts["external_web_requests"] += int((logs.user_id.notna() & ~logs.user_id.isin(dims["users"].id)).sum())
                sub = batch.source("submissions")
                scenario_counts["reattempted_submissions"] += int(sub.attempt.gt(1).sum())
                scenario_counts["excused_submissions"] += int(sub.excused.fillna(False).sum())
                scenario_counts["graded_without_submitted_timestamp"] += int((sub.graded_at.notna() & sub.submitted_at.isna() & sub.score.gt(0)).sum())
                scenario_counts["confirmed_withdrawals"] += len(batch.withdrawal_events)
                scenario_counts["individual_extensions"] += len(batch.source("assignment_overrides"))
                if term < 5:
                    scores = batch.source("scores")
                    withdrawn = set(batch.withdrawal_events.enrollment_id)
                    scenario_counts["passing_grade_withdrawals"] += int((scores.enrollment_id.isin(withdrawn) & scores.final_score.ge(60)).sum())
                    scenario_counts["failing_without_withdrawal"] += int((~scores.enrollment_id.isin(withdrawn) & scores.final_score.lt(60)).sum())
                    scenario_counts["unqualified_gradebooks"] += int((~batch.qualification.qualified).sum())
                if sink is not None:
                    sink.batch(batch)
            print(dumps({"term": term, "source_counts": term_counts}), flush=True)
    finally:
        if sink is not None:
            sink.close()
    features = pd.concat(frames, ignore_index=True)
    features["label_available_at"] = pd.to_datetime(features.label_available_at, utc=True)
    if features.duplicated(["enrollment_id", "cutoff_week"]).any():
        raise ValueError("Duplicate feature snapshot key")
    if features.groupby("enrollment_id").partition.nunique().gt(1).any():
        raise ValueError("Temporal split crosses enrollment")
    if features.loc[features.term == 5, "label"].notna().any():
        raise ValueError("Current-term label contamination")
    if not all(scenario_counts[name] > 0 for name in (
        "confirmed_withdrawals", "passing_grade_withdrawals", "failing_without_withdrawal", "unqualified_gradebooks")):
        raise ValueError(f"Required stochastic counterexample absent; do not silently regenerate: {scenario_counts}")
    excluded = pd.concat(excluded_frames, ignore_index=True)
    target = {
        "student_users": size["students"], "teacher_users": size["courses_per_term"],
        "courses": 5 * size["courses_per_term"], "sections": 5 * size["courses_per_term"] * size["sections_per_course"],
        "student_enrollments": 5 * size["courses_per_term"] * size["enrollments_per_course"],
        "submission_records": 5 * size["courses_per_term"] * size["enrollments_per_course"] * size["assignments_per_course"],
        "full_profile_web_log_target": "2-3 million full longitudinal events; published current term stops at cutoff, so published count is lower",
    }
    summary = {
        "synthetic": True, "profile": config["mode"], "seed": config["seed"], "target_counts": target,
        "observed_source_counts": dict(counts), "observed_history_counts": dict(revisions),
        "observed_scenarios": dict(scenario_counts), "feature_snapshots": len(features),
        "label_counts_by_term": features.groupby(["term", "label"], dropna=False).size().rename("rows").reset_index().to_dict("records"),
        "excluded_label_counts": excluded.label_status.value_counts().to_dict(),
        "assertions": ["all_selected_source_types_nullability_enums_checked", "no_duplicate_source_PK",
                       "cross_course_integer_PK_ranges_disjoint", "web_UUIDs_derived_from_unique_course_and_request",
                       "mandatory_projected_FK_integrity", "section_course_consistency", "grade_point_bounds",
                       "source_revision_both_times_checked", "current_future_truth_not_published",
                       "snapshot_keys_unique", "enrollment_temporal_partitions_disjoint"],
        "environment": environment_versions(),
        "limitations": [
            "Selected source projection, not complete Canvas replication.",
            "Historical web data is synthetic; production DAP retention is 30 days.",
            "All term-four course offering IDs are new by construction; this is not evidence of unseen curriculum generalization.",
            "Returning learners cross historical terms; the temporal test is not entirely learner-disjoint.",
            "Unknown gradebooks remain unlabeled, rather than becoming fail.",
            "No causal or real-world validity claim; all performance describes synthetic data.",
            "Week-four model only. Optional week-two/six snapshots are not pooled into training or represented as validated deployments.",
        ],
    }
    return features, summary


def evaluation_frame(result):
    return pd.DataFrame([{
        "model_version": result.version, "model_name": row["model_name"], "split": row["split"],
        "slice": row["slice"], "metrics_json": dumps(row["metrics"]), "synthetic": True,
        "selected": row["model_name"] == result.model_name,
    } for row in result.evaluations])


def quality_frame(summary):
    return pd.DataFrame([{"category": key, "details_json": dumps(value), "synthetic": True}
                         for key, value in summary.items()])

