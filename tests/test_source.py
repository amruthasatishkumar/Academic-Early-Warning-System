import ast
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aews.core.contract import HISTORY, as_of, frame, validate_table
from aews.core.execution import arrow_schema
from aews.core.features import CLASSES, FEATURES, PARTITIONS, labels, weekly_features
from aews.model.modeling import predictors
from aews.core.simulation import DAY, dimensions, generate_course, validate_batch


class SyntheticContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "config" / "simulation.json").read_text(encoding="utf-8"))
        cls.contract = json.loads((ROOT / "config" / "canvas-source-contract.json").read_text(encoding="utf-8"))
        cls.batch = generate_course(cls.config, cls.contract, 1, 1)
        cls.dims = dimensions(cls.config, cls.contract)

    def test_reproducible_source_and_features(self):
        repeated = generate_course(self.config, self.contract, 1, 1)
        for table, data in self.batch.revisions.items():
            pd.testing.assert_frame_equal(data, repeated.revisions[table])
        pd.testing.assert_frame_equal(weekly_features(self.batch, 4), weekly_features(repeated, 4))

    def test_pk_fk_types_enums_nullability_and_arrow_schema(self):
        validate_batch(self.batch, self.dims, self.contract)
        import pyarrow as pa
        for table, history in self.batch.revisions.items():
            source = self.batch.source(table).drop(columns=list(HISTORY))
            array = pa.Table.from_pandas(source, schema=arrow_schema(table, self.contract), preserve_index=False, safe=True)
            array.validate(full=True)
            for field in array.schema:
                self.assertEqual(field.nullable, self.contract["tables"][table]["columns"][field.name]["nullable"])
        invalid = self.dims["users"].copy()
        invalid.loc[0, "workflow_state"] = "invented"
        with self.assertRaisesRegex(ValueError, "enum"):
            validate_table("users", invalid, self.contract)
        invalid = self.dims["users"].copy()
        invalid.loc[0, "id"] = pd.NA
        with self.assertRaisesRegex(ValueError, "Null"):
            validate_table("users", invalid, self.contract)
        duplicate = pd.concat([self.dims["users"], self.dims["users"].head(1)], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_table("users", duplicate, self.contract)
        invalid_batch = copy.deepcopy(self.batch)
        invalid_batch.revisions["enrollments"].loc[0, "course_id"] = 999999
        with self.assertRaisesRegex(ValueError, "Orphan"):
            validate_batch(invalid_batch, self.dims, self.contract)
        invalid = self.dims["users"].copy()
        invalid["id"] = invalid.id.astype(float)
        with self.assertRaisesRegex(ValueError, "integer"):
            validate_table("users", invalid, self.contract)
        self.assertIn("on_paper", self.contract["tables"]["assignments"]["columns"]["submission_types"]["enum"])

    def test_future_mutation_invariance(self):
        altered = copy.deepcopy(self.batch)
        cutoff = self.batch.start + 28 * DAY
        for table, history in altered.revisions.items():
            future = (history.effective_at > cutoff) | (history.recorded_at > cutoff)
            for col in ("score", "final_score", "current_score"):
                if col in history:
                    history.loc[future, col] = 999.
            if "submitted_at" in history:
                history.loc[future, "submitted_at"] = cutoff - DAY
            if "due_at" in history:
                history.loc[future, "due_at"] = cutoff - 10 * DAY
            if "workflow_state" in history:
                history.loc[future, "workflow_state"] = "deleted"
            if table == "web_logs":
                history.loc[future, "timestamp"] = cutoff - DAY
        future_wd = ((altered.withdrawal_events.effective_at > cutoff) |
                     (altered.withdrawal_events.recorded_at > cutoff))
        altered.withdrawal_events.loc[future_wd, "enrollment_id"] = -999
        pd.testing.assert_frame_equal(weekly_features(self.batch, 4), weekly_features(altered, 4))

    def fixture(self):
        batch = copy.deepcopy(self.batch)
        e = batch.revisions["enrollments"].loc[lambda d: d.type == "StudentEnrollment"].iloc[0].to_dict()
        e.update(start_at=batch.start, workflow_state="active", effective_at=batch.start - DAY,
                 recorded_at=batch.start, revision=1)
        a = batch.revisions["assignments"].iloc[0].to_dict()
        a.update(due_at=batch.start + 20 * DAY, unlock_at=batch.start, points_possible=100.,
                 submission_types=["on_paper"])
        for table, data in batch.revisions.items():
            batch.revisions[table] = data.iloc[:0].copy()
        batch.revisions["enrollments"] = frame("enrollments", [e], self.contract, history=True)
        batch.revisions["assignments"] = frame("assignments", [a], self.contract, history=True)
        s = {"id": 1, "user_id": e["user_id"], "assignment_id": a["id"], "submitted_at": None,
             "graded_at": None, "score": None, "attempt": 0, "excused": False, "submission_type": None,
             "created_at": batch.start, "updated_at": batch.start, "workflow_state": "unsubmitted",
             "effective_at": batch.start, "recorded_at": batch.start, "revision": 1}
        batch.revisions["submissions"] = frame("submissions", [s], self.contract, history=True)
        batch.withdrawal_events = batch.withdrawal_events.iloc[:0].copy()
        return batch, e, a, s

    def test_extension_requires_effective_and_recorded_availability(self):
        batch, e, a, _ = self.fixture()
        ov = {"id": 1, "assignment_id": a["id"], "set_type": "ADHOC", "set_id": None,
              "due_at": batch.start + 35 * DAY, "due_at_overridden": True, "title": "Extension",
              "created_at": batch.start + 19 * DAY, "updated_at": batch.start + 19 * DAY,
              "workflow_state": "active", "effective_at": batch.start + 19 * DAY,
              "recorded_at": batch.start + 29 * DAY, "revision": 1}
        mapping = {"id": 1, "user_id": e["user_id"], "assignment_id": a["id"], "assignment_override_id": 1,
                   "created_at": ov["created_at"], "updated_at": ov["updated_at"], "workflow_state": "active",
                   "effective_at": ov["effective_at"], "recorded_at": ov["recorded_at"], "revision": 1}
        batch.revisions["assignment_overrides"] = frame("assignment_overrides", [ov], self.contract, history=True)
        batch.revisions["assignment_override_students"] = frame("assignment_override_students", [mapping], self.contract, history=True)
        self.assertEqual(weekly_features(batch, 4).iloc[0].due_count, 1)
        for table in ("assignment_overrides", "assignment_override_students"):
            batch.revisions[table]["recorded_at"] = batch.start + 19 * DAY
        self.assertEqual(weekly_features(batch, 4).iloc[0].due_count, 0)
        batch.revisions["assignment_overrides"]["effective_at"] = batch.start + 29 * DAY
        self.assertEqual(weekly_features(batch, 4).iloc[0].due_count, 1)

    def test_grading_delay_paper_and_excusal_availability(self):
        batch, _, _, s = self.fixture()
        grade = dict(s, score=80., graded_at=batch.start + 25 * DAY, updated_at=batch.start + 25 * DAY,
                     effective_at=batch.start + 25 * DAY, recorded_at=batch.start + 30 * DAY,
                     workflow_state="graded", revision=2)
        batch.revisions["submissions"] = frame("submissions", [s, grade], self.contract, history=True)
        row = weekly_features(batch, 4).iloc[0]
        self.assertTrue(pd.isna(row.grade_mean))
        self.assertEqual(row.paper_unknown_ratio, 1)
        self.assertTrue(pd.isna(row.missing_online_ratio))
        batch.revisions["submissions"].loc[1, "recorded_at"] = batch.start + 27 * DAY
        row = weekly_features(batch, 4).iloc[0]
        self.assertEqual(row.grade_mean, 80)
        self.assertEqual(row.completion_ratio, 1)
        excusal = dict(grade, excused=True, effective_at=batch.start + 27 * DAY,
                       recorded_at=batch.start + 29 * DAY, revision=3)
        batch.revisions["submissions"] = frame("submissions", [s, grade, excusal], self.contract, history=True)
        self.assertEqual(weekly_features(batch, 4).iloc[0].due_count, 1)
        batch.revisions["submissions"].loc[2, "recorded_at"] = batch.start + 27 * DAY
        self.assertEqual(weekly_features(batch, 4).iloc[0].due_count, 0)

    def test_backfilled_old_revision_does_not_replace_newer_source_state(self):
        revisions = self.batch.revisions["submissions"].copy()
        identity = revisions.id.iloc[0]
        rows = revisions.loc[revisions.id == identity].copy()
        if len(rows) < 2:
            rows = pd.concat([rows, rows], ignore_index=True)
            rows.loc[1, "revision"] = 2
            rows.loc[1, "effective_at"] += DAY
        rows.loc[rows.index[0], "recorded_at"] = self.batch.start + 100 * DAY
        expected = rows.sort_values(["effective_at", "revision"]).iloc[-1].revision
        self.assertEqual(as_of(rows, self.batch.publication).iloc[0].revision, expected)

    def test_label_precedence_unknown_and_minority_withdrawal(self):
        output = labels(self.batch, self.batch.start + 28 * DAY)
        self.assertTrue(set(output.label.dropna()) <= set(CLASSES))
        self.assertFalse(output.duplicated("enrollment_id").any())
        self.assertGreater((output.label == "withdrawn").sum(), 0)
        self.assertLess((output.label == "withdrawn").sum(), len(output) / 2)
        scores = self.batch.source("scores").set_index("enrollment_id")
        for row in output.loc[output.label == "fail"].itertuples():
            self.assertLess(scores.loc[row.enrollment_id].final_score, 60)
        invalid_quality = copy.deepcopy(self.batch)
        invalid_quality.qualification["qualified"] = False
        quality_labels = labels(invalid_quality, self.batch.start + 28 * DAY)
        self.assertFalse(quality_labels.label.isin(["fail", "pass", "excellent"]).any())
        late = copy.deepcopy(self.batch)
        wd = late.withdrawal_events.iloc[0].enrollment_id
        late.withdrawal_events.loc[late.withdrawal_events.enrollment_id == wd, "effective_at"] = late.start + DAY
        selected = labels(late, late.start + 28 * DAY).set_index("enrollment_id").loc[wd]
        self.assertTrue(pd.isna(selected.label))
        self.assertIn("before_cutoff", selected.label_status)

    def test_final_reconciled_totals_match_source_submissions(self):
        batch = self.batch
        e = batch.source("enrollments").set_index("id")
        s = batch.source("submissions")
        a = batch.source("assignments").set_index("id")
        for q in batch.qualification.loc[batch.qualification.qualified].itertuples():
            selected = s.loc[(s.user_id == e.loc[q.enrollment_id].user_id) & ~s.excused.fillna(False)]
            possible = selected.assignment_id.map(a.points_possible).sum()
            expected = selected.score.sum() * 100 / possible
            scores = batch.source("scores").set_index("enrollment_id")
            self.assertAlmostEqual(scores.loc[q.enrollment_id].final_score, expected, places=8)

    def test_current_term_hides_future_and_all_snapshots_keep_partition(self):
        current = generate_course(self.config, self.contract, 5, 1)
        validate_batch(current, self.dims, self.contract)
        self.assertEqual(len(current.qualification), 0)
        self.assertEqual(len(current.source("scores")), 0)
        with self.assertRaisesRegex(ValueError, "hidden truth"):
            labels(current, current.publication)
        with self.assertRaisesRegex(ValueError, "future cutoff"):
            weekly_features(current, 6)
        snapshots = pd.concat([weekly_features(self.batch, w) for w in (2, 4, 6)])
        self.assertTrue((snapshots.groupby("enrollment_id").partition.nunique() == 1).all())
        self.assertEqual(PARTITIONS, {1: "train", 2: "train", 3: "validation", 4: "previously_exposed_diagnostic", 5: "score"})

    def test_predictors_never_include_identifiers_or_simulator_truth(self):
        data = weekly_features(self.batch, 4)
        data["latent_preparation"] = 999.
        data["future_final_score"] = 100.
        data["label"] = "pass"
        self.assertEqual(list(predictors(data)), FEATURES)
        forbidden = {"enrollment_id", "user_id", "course_id", "term", "cutoff_week", "label",
                     "latent_preparation", "future_final_score", "name"}
        self.assertFalse(forbidden.intersection(predictors(data).columns))

