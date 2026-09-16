import numpy as np
import pandas as pd

from aews.core.simulation import DAY

FEATURE_VERSION = "past-only-weekly-1"
FEATURES = [
    "days_enrolled", "due_count", "completion_ratio", "missing_online_ratio",
    "late_ratio", "paper_unknown_ratio", "excused_ratio", "graded_ratio",
    "grade_mean", "grade_recent_mean", "grade_trend", "reattempt_ratio",
    "log_count", "recent_log_count", "activity_change", "days_since_activity",
]
CLASSES = ["excellent", "fail", "pass", "withdrawn"]
PARTITIONS = {1: "train", 2: "train", 3: "validation", 4: "previously_exposed_diagnostic", 5: "score"}


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else np.nan


def weekly_features(batch, week):
    cutoff = batch.start + 7 * week * DAY
    if batch.term == 5 and cutoff > batch.publication:
        raise ValueError("Cannot score an unpublished future cutoff")
    e = batch.source("enrollments", cutoff)
    e = e.loc[(e.type == "StudentEnrollment") & (e.workflow_state == "active") &
              (e.start_at.isna() | (e.start_at <= cutoff))].copy()
    known_wd = batch.withdrawal_events
    known_wd = known_wd.loc[(known_wd.effective_at <= cutoff) & (known_wd.recorded_at <= cutoff)]
    e = e.loc[~e.id.isin(known_wd.enrollment_id)]
    a = batch.source("assignments", cutoff)
    a = a.loc[(a.context_type == "Course") & (a.context_id == batch.course_id) &
              (a.workflow_state == "published") & (~a.omit_from_final_grade) &
              (a.unlock_at.isna() | (a.unlock_at <= cutoff))]
    a = a.rename(columns={"id": "assignment_id"})
    s = batch.source("submissions", cutoff)
    s = s[["user_id", "assignment_id", "submitted_at", "graded_at", "score", "attempt", "excused"]]
    overrides = batch.source("assignment_overrides", cutoff)
    mapping = batch.source("assignment_override_students", cutoff)
    overrides = overrides.loc[(overrides.workflow_state == "active") & (overrides.set_type == "ADHOC") &
                              overrides.due_at_overridden]
    mapping = mapping.loc[mapping.workflow_state == "active"]
    extension = mapping[["user_id", "assignment_id", "assignment_override_id"]].merge(
        overrides[["id", "due_at"]], left_on="assignment_override_id", right_on="id", how="inner")
    if extension.duplicated(["user_id", "assignment_id"]).any():
        raise ValueError("Ambiguous overlapping individual overrides")
    extension = extension[["user_id", "assignment_id", "due_at"]].rename(columns={"due_at": "override_due"})
    work = e[["id", "user_id", "start_at"]].merge(
        a[["assignment_id", "due_at", "points_possible", "submission_types"]], how="cross")
    work = work.merge(s, on=["user_id", "assignment_id"], how="left", validate="one_to_one")
    work = work.merge(extension, on=["user_id", "assignment_id"], how="left", validate="one_to_one")
    work["effective_due"] = work.override_due.combine_first(work.due_at)
    work["due"] = (work.effective_due <= cutoff) & (work.effective_due >= work.start_at)
    work["excused"] = work.excused.fillna(False).astype(bool)
    work["paper"] = work.submission_types.map(lambda x: bool(set(x) & {"on_paper", "external_tool"}))
    work["submitted"] = work.submitted_at.notna() & (work.submitted_at <= cutoff)
    work["graded"] = work.graded_at.notna() & (work.graded_at <= cutoff) & work.score.notna()
    work["normalized_grade"] = work.score * 100 / work.points_possible
    logs = batch.source("web_logs", cutoff)
    logs = logs.loc[(logs.timestamp >= batch.start) & (logs.timestamp <= cutoff) & (logs.course_id == batch.course_id)]
    rows = []
    work_groups = dict(tuple(work.groupby("id", sort=False)))
    log_groups = dict(tuple(logs.groupby("user_id", sort=False)))
    for enrollment in e.itertuples(index=False):
        items = work_groups.get(enrollment.id, work.iloc[:0])
        due_all = items.loc[items.due]
        due = due_all.loc[~due_all.excused]
        online = due.loc[~due.paper]
        known_grade = due.loc[due.graded & (due.points_possible > 0)]
        recent_grade = known_grade.loc[known_grade.graded_at > cutoff - 14 * DAY]
        prior_grade = known_grade.loc[known_grade.graded_at <= cutoff - 14 * DAY]
        observed_logs = log_groups.get(enrollment.user_id, logs.iloc[:0])
        observed_logs = observed_logs.loc[observed_logs.timestamp >= enrollment.start_at]
        recent_count = int((observed_logs.timestamp > cutoff - 14 * DAY).sum())
        prior_count = int(((observed_logs.timestamp > cutoff - 28 * DAY) &
                           (observed_logs.timestamp <= cutoff - 14 * DAY)).sum())
        complete = due.submitted | (due.paper & due.graded & due.score.gt(0))
        last_log = observed_logs.timestamp.max()
        window_start = max(batch.start, enrollment.start_at)
        row = {
            "enrollment_id": int(enrollment.id), "user_id": int(enrollment.user_id),
            "course_id": batch.course_id, "term": batch.term, "cutoff_week": week,
            "as_of_time": cutoff, "observation_start": window_start, "observation_end": cutoff,
            "feature_version": FEATURE_VERSION, "partition": PARTITIONS[batch.term],
            "days_enrolled": float((cutoff - window_start) / DAY),
            "due_count": float(len(due)), "completion_ratio": ratio(complete.sum(), len(due)),
            "missing_online_ratio": ratio((~online.submitted).sum(), len(online)),
            "late_ratio": ratio((online.submitted & (online.submitted_at > online.effective_due)).sum(),
                                online.submitted.sum()),
            "paper_unknown_ratio": ratio((due.paper & ~due.graded).sum(), len(due)),
            "excused_ratio": ratio(due_all.excused.sum(), len(due_all)),
            "graded_ratio": ratio(len(known_grade), len(due)),
            "grade_mean": ratio(known_grade.score.sum() * 100, known_grade.points_possible.sum()),
            "grade_recent_mean": ratio(recent_grade.score.sum() * 100, recent_grade.points_possible.sum()),
            "grade_trend": (ratio(recent_grade.score.sum() * 100, recent_grade.points_possible.sum()) -
                            ratio(prior_grade.score.sum() * 100, prior_grade.points_possible.sum())),
            "reattempt_ratio": ratio(due.attempt.gt(1).sum(), len(due)),
            "log_count": float(len(observed_logs)), "recent_log_count": float(recent_count),
            "activity_change": float(recent_count - prior_count),
            "days_since_activity": float((cutoff - last_log) / DAY) if pd.notna(last_log) else float((cutoff - window_start) / DAY + 1),
            "data_quality_flag": "no_due_work" if not len(due) else ("no_known_grades" if not len(known_grade) else "usable"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def labels(batch, cutoff):
    """Read outcome-side future only for finalized historical terms."""
    if batch.term == 5:
        raise ValueError("Term-five hidden truth is never exposed to labels")
    enrollments = batch.source("enrollments")
    scores = batch.source("scores")
    scores = scores.loc[scores.course_score]
    if scores.duplicated("enrollment_id").any():
        raise ValueError("Multiple aggregate course scores for one enrollment")
    score_map = scores.set_index("enrollment_id").final_score.to_dict()
    q = batch.qualification
    q = q.loc[(q.recorded_at <= batch.publication) & (q.effective_at <= batch.publication)]
    quality = q.set_index("enrollment_id").to_dict("index")
    wd = batch.withdrawal_events
    wd = wd.loc[(wd.recorded_at <= batch.publication) & (wd.effective_at <= batch.end)]
    wd_map = wd.set_index("enrollment_id").effective_at.to_dict()
    result = []
    for e in enrollments.loc[enrollments.type == "StudentEnrollment"].itertuples(index=False):
        outcome, reason = None, None
        effective = wd_map.get(e.id)
        if effective is not None:
            if effective <= cutoff:
                reason = "withdrawn_before_cutoff_including_late_recording"
            else:
                outcome = "withdrawn"
        elif e.id not in quality or not quality[e.id]["qualified"]:
            reason = "incomplete_or_unqualified_gradebook"
        elif quality[e.id]["course_finished_at"] > batch.publication:
            reason = "course_not_finished"
        elif pd.isna(score_map.get(e.id)):
            reason = "unknown_final_grade"
        else:
            grade = score_map[e.id]
            outcome = "fail" if grade < 60 else ("excellent" if grade >= 90 else "pass")
        result.append({"enrollment_id": int(e.id), "label": outcome, "label_status": reason or "known",
                       "label_available_at": batch.publication})
    return pd.DataFrame(result)


def feature_contract():
    return {
        "version": FEATURE_VERSION, "predictors": FEATURES,
        "units": {"grade_mean": "percent", "grade_recent_mean": "percent", "grade_trend": "percentage_point_difference",
                  "days_enrolled": "days", "days_since_activity": "days", "activity_change": "request_count_difference"},
        "entity": "enrollment_id + cutoff_week; course weeks relative to course start",
        "availability": "Both effective_at and recorded_at <= cutoff; no latest-state historical filtering",
        "missing_values": "NaN denotes unknown or zero eligible denominator, train-only median imputation plus missing indicators",
        "deadline": "Known active individual extension supersedes known base due_at; only due work after enrollment start",
        "paper": "Ungraded paper/external work is unknown, never inferred missing from null submitted_at",
        "prohibited_predictors": ["names", "identifiers", "protected_traits", "latent_state", "final_score", "future_withdrawal", "label"],
        "current_truth": "Not exposed or persisted; no current labels",
        "partitions": PARTITIONS, "classes": CLASSES,
        "warning": "Four course outcomes; synthetic correlations do not demonstrate institutional validity or causality",
    }
