from dataclasses import dataclass
import hashlib
import uuid

import numpy as np
import pandas as pd

from aews.core.contract import HISTORY, as_of, frame, validate_table

DAY = pd.Timedelta(days=1)
ORIGIN = pd.Timestamp("2023-01-09", tz="UTC")
STATIC = ("accounts", "users", "enrollment_terms")


@dataclass
class CourseBatch:
    term: int
    course_id: int
    start: pd.Timestamp
    end: pd.Timestamp
    publication: pd.Timestamp
    revisions: dict
    withdrawal_events: pd.DataFrame
    qualification: pd.DataFrame

    def source(self, table, cutoff=None):
        return as_of(self.revisions[table], cutoff if cutoff is not None else self.publication)


def validate_config(config):
    size = config["profiles"][config["mode"]]
    if config["terms"] != 5 or config["term_days"] != 84:
        raise ValueError("This version requires five 84-day terms.")
    if not set(config["cutoff_weeks"]) <= {2, 4, 6} or 4 not in config["cutoff_weeks"]:
        raise ValueError("Week four is required; optional cutoffs are two and six.")
    if config["current_week"] not in config["cutoff_weeks"]:
        raise ValueError("Current week must be an enabled cutoff.")
    if not 1 <= size["courses_per_term"] <= 100 or not 1 <= size["assignments_per_course"] <= 30:
        raise ValueError("Course/assignment count exceeds bounded ID space.")
    if not 1 <= size["enrollments_per_course"] <= min(500, int(size["students"] * .8)):
        raise ValueError("Invalid enrollment sample or bounded course chunk.")
    if not 1 <= size["sections_per_course"] <= 9:
        raise ValueError("Section count exceeds bounded ID space.")
    if size["students"] > 5000:
        raise ValueError("This bounded build is bounded to 5,000 students.")
    return size


def dimensions(config, contract):
    size = validate_config(config)
    created = ORIGIN - 120 * DAY
    def dated(row, state):
        return dict(row, created_at=created, updated_at=created, workflow_state=state)
    users = [dated({"id": i, "name": f"Fictional Learner {i:05d}"}, "registered")
             for i in range(1, size["students"] + 1)]
    users += [dated({"id": size["students"] + i, "name": f"Fictional Instructor {i:03d}"}, "registered")
              for i in range(1, size["courses_per_term"] + 1)]
    terms = [dated({"id": t, "name": f"Synthetic Term {t}", "start_at": ORIGIN + (t - 1) * 120 * DAY,
                    "end_at": ORIGIN + ((t - 1) * 120 + 84) * DAY}, "active") for t in range(1, 6)]
    rows = {"accounts": [dated({"id": 1, "name": "Synthetic Learning Sandbox"}, "active")],
            "users": users, "enrollment_terms": terms}
    return {name: frame(name, records, contract) for name, records in rows.items()}


def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def generate_course(config, contract, term, course_number):
    size = validate_config(config)
    seed = config["seed"]
    rng = np.random.default_rng(np.random.SeedSequence([seed, term, course_number]))
    student_rng = np.random.default_rng(seed)
    # Stable learner-level preparation across terms; none of these arrays leaves
    # the generator or is placed in the source/feature contract.
    prep = student_rng.normal(0, 1, size["students"])
    base_engagement = student_rng.normal(0, .8, size["students"])
    start = ORIGIN + (term - 1) * 120 * DAY
    end = start + 84 * DAY
    publication = start + (7 * config["current_week"] if term == 5 else 114) * DAY
    created = start - 30 * DAY
    cid = term * 1000 + course_number
    rows = {name: [] for name in contract["tables"] if name not in STATIC}
    versions = {}

    def emit(table, row, event=None, recorded=None):
        event = event if event is not None else created
        recorded = recorded if recorded is not None else event + pd.Timedelta(hours=2)
        key = table, row["id"]
        versions[key] = versions.get(key, 0) + 1
        rows[table].append(dict(row, effective_at=event, recorded_at=recorded, revision=versions[key]))

    def dated(row, state="active", at=created):
        return dict(row, created_at=at, updated_at=at, workflow_state=state)

    course = dated({"id": cid, "name": f"Synthetic Course {course_number:03d}, Term {term}",
                    "account_id": 1, "enrollment_term_id": term, "start_at": start,
                    "conclude_at": end}, "available")
    emit("courses", course)
    for s in range(size["sections_per_course"]):
        emit("course_sections", dated({"id": cid * 10 + s, "course_id": cid, "enrollment_term_id": term,
             "name": f"Section {s + 1}", "start_at": start, "end_at": end}))
    emit("assignment_groups", dated({"id": cid, "name": "Point-weighted coursework",
                                    "context_id": cid, "context_type": "Course", "group_weight": 100.}, "available"))
    assignments = []
    for a in range(size["assignments_per_course"]):
        aid = cid * 100 + a
        due_day = 4 + (77 * a / max(1, size["assignments_per_course"] - 1))
        mode = "on_paper" if a % 9 == 3 else ("external_tool" if a % 11 == 5 else "online_upload")
        assignment = dated({"id": aid, "title": f"Synthetic assessment {a + 1}",
                            "context_id": cid, "context_type": "Course", "assignment_group_id": cid,
                            "due_at": start + due_day * DAY, "unlock_at": start + max(0, due_day - 14) * DAY,
                            "points_possible": float(rng.choice([50, 100, 150])),
                            "submission_types": [mode], "grading_type": "points", "omit_from_final_grade": False}, "published")
        emit("assignments", assignment)
        assignments.append(assignment)
    teacher_id = size["students"] + course_number
    teacher_enrollment = dated({"id": cid * 1000 + 999, "user_id": teacher_id, "course_id": cid,
                               "course_section_id": cid * 10, "type": "TeacherEnrollment",
                               "start_at": start, "end_at": end})
    emit("enrollments", teacher_enrollment)
    pool = int(size["students"] * .8) if term <= 2 else size["students"]
    learner_ids = rng.choice(np.arange(1, pool + 1), size["enrollments_per_course"], replace=False)
    difficulty = rng.normal(0, .5)
    withdrawals, qualifications = [], []
    event_serial = 0

    def log_event(uid, when):
        nonlocal event_serial
        event_serial += 1
        # A deterministic UUID per unique course request, not a learner key.
        log_id = str(uuid.UUID(bytes=hashlib.sha256(f"{seed}:{cid}:{event_serial}".encode()).digest()[:16]))
        row = {"id": log_id, "timestamp": when, "user_id": uid, "course_id": cid,
               "url": f"/courses/{cid}/assignments", "http_method": "GET", "http_status": 200,
               "http_version": "2.0", "remote_ip": f"192.0.2.{1 + event_serial % 253}",
               "interaction_micros": int(rng.integers(1000, 90000)), "participated": bool(rng.random() < .35)}
        emit("web_logs", row, when, when + rng.uniform(.02, 36) * pd.Timedelta(hours=1))

    for i, uid in enumerate(learner_ids):
        uid = int(uid)
        eid = cid * 1000 + i
        enrollment_day = int(rng.choice([0, 0, 0, 0, 0, 5, 12]))
        enrolled = start + enrollment_day * DAY
        enrollment = dated({"id": eid, "user_id": uid, "course_id": cid,
                             "course_section_id": cid * 10 + i % size["sections_per_course"],
                             "type": "StudentEnrollment", "start_at": enrolled, "end_at": None},
                            at=enrolled - DAY)
        emit("enrollments", enrollment, enrolled - DAY)
        shock_week = int(rng.integers(2, 10))
        shock = rng.exponential(.85) if rng.random() < .4 else 0.
        recovery = rng.uniform(.08, .45)
        innovation = rng.normal(0, .24, 12)
        engagement = np.empty(12)
        performance = np.empty(12)
        withdrawn_at = None
        for w in range(12):
            disruption = shock * np.exp(-recovery * max(0, w - shock_week)) if w >= shock_week else 0.
            engagement[w] = .9 * base_engagement[uid - 1] - disruption + innovation[w] + .02 * w
            performance[w] = 85 + 14 * prep[uid - 1] - 9 * difficulty + 4 * engagement[w] - 6 * disruption
            hazard = .004 + .026 * _sigmoid(-engagement[w] - .5 * prep[uid - 1] + disruption - 1.1)
            if withdrawn_at is None and w >= 1 and rng.random() < hazard:
                withdrawn_at = start + (w * 7 + rng.uniform(0, 7)) * DAY
            num_logs = int(rng.poisson(np.clip(3.8 + 2.1 * engagement[w], .15, 12)))
            for _ in range(num_logs):
                event = start + (w * 7 + rng.uniform(0, 7)) * DAY
                if event >= enrolled and (withdrawn_at is None or event < withdrawn_at):
                    log_event(uid, event)
        if withdrawn_at is not None:
            recorded = withdrawn_at + rng.uniform(.1, 8) * DAY
            withdrawals.append({"id": eid, "enrollment_id": eid, "effective_at": withdrawn_at,
                                "recorded_at": recorded, "event_type": "CONFIRMED_COURSE_WITHDRAWAL"})
            emit("enrollments", dict(enrollment, updated_at=recorded, end_at=withdrawn_at, workflow_state="inactive"),
                 withdrawn_at, recorded)

        earned, possible, graded_possible = 0., 0., 0.
        incomplete = bool(rng.random() < .035)
        for a, assignment in enumerate(assignments):
            aid = assignment["id"]
            sid = eid * 100 + a
            due = assignment["due_at"]
            w = min(11, int((due - start) / DAY) // 7)
            if due < enrolled:
                excused = True
            else:
                excused = rng.random() < .025
            paper = assignment["submission_types"][0] in ("on_paper", "external_tool")
            submission = dated({"id": sid, "user_id": uid, "assignment_id": aid, "submitted_at": None,
                                "graded_at": None, "score": None, "attempt": 0, "excused": False,
                                "submission_type": None if paper else "online_upload"}, "unsubmitted", created)
            emit("submissions", submission)
            if rng.random() < .09:
                override_time = due - rng.uniform(1, 5) * DAY
                override = dated({"id": sid, "assignment_id": aid, "set_type": "ADHOC", "set_id": None,
                                  "due_at": min(end - 2 * DAY, due + rng.uniform(2, 9) * DAY), "due_at_overridden": True,
                                  "title": "Synthetic approved individual extension"}, at=override_time)
                recorded = override_time + rng.uniform(.1, 4) * DAY
                emit("assignment_overrides", override, override_time, recorded)
                emit("assignment_override_students", dated({"id": sid, "user_id": uid, "assignment_id": aid,
                     "assignment_override_id": sid}, at=override_time), override_time, recorded)
                due = override["due_at"]
            if excused:
                at = max(enrolled, due - 3 * DAY)
                emit("submissions", dict(submission, excused=True, updated_at=at), at, at + DAY)
                continue
            points = assignment["points_possible"]
            possible += points
            completion_probability = .57 + .425 * _sigmoid(1.5 * engagement[w] + .6 * prep[uid - 1] + 1.8)
            completed = rng.random() < completion_probability and (withdrawn_at is None or due < withdrawn_at)
            if not completed:
                # Explicit instructor zero only after final reconciliation, not
                # a conversion of a null in-progress submission into failure.
                at = end + 7 * DAY
                if not incomplete:
                    emit("submissions", dict(submission, score=0., graded_at=at, updated_at=at, workflow_state="graded"),
                         at, at + DAY)
                    graded_possible += points
                continue
            delay = rng.normal(-.8, 1.8) + max(0, -engagement[w]) * 1.7
            submitted = min(end + 2 * DAY, max(enrolled, due + delay * DAY))
            grade_time = min(end + 12 * DAY, submitted + rng.uniform(1, 12) * DAY)
            grade = float(np.clip(performance[w] + rng.normal(0, 8), 5, 100) * points / 100)
            available_submission = dict(submission, submitted_at=None if paper else submitted,
                                        attempt=1, updated_at=submitted, workflow_state="unsubmitted" if paper else "submitted")
            if not paper:
                emit("submissions", available_submission, submitted, submitted + rng.uniform(.05, 3) * DAY)
            if not (incomplete and a >= len(assignments) - 3):
                graded = dict(available_submission, score=grade, graded_at=grade_time,
                              updated_at=grade_time, workflow_state="graded")
                emit("submissions", graded, grade_time, grade_time + rng.uniform(.05, 2) * DAY)
                graded_possible += points
                if not paper and rng.random() < .1:
                    retry = min(end + 14 * DAY, grade_time + rng.uniform(1, 4) * DAY)
                    retry_grade = min(end + 18 * DAY, retry + rng.uniform(1, 4) * DAY)
                    grade = float(np.clip(grade + rng.normal(4, 5) * points / 100, 0, points))
                    emit("submissions", dict(available_submission, attempt=2, submitted_at=retry,
                         updated_at=retry, workflow_state="submitted"), retry, retry + .1 * DAY)
                    emit("submissions", dict(graded, score=grade, attempt=2, submitted_at=retry,
                         graded_at=retry_grade, updated_at=retry_grade), retry_grade, retry_grade + .1 * DAY)
                earned += grade
        final_score = 100 * earned / possible if possible else None
        current_score = 100 * earned / graded_possible if graded_possible else None
        score_time = end + 20 * DAY
        emit("scores", {"id": eid, "enrollment_id": eid, "course_score": True,
                       "current_score": current_score, "final_score": final_score, "updated_at": score_time},
             score_time, score_time + DAY)
        qualifications.append({"enrollment_id": eid, "effective_at": end + 21 * DAY, "recorded_at": end + 22 * DAY,
                               "qualified": bool(not incomplete and possible > 0),
                               "reason": "final_reconciliation_complete" if not incomplete else "incomplete_gradebook",
                               "course_finished_at": end})
        if withdrawn_at is None:
            emit("enrollments", dict(enrollment, updated_at=end, end_at=end, workflow_state="completed"), end, end + DAY)
    emit("courses", dict(course, updated_at=end, workflow_state="completed"), end, end + DAY)
    # Logs legitimately include requests without a local enrolled learner.
    log_event(None, start + DAY)
    log_event(size["students"] + size["courses_per_term"] + 100000, start + 2 * DAY)
    revision_frames = {name: frame(name, records, contract, history=True) for name, records in rows.items()}
    if term == 5:
        revision_frames = {name: data.loc[(data.effective_at <= publication) & (data.recorded_at <= publication)].reset_index(drop=True)
                           for name, data in revision_frames.items()}
    wd = pd.DataFrame(withdrawals, columns=["id", "enrollment_id", "effective_at", "recorded_at", "event_type"])
    quality = pd.DataFrame(qualifications)
    for data in (wd, quality):
        for name in ("effective_at", "recorded_at"):
            data[name] = pd.to_datetime(data[name], utc=True).dt.floor("us")
    if term == 5:
        wd = wd.loc[(wd.effective_at <= publication) & (wd.recorded_at <= publication)].reset_index(drop=True)
        quality = quality.iloc[:0].copy()
    return CourseBatch(term, cid, start, end, publication, revision_frames, wd, quality)


def validate_batch(batch, dims, contract):
    sources = {name: batch.source(name).drop(columns=list(HISTORY)) for name in batch.revisions}
    references = dict(dims, **sources)
    counts = {}
    for name, data in batch.revisions.items():
        validate_table(name, data, contract, references, history=True)
        counts[name] = validate_table(name, sources[name], contract, references)
    enrollments = sources["enrollments"]
    if enrollments.duplicated(["user_id", "course_id", "type"]).any():
        raise ValueError("Duplicate learner-course enrollment")
    sections = sources["course_sections"].set_index("id")
    if not enrollments.course_section_id.map(sections.course_id).eq(enrollments.course_id).all():
        raise ValueError("Enrollment section belongs to a different course")
    assignments = sources["assignments"].set_index("id")
    sub = batch.revisions["submissions"]
    if not sub.user_id.isin(enrollments.user_id).all():
        raise ValueError("Submission user is not enrolled")
    if (sub.score.dropna() < 0).any() or (sub.score > sub.assignment_id.map(assignments.points_possible)).any():
        raise ValueError("Submission score outside point range")
    if ((sub.graded_at < sub.submitted_at) & sub.graded_at.notna() & sub.submitted_at.notna()).any():
        raise ValueError("Grade precedes submission")
    if not batch.withdrawal_events.enrollment_id.isin(enrollments.id).all():
        raise ValueError("Orphan supplemental withdrawal")
    if batch.withdrawal_events.duplicated("enrollment_id").any():
        raise ValueError("Duplicate confirmed withdrawal")
    if batch.term == 5:
        if len(batch.qualification) or len(sources["scores"]):
            raise ValueError("Current-term future label contamination")
        if any((d.recorded_at > batch.publication).any() or (d.effective_at > batch.publication).any()
               for d in batch.revisions.values()):
            raise ValueError("Current-term future source contamination")
    return counts
