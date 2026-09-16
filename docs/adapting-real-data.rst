Adapting to institutional data
==============================

Important boundary
------------------

This repository implements synthetic generation, not a Canvas API, CD2 export
or operational data store connector. ``prepare()`` calls the simulator
directly. Changing workspace names, profile settings or credentials does not
read real student records. Do not deploy synthetic model weights as an
institutional decision system.

Required adapter and policy work
--------------------------------

1. Establish institutional authorization, privacy boundaries, retention,
   least-privilege access, responsible-use review and permitted human action.
   Do not place real records, credentials or exported model evidence into a
   public copy of this repository.

2. Implement an explicit CD2/ODS ingestion and normalization adapter. Map to
   the agreed selected projection, preserving source keys, types, enum
   semantics, revisions and availability timestamps. Keep raw provenance and
   validation failures; never silently coerce unknown records to a supported
   state. The public contract here is a selected projection, not all Canvas
   tables or an assurance of compatibility with every institutional export.

3. Obtain institution-approved authoritative course outcomes, withdrawal
   event dates and gradebook qualification. ``sim_support`` is supplemental
   synthetic evidence, not a claim that these tables exist in Canvas.
   A confirmed course withdrawal is not university departure. Inactivity,
   low grades, enrollment deletion and ambiguous workflow states must not
   be reinterpreted as withdrawal without approved semantics.

4. Replace demonstration-specific rules: five term IDs, fixed dates,
   84-day terms, week-four checkpoints, grade thresholds 60/90, finalization
   timing, course aggregate scores, grading/weighting and known-outcome
   qualification. Resolve section/course/term joins, repeated enrollments,
   enrollment start dates and label availability for the real institution.

5. Reconstruct point-in-time observations using **both**
   ``effective_at <= cutoff`` and ``recorded_at <= cutoff``. A late-recorded
   historical revision must not appear earlier than it was available, and
   an old backfill must not displace a newer effective state. Latest-state
   snapshots alone cannot establish this history. If history is unavailable,
   disclose unavailable features or implement prospective capture first.
   Canvas web-log DAP retention is 30 days; longitudinal reconstruction may
   require an existing institution-approved archive.

6. Preserve source semantics: individual assignment overrides affect due
   dates only when available; excusals affect eligible denominators only
   when known. A null submitted timestamp for paper/external work is not
   automatically missing work. Anonymous/external web-log users are counted,
   not fabricated as enrolled learners. Foreign-key validation must respect
   these documented exceptions rather than deleting inconvenient records.

7. Emit the agreed historical feature/label and current feature-only
   contracts through an explicit real-data entry point. Keep unknown final
   outcomes unlabeled. Exclude identifiers, names, protected traits, latent
   simulation state and all future outcomes from predictors. Preserve
   enrollment/course grain and ensure an enrollment cannot cross splits.

8. Reserve an actually untouched later cohort before exploration. The
   supplied synthetic term four is already known from prototype work and
   cannot become independent evidence by renaming it. Evaluate returning
   versus unseen learners, different courses and time shifts. Course
   offering IDs changing every term does not prove curriculum generalization.

9. Fit imputation, missing indicators, scaling and candidates only on
   training data. Keep calibration separate from model-choice data. Freeze
   the choice before untouched evaluation. Compare class-specific
   precision/recall, average precision against prevalence, calibration,
   false alarms, review workload and simpler institutional/grade-only rules.
   The package's prevalence comparison is not a completed operational
   acceptance study.

10. Agree on acceptance criteria, fair access to support, subgroup evaluation,
    human review, contestability and monitoring before operational use.
    Establish data drift, calibration drift, retraining, retention, audit and
    rollback policies. No accepted-model registration, contact workflow or
    autonomous academic action is implemented here.

Explanation boundary
--------------------

The exact contribution method is specific to the fitted logistic pipeline.
If the model family changes, implement and validate an appropriate explanation
method rather than reusing the linear formula or relabeling it as generic SHAP.
Neither a coefficient nor an explanation establishes causation, feature
significance, or that intervening on a feature will change a student's outcome.
