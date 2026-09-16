Academic Early Warning System
=============================

A self-contained, synthetic-data starter for explaining an early-warning
method at student-course enrollment grain. It is not a production system,
an approved Microsoft offering, or an institutional accuracy claim.

.. image:: assets/academic-early-warning-system.svg
   :alt: Academic Early Warning System architecture
   :target: assets/academic-early-warning-system.svg

Presentation assets: `vector diagram <assets/academic-early-warning-system.svg>`_,
`presentation image <assets/academic-early-warning-system.png>`_ and
`editable diagram <assets/academic-early-warning-system.excalidraw>`_.

Start here: upload one folder, run one notebook
----------------------------------------------

.. important::

   **Do not use Load to Tables for the replication Quickstart.**
   Extract the simulated-data ZIP on your computer, upload the whole extracted
   folder to Lakehouse **Files** using **Upload folder**, then run
   ``00_load_lakehouse_tables.ipynb``. It creates all **27 input tables**.
   There is no per-table wizard, file-type selection or table naming to do.

The intended sequence is::

    Extract repository ZIP, then extract its simulated-data ZIP
        -> Files: Upload folder, keeping all files and nesting
        -> 00: Run all to create or verify the 27 input tables
        -> Optional 01: validate, then authorize model training and publication
        -> Optional 02: explore that completed model publication

Already uploaded the complete extracted folder? **Use that folder; do not
upload the ZIP again or delete existing data.** Set ``SOURCE_FOLDER`` to its
exact Lakehouse path. The loader verifies and skips identical existing tables.
If an existing table differs, it stops before creating any new tables so you
can review the conflict without an automatic replacement.

Quickstart: replicate with the included dataset
-----------------------------------------------

**Everything needed for the synthetic input is included.** You do not need to
generate data or install Python locally to use the prebuilt notebooks with the
default resource names. You do need access to Microsoft Fabric, a capacity
with available quota, and permission to create a workspace and Lakehouse.
Capacity and storage charges are your responsibility.

1. Download or clone the complete repository. If you downloaded a ZIP, use
   **Extract All** on your computer and open the extracted repository folder.
   You should see ``README.rst``, ``notebooks`` and ``sample-data``.
   In ``sample-data``, also use **Extract All** on
   ``academic-early-warning-simulated-data.zip``. Open the resulting data
   folder and confirm it directly contains ``source_parquet``, ``features``,
   ``manifest.json`` and ``summary.json``.

2. In your approved Fabric environment, create or select a capacity and create
   a workspace named **Academic Early Warning System**, assigned to that
   capacity. Create a Lakehouse named **academic_early_warning**, with
   **Lakehouse schemas enabled**. Set its workspace Spark runtime to
   **1.3 (Spark 3.5, Delta 3.2)**.

3. Open that Lakehouse. Under **Files**, choose **Upload folder** and select
   the whole extracted data folder. Keep all files, filenames and nested
   folders. Confirm there is **only one dataset root**, not a second copy of
   the root folder inside it. The expected Lakehouse layout is::

       Files/
           academic-early-warning-simulated-data/
               source_parquet/
                   canvas/
                   canvas_logs/
                   sim_history/
                   sim_support/
               features/
                   historical.parquet
                   current.parquet
               manifest.json
               summary.json

4. Import ``notebooks/00_load_lakehouse_tables.ipynb`` into the workspace.
   Attach **academic_early_warning** as its **default Lakehouse**.
   In its configuration cell (block 3), check::

       SOURCE_FOLDER = 'Files/academic-early-warning-simulated-data'

   If your uploaded folder has a different name, set the exact path, for
   example ``Files/academic-early-warning-simulated-data-fixed``. Do not
   point at ``source_parquet`` or ``features`` alone. No tenant IDs,
   credentials or overwrite acknowledgment are needed.

5. Choose **Run all** once. The loader validates the existing Runtime 1.3
   session, exact workspace and Lakehouse, trusted release manifest hash,
   file bytes/checksums, types and row counts. It creates **25 source,
   history and support tables plus two silver feature tables**. Each table
   reports ``created`` or ``already_matches`` and its verified Spark row
   count. ``tables_loaded`` appears only when all 27 are complete.

   You can stop here if you only need the tables. No files or tables are
   deleted or overwritten, and no model, predictions or ``gold`` placeholders
   are created. Reruns verify and skip matching tables. Use one writer;
   a failed run can leave newly created tables for the next verified rerun.

6. **Optional model and explanations:** follow the section below using the
   same uploaded folder. No regeneration or second upload is needed.

7. Stop notebook sessions and pause idle capacity according to your
   organization's policy. This repository does not perform those operations
   automatically. Keep your input files and any publication archives.

**Using other resource names?** Edit ``intended_workspace``,
``intended_lakehouse`` and ``experiment_name`` in
``config/simulation.json``, then rebuild all three notebooks with
``python tools\build_notebooks.py`` before importing. Alternatively, edit
those three ``SOURCE_CONFIG`` entries consistently in **every notebook you use**
after their configuration cell initializes them. Do not bypass the guards.

Optional: train, publish and explain
------------------------------------

1. Import ``01_build_and_publish.ipynb`` and ``02_explore_and_explain.ipynb``.
   Attach the same default Lakehouse in the configured workspace.
   In notebook 01 keep::

       INPUT_MODE = 'folder'
       SOURCE_FOLDER = 'Files/academic-early-warning-simulated-data'
       RUN_MODE = 'validate'
       ALLOW_DEMO_TABLE_REPLACEMENT = False

   Set ``SOURCE_FOLDER`` to the same exact root used in notebook 00.
   Run all. Validation checks the environment, all included files, source
   schema/counts/keys and prepared feature boundaries without writes or training.
   Proceed only after ``dataset: verified`` and ``validation: passed``.

2. Review `Safety and replacement scope`_. In this synthetic development
   Lakehouse only, change ``RUN_MODE = 'publish'`` and
   ``ALLOW_DEMO_TABLE_REPLACEMENT = True``. Run all again. Notebook 01 uses
   the folder's prepared features, trains and calibrates one four-class model,
   and publishes results. It does not regenerate the input data.
   **It still archives and verifies all existing named input tables before
   replacing them**, including the 27 tables imported by notebook 00, along
   with existing named ``gold`` outputs. This is not the non-overwriting
   table-only operation performed by notebook 00.

3. Only after ``publication_completed``, run notebook 02. A completed
   ``tables_loaded`` import alone is **not enough** for exploration. Notebook
   02 reads the model publication, its current predictions and explanations.
   For the included small profile, the input has 583 current feature rows.
   Block 6 automatically fills and prints ``ENROLLMENT_TO_EXPLAIN`` and
   ``OUTCOME_TO_EXPLAIN`` using the first current enrollment and its actual
   predicted outcome. To inspect a different enrollment, edit its assignment
   in block 6 and rerun blocks 6 and 7.

ZIP upload alternative
----------------------

If you prefer to upload one compressed file, keep the inner simulated-data ZIP
compressed. In Lakehouse **Files/uploads**, choose **Upload files** and upload
the original ``sample-data/academic-early-warning-simulated-data.zip`` unchanged::

    Files/
        uploads/
            academic-early-warning-simulated-data.zip

Skip notebook 00, which reads extracted folders. In notebook 01 set
``INPUT_MODE = 'uploaded'``. Its configured ``UPLOAD_PATH`` then selects
``Files/uploads/academic-early-warning-simulated-data.zip`` from ``BUNDLE_INFO``.
Run validation, review the same replacement scope, authorize publication and
then use notebook 02 as above. Notebook 01 verifies and unpacks the ZIP for
this alternative. Do not recompress it or use the outer repository ZIP.

Included data
-------------

``sample-data/academic-early-warning-simulated-data.zip`` is approximately
15.6 MB compressed (24.6 MB expanded). It contains 117 files: selected
source-shaped records, immutable revisions, supplemental outcome events,
historical/current feature Parquet files, checksums and generation summary.
No pretrained model, prediction results or real student records are included.

=============================== =========
Included synthetic data         Rows
=============================== =========
Learners / instructors          400 / 8
Course offerings / terms        40 / 5
Student enrollments             3,000
Submissions                     60,000
Activity records                111,217
Historical feature snapshots    2,336
Current feature snapshots       583
=============================== =========

The feature tables were computed from the bundled histories during dataset
preparation, not during upload. Training and predictions are computed in your
environment. Current-term feature files contain no future labels. See
``sample-data/README.txt`` and ``sample-data/bundle-info.json`` for provenance,
exact hashes and counts. These files must stay paired with the notebooks.

Parquet file-to-table names
---------------------------

**These are destination table names, not instructions to rename the files.**
Keep the extracted filenames and table folders unchanged. Paths below are
relative to the extracted simulated-data folder. The 115 Parquet files map
to **27 input tables**; each row below is one table.

Notebook 00 applies this mapping automatically in the schema-enabled
Lakehouse. For example, ``features\current.parquet`` becomes
``silver.current_weekly_features``, not a table in ``dbo``. The columns below
are a reference for the destination schema and table name, not wizard steps.

``part-*.parquet`` means **all five files, part-01.parquet through
part-05.parquet, in that one table folder**. The loader combines those parts
into the named table. Do not create a table per part, flatten files from
different folders, or overwrite the table
repeatedly with individual parts. The three ``part-00.parquet`` rows below
each contain just one file.

.. list-table:: Parquet files and destination Delta tables
   :header-rows: 1
   :widths: 60 15 25

   * - Parquet file(s) to load together
     - Schema
     - New table name
   * - ``features\current.parquet``
     - ``silver``
     - ``current_weekly_features``
   * - ``features\historical.parquet``
     - ``silver``
     - ``historical_weekly_features``
   * - ``source_parquet\canvas\accounts\part-00.parquet``
     - ``canvas``
     - ``accounts``
   * - ``source_parquet\canvas\assignment_groups\part-*.parquet``
     - ``canvas``
     - ``assignment_groups``
   * - ``source_parquet\canvas\assignment_override_students\part-*.parquet``
     - ``canvas``
     - ``assignment_override_students``
   * - ``source_parquet\canvas\assignment_overrides\part-*.parquet``
     - ``canvas``
     - ``assignment_overrides``
   * - ``source_parquet\canvas\assignments\part-*.parquet``
     - ``canvas``
     - ``assignments``
   * - ``source_parquet\canvas\course_sections\part-*.parquet``
     - ``canvas``
     - ``course_sections``
   * - ``source_parquet\canvas\courses\part-*.parquet``
     - ``canvas``
     - ``courses``
   * - ``source_parquet\canvas\enrollment_terms\part-00.parquet``
     - ``canvas``
     - ``enrollment_terms``
   * - ``source_parquet\canvas\enrollments\part-*.parquet``
     - ``canvas``
     - ``enrollments``
   * - ``source_parquet\canvas\scores\part-*.parquet``
     - ``canvas``
     - ``scores``
   * - ``source_parquet\canvas\submissions\part-*.parquet``
     - ``canvas``
     - ``submissions``
   * - ``source_parquet\canvas\users\part-00.parquet``
     - ``canvas``
     - ``users``
   * - ``source_parquet\canvas_logs\web_logs\part-*.parquet``
     - ``canvas_logs``
     - ``web_logs``
   * - ``source_parquet\sim_history\assignment_groups_revisions\part-*.parquet``
     - ``sim_history``
     - ``assignment_groups_revisions``
   * - ``source_parquet\sim_history\assignment_override_students_revisions\part-*.parquet``
     - ``sim_history``
     - ``assignment_override_students_revisions``
   * - ``source_parquet\sim_history\assignment_overrides_revisions\part-*.parquet``
     - ``sim_history``
     - ``assignment_overrides_revisions``
   * - ``source_parquet\sim_history\assignments_revisions\part-*.parquet``
     - ``sim_history``
     - ``assignments_revisions``
   * - ``source_parquet\sim_history\course_sections_revisions\part-*.parquet``
     - ``sim_history``
     - ``course_sections_revisions``
   * - ``source_parquet\sim_history\courses_revisions\part-*.parquet``
     - ``sim_history``
     - ``courses_revisions``
   * - ``source_parquet\sim_history\enrollments_revisions\part-*.parquet``
     - ``sim_history``
     - ``enrollments_revisions``
   * - ``source_parquet\sim_history\scores_revisions\part-*.parquet``
     - ``sim_history``
     - ``scores_revisions``
   * - ``source_parquet\sim_history\submissions_revisions\part-*.parquet``
     - ``sim_history``
     - ``submissions_revisions``
   * - ``source_parquet\sim_history\web_logs_revisions\part-*.parquet``
     - ``sim_history``
     - ``web_logs_revisions``
   * - ``source_parquet\sim_support\course_withdrawal_events\part-*.parquet``
     - ``sim_support``
     - ``course_withdrawal_events``
   * - ``source_parquet\sim_support\gradebook_qualification\part-*.parquet``
     - ``sim_support``
     - ``gradebook_qualification``

``manifest.json``, ``summary.json`` and ``bundle-info.json`` are metadata,
not tables. There are no input Parquet files for
``gold.student_course_predictions``, ``gold.prediction_explanations``,
``gold.model_evaluation`` or ``gold.data_quality_summary``. Notebook 01
creates those outputs after training and scoring; do not create empty tables
for them. Correct input tables alone do not replace the model publication
workflow. Folder mode uses the same uploaded data without a ZIP upload.

What it demonstrates
--------------------

1. Load the included correlated Canvas-shaped synthetic history, or generate
   a larger population with the fixed seed.
2. Reconstruct what was observable at week four, using both effective time
   and recorded/available time. Join later outcomes only to historical rows.
3. Fit training-only transformations, calibrate and select one logistic
   classifier that returns four probabilities:

   * Excellent: qualified final course grade at least 90.
   * Pass: qualified final grade from 60 to below 90.
   * Fail: qualified final grade below 60.
   * Withdrawn: an explicit course-withdrawal event after the checkpoint.
     This takes precedence over a passing/failing grade.

   Unknown or unqualified outcomes are not silently treated as Fail.
   These thresholds and time rules are illustrative, not institutional policy.

4. Score current enrollments without future labels. Explain each predicted
   class and global per-outcome feature contributions.

No separate withdrawal workflow, student outreach, automatic acceptance or
accepted-model registration is implemented. Real Canvas ingestion is also
**not implemented**: changing configuration or credentials does not connect
real data. An explicit CD2/ODS adapter and institution-approved history/label
mapping are required. See `adapting real data <docs/adapting-real-data.rst>`_.

Three notebooks, one code base
------------------------------

``notebooks/00_load_lakehouse_tables.ipynb``
    Self-contained table-only import from the uploaded extracted folder.
    Creates or verifies all 27 input tables without overwrites or training.

``notebooks/01_build_and_publish.ipynb``
    Self-contained folder/ZIP data loading or optional source generation,
    modeling and guarded publication. Default mode validates the attached
    environment and included data without table changes.

``notebooks/02_explore_and_explain.ipynb``
    Read-only inspection of the complete publication, development evaluation,
    global training contributions and selectable enrollment explanations.
    It verifies that the saved model reproduces the published probabilities.

The workflow is: same source code -> build notebooks -> import into Fabric ->
attach the intended Lakehouse -> load folder tables -> optionally authorize
model publication -> explore. The dataset is supplied; no downloaded model or
external fixture is needed. Rebuild and reimport
all notebooks after source or configuration changes.

This is a portable implementation of the method, not a distribution of the
original demonstration's trained weights. Fresh runs create their own model
and publication identifiers. The current four-outcome selection protocol is
documented below; do not assume byte-identical scores to another deployment.

Local quickstart (PowerShell, Python 3.11)
----------------------------------------

Run these commands from this repository's root::

    py -3.11 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe tools\build_notebooks.py
    .\.venv\Scripts\python.exe -m unittest discover -s tests -v
    .\.venv\Scripts\python.exe tools\validate_included_data.py
    .\.venv\Scripts\python.exe tools\scan_package.py

This optional local route loads the included archive, validates it, and trains
and scores locally without generating data. It validates source keys, types,
enums, nullability, references, feature boundaries, training splits, probabilities
and exact explanation arithmetic. Temporary extraction is removed on
completion. It does not invoke Spark, authenticate,
call a cloud API, train in Fabric or log to a remote MLflow experiment.

To generate new data locally instead, run
``python tools\run_local.py --profile small``. Each generator run creates a
fresh ignored ``outputs/small/<timestamp>/`` directory.
Read ``validation-summary.json`` for actual counts. The supplied small
profile has 400 learners, 8 instructors, 40 courses, 3,000 student enrollments
(3,040 including teachers), and 60,000 submissions. Current predictions can
be fewer than enrollments because already-withdrawn learners are not scored.
The full profile has 5,000 learners, 80 instructors, 400 courses, 60,000
student enrollments and 1.2 million submissions. Generated data, model binaries
and validation output are not distributed in this repository, except for
the explicitly included synthetic input ZIP.

Use ``--features-only`` to skip raw Parquet staging, or ``--profile full`` for
a larger local run. Neither option changes the simulator's label rules.
Local runs do not persist a serialized fitted model; rebuild from source.

Fabric manual setup
-------------------

Use a separate, institution-approved development environment. Manually create
or select capacity and a workspace, assign the workspace to capacity, then
create a **schema-enabled** Lakehouse. Use Runtime 1.3:
Python 3.11, Spark 3.5 and Delta 3.2. The compatibility requirements are
NumPy 1.26.4, pandas 2.1.4, scikit-learn 1.2.2, PyArrow 14.0.2 and
MLflow 2.12.2. The cloud preflight checks these versions. Do not install this
local requirements file into an active Spark session.

Import notebook 00 for table-only loading. Import notebooks 01 and 02 if you
also want modeling and explanations. Attach the configured Lakehouse as the
default for each, in the configured workspace. Run notebook 00 with the exact
uploaded ``SOURCE_FOLDER``. For modeling, run notebook 01 first with
``RUN_MODE = 'validate'`` and ``ALLOW_DEMO_TABLE_REPLACEMENT = False``.
Keep ``INPUT_MODE = 'folder'`` to reuse that folder. Only after it passes
and you have reviewed replacement scope, change
``RUN_MODE`` to ``'publish'`` and the acknowledgment to ``True``, then run
top to bottom. Folder and ZIP publication use the supplied small profile.
For a larger dataset, set ``INPUT_MODE = 'generate'`` before running the
configuration cell; the generator branch uses the full profile and computes
its features from scratch. Do not run these workflows concurrently.

The package's local validation does not verify your tenant's permissions,
capacity, catalog behavior or MLflow integration. These remain manual runtime
checks. Exact numbered steps are in `replication guide <docs/replication-guide.rst>`_.
The new portable folder path has not been run end to end in Fabric. Local
tests are not a cloud execution claim.

Configuration and storage
-------------------------

``config/simulation.json`` controls profiles, seed and the exact target
workspace, Lakehouse and MLflow experiment names. Defaults are workspace
``Academic Early Warning System``, Lakehouse ``academic_early_warning``,
experiment ``Academic Early Warning System Training``, seed 271828.
``config/modeling.json`` controls the fixed calibration and selection protocol.
``config/canvas-source-contract.json`` describes the selected public projection.
No tenant, workspace, Lakehouse or item identifiers are embedded.

Builders consume these files without an environment-specific fallback::

    .\.venv\Scripts\python.exe tools\build_notebooks.py --config-dir config

Stable Lakehouse schemas are ``canvas``, ``canvas_logs``, ``sim_history``,
``sim_support``, ``silver`` and ``gold``. The uploaded input root is
``Files/academic-early-warning-simulated-data`` unless you choose another name.
Publication Files roots are ``data``, ``model_artifacts`` and ``_archive``.
The current publication pointer is
``Files/model_artifacts/current.json`` and includes code, training, source,
model and run provenance.

Safety and replacement scope
----------------------------

* Notebook 00 never overwrites files or tables. It preflights all existing
  target tables, verifies and skips identical ones, and refuses mismatches
  before creating new tables. It creates neither a model nor ``gold`` outputs.
* Never attach a replacement-enabled notebook 01 to production tables.
* In notebook 01, only the contract's named source tables, their
  ``sim_history`` revision tables, two ``sim_support`` tables, two ``silver``
  feature tables and four ``gold`` result tables are replaced.
  Other tables are not deleted.
* All existing named tables are archived and verified before the first
  replacement, including tables imported by notebook 00. New immutable
  snapshots are created before the pointer changes.
* ``current.json`` advances only after verification. The exploration notebook
  reads that publication's immutable snapshots, not a mixture of active tables.
* Publication is not a multi-table transaction. On a failure, stable tables
  may be partially replaced; the previous pointer and snapshots remain for
  recovery. Use one writer only and do not explore during publication.
* No automatic archive retention, rollback, locking, budget alerts, scheduling
  or capacity pause/resume is supplied. Stop idle sessions and use approved
  cost controls. Archived data and storage can continue to incur charges.
* MLflow model loading deserializes code. Only load this trusted workspace's
  own artifacts; never point the reader to an untrusted model.

Interpretation and release review
--------------------------------

Terms 1-2 fit imputation, scaling and candidates. Term 3 is split into
calibration and decision folds. The decision fold selects a logistic candidate
by macro one-vs-rest average precision, then log loss, with a prevalence
baseline for context. Term 4 was previously exposed in the prototype and is
**not a new independent validation cohort**. Optional term-4 diagnostics
default off. All synthetic metrics are demonstration-only.

Global importance is mean absolute training-centered contribution to each
underlying linear class score. Local values are signed class-score
contributions, including grouped missing indicators. These are not causal
effects, percentage contributions, probability-point changes or tests of
feature significance. See `model card <docs/model-card.rst>`_.

No open-source license is selected. Review ``LICENSE_NOTICE.txt`` and complete
organization-approved licensing, privacy, security and release review before
public distribution. The hygiene scanner is a convenience, not a release
approval. Exclude ``outputs``, environments, caches, credentials and generated
artifacts when sharing or creating an archive. The deliberate exception is the
single version-matched input ZIP under ``sample-data``; keep it included.

Troubleshooting
---------------

* **Illegal Parquet type: INT64 (TIMESTAMP(NANOS,true)):** an earlier bundle
  encoded timestamps in the two prepared feature files at nanosecond precision.
  Use the corrected simulated-data ZIP and matching build notebook from the
  same repository revision. Replace the previously uploaded affected files
  in your own demo folder. Selecting Parquet in the UI cannot change the
  timestamp encoding already inside a file.
* **Folder not found or missing manifest:** set ``SOURCE_FOLDER`` to the
  exact uploaded root, including a ``-fixed`` suffix if present. Check for
  accidental double nesting. Its direct children must include ``source_parquet``,
  ``features``, ``manifest.json`` and ``summary.json``.
* **Existing table mismatch in notebook 00:** no new tables are created.
  Review the conflict or use a separate empty development Lakehouse with
  matching configuration. Do not delete data to bypass the guard.
* **ZIP not found:** upload the inner dataset ZIP to ``Files/uploads``, not
  the outer repository ZIP. Names are case-sensitive.
* **Checksum mismatch:** use the data shipped with these notebooks. Keep all
  extracted files unchanged; do not recompress the ZIP alternative. After a
  deliberate dataset rebuild using ``python tools\build_data_bundle.py --replace``,
  rebuild all notebooks.
* **Target mismatch:** attach the Lakehouse as default and configure names
  consistently in all notebooks. Restart after changing the default Lakehouse.
* **Runtime mismatch:** choose Runtime 1.3 and review the reported versions.
  Do not bypass guards or install packages mid-session without review.
* **No current publication:** finish notebook 01 in publish mode.
  Notebook 00's ``tables_loaded`` message only confirms input table loading.
* **Paused or throttled capacity:** resume or wait for existing work; do not
  start concurrent copies of the same notebook.
* **Poor class metrics:** preserve the negative finding. Synthetic results
  are illustrative, not institutional performance guarantees.
