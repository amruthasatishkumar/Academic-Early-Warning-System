Replication guide
=================

Start here
----------

**Do not use Load to Tables for the replication Quickstart.** Follow the root
``README.rst`` to extract the simulated-data ZIP, choose **Files > Upload
folder**, and run ``00_load_lakehouse_tables.ipynb`` once. It creates all 27
input tables without a per-table wizard. No local generation or Python
installation is needed with default names.

Keep all files and nesting, with only one dataset root. If your folder is
already uploaded, set ``SOURCE_FOLDER`` to its exact root and reuse it.
Do not upload a ZIP again. Notebook 00 creates or verifies tables only.
Notebook 01 optionally trains and publishes using that same folder; notebook
02 requires that completed model publication, not just the imported tables.

For the mapping applied automatically by notebook 00, see the root README's
`Parquet file-to-table names <../README.rst#parquet-file-to-table-names>`_
table. It maps all 115 Parquet files to 27 input tables with separate
**Schema** and **New table name** values. Keep filenames unchanged; all parts
from a single table folder form one table.

This guide adds local development instructions and details about publication,
preservation and adapting the target environment.

Optional local prerequisites
-----------------------------

1. Obtain this complete repository, including ``aews``, ``config``, ``tools``,
   ``tests``, ``notebooks``, ``sample-data`` and the presentation assets. Do not copy another
   project's working directory, outputs, credentials or model binaries.

2. Install Python 3.11 locally. From the repository root in PowerShell::

       py -3.11 -m venv .venv
       .\.venv\Scripts\python.exe -m pip install -r requirements.txt

   These pins reproduce the supported local Python-library target, not a
   local Fabric Spark service. No Python 3.10 equivalence is implied.

3. Edit ``config/simulation.json`` to your chosen workspace, Lakehouse and
   experiment names. All three notebooks use the same configuration. Keep the
   provided seed and label logic when reproducing the synthetic example.
   Do not add credentials or private resource identifiers.

4. Build and validate locally::

       .\.venv\Scripts\python.exe tools\build_notebooks.py
       .\.venv\Scripts\python.exe -m unittest discover -s tests -v
       .\.venv\Scripts\python.exe tools\validate_included_data.py
       .\.venv\Scripts\python.exe tools\scan_package.py

   The included-data command validates the shipped ZIP and trains on its
   prepared features, without calling the generator. To generate a new
   population locally, use ``tools\run_local.py --profile small``; this writes
   a new ignored output directory. Failed class coverage is an error, not
   permission to search for a more favorable seed.

5. Confirm all three notebooks contain no executed output. They embed the required
   source modules and configuration; cloud imports do not depend on this
   local folder. Editing a Python file alone does not update a cloud notebook.
   Always rebuild and reimport after edits.

Manual Fabric setup and run
---------------------------

6. In your approved development environment, select capacity and region
   according to quota, budget and policy. Create the configured workspace,
   assign it to capacity, and grant only the access needed for the demo.
   No capacity size, region, entitlement or cost estimate is guaranteed here.

7. Create the configured Lakehouse **with schemas enabled** in that workspace.
   Choose Fabric Runtime 1.3: Python 3.11, Spark 3.5, Delta 3.2. Confirm
   NumPy 1.26.4, pandas 2.1.4, scikit-learn 1.2.2, PyArrow 14.0.2 and
   MLflow 2.12.2 is required only for notebooks 01 and 02, not table loading
   in notebook 00. If MLflow package metadata is absent, its module version is
   checked instead. Delta's Python and Spark classes must be available; Fabric
   may provide them without ``delta-spark`` package metadata. In that case the
   notebook reports the exact Delta version as unavailable, not a verified 3.2.
   A present incompatible package version is still rejected. Investigate
   runtime changes rather than installing packages mid-session.

8. Extract ``sample-data/academic-early-warning-simulated-data.zip`` on your
   computer. In the target Lakehouse **Files**, choose **Upload folder**,
   not **Load to Tables**, and select the complete extracted folder.
   Retain all files and nesting. Confirm the root is not doubly nested::

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

9. Import ``notebooks/00_load_lakehouse_tables.ipynb`` into the configured
   workspace. Its first cell sets the session's default Lakehouse by name::

       %%configure
       {
         "defaultLakehouse": {
           "name": "academic_early_warning"
         }
       }

   Stop an existing notebook session before **Run all** so this first cell
   takes effect. It selects an existing Lakehouse in the notebook's workspace;
   it does not create a Lakehouse, publish identifiers, or change capacity.
   In block 3 set
   ``SOURCE_FOLDER = 'Files/academic-early-warning-simulated-data'`` to the
   exact uploaded root. For a folder ending in ``-fixed``, use
   ``Files/academic-early-warning-simulated-data-fixed`` instead.
   Choose **Run all**. Fabric starts the configured session; the Python code
   never creates a SparkSession or installs packages.

   The loader verifies runtime, target names, the release manifest hash,
   file bytes/checksums, schemas and rows. It preflights existing targets:
   identical tables are verified and skipped; any mismatch stops before
   new tables are created. It reports each ``schema.table``, ``created`` or
   ``already_matches``, and its verified Spark row count. ``tables_loaded``
   appears only after all **25 source/history/support and two silver feature
   tables** are complete. No files or tables are overwritten or deleted.
   It does not train, create gold placeholders, or publish a model.
   If you only need tables, skip to step 14.

10. **Optional model workflow:** import ``notebooks/01_build_and_publish.ipynb``
    and ``notebooks/02_explore_and_explain.ipynb`` with the same default
    Lakehouse. They must run in the configured workspace. In notebook 01
    leave ``INPUT_MODE = 'folder'``, ``RUN_MODE = 'validate'`` and
    ``ALLOW_DEMO_TABLE_REPLACEMENT = False``. Set ``SOURCE_FOLDER`` to the
    same uploaded root used in notebook 00. No second upload is needed.
    Run top to bottom. Validation checks package versions, active session,
    target names, Files mount, namespace, file integrity and source/feature
    schemas, counts and boundaries. It writes no tables or model experiment.
    Correct mismatches before proceeding.

11. Review the scope below. Only for the intended synthetic development
    Lakehouse, set ``RUN_MODE = 'publish'`` and
    ``ALLOW_DEMO_TABLE_REPLACEMENT = True``. Keep ``INPUT_MODE = 'folder'``
    and registration disabled. The dataset profile is truthfully recorded as
    ``small``. No source generation or feature recomputation is performed:
    the prepared features shipped with the source histories are used.
    Run top to bottom with a single writer.
    Notebook 01 still archives and verifies the 27 named input tables before
    replacing them, including tables just imported by notebook 00, as well
    as existing named gold outputs. Notebook 00's non-overwriting behavior
    does not change the publication replacement policy.
    This is a separate, explicitly authorized cloud workload.

12. Wait for ``publication_completed``. Review the printed actual source
    counts, source assertions, unknown-label exclusions, calibration/decision
    sample sizes, per-class metrics and the current manifest. Do not invent
    accuracy expectations or alter outcomes to make a chart look better.

13. Open notebook 02 only after notebook 01's publication has finished.
    ``tables_loaded`` from notebook 00 alone does not satisfy this requirement.
    Attach the same Lakehouse and run top to bottom. It reads ``current.json``, verifies
    immutable small/full snapshot counts and training provenance, loads that run's model,
    and checks all current probabilities and predicted classes.
    Review global contributions for all four outcomes. Block 6 automatically
    fills and prints a valid enrollment ID and its predicted outcome, ready
    for block 7. The enrollment ID originates in ``canvas.enrollments.id``;
    it refers to a student-course enrollment, not the student alone.
    ``predicted_class`` is the model's highest-probability outcome, not a
    known future result.

    To select another case, replace the ``ENROLLMENT_TO_EXPLAIN`` assignment
    with an ID from the displayed predictions and rerun blocks 6 and 7.
    The outcome will be filled from that enrollment's prediction. To explain
    a different class, explicitly set ``OUTCOME_TO_EXPLAIN`` to
    ``'excellent'``, ``'pass'``, ``'fail'`` or ``'withdrawn'`` in block 6.
    This changes only the score being explained, not the model's prediction.

14. Stop notebook sessions when finished. Apply your approved capacity,
    archive-retention and budget controls manually. Storage and previously
    consumed compute can remain billable. This starter does not pause capacity
    or implement a scheduled operations service.

ZIP upload alternative
----------------------

Keep the inner simulated-data ZIP compressed and unchanged if you prefer
this supported alternative. Create ``Files/uploads``, choose **Upload files**
and upload ``sample-data/academic-early-warning-simulated-data.zip`` there::

    Files/
        uploads/
            academic-early-warning-simulated-data.zip

Skip notebook 00. Set ``INPUT_MODE = 'uploaded'`` in notebook 01.
``UPLOAD_PATH`` then uses ``BUNDLE_INFO['expected_lakehouse_path']``.
The notebook verifies the ZIP's release checksum and included files, then
uses its prepared features. Follow the same validation, explicit publication
authorization and exploration steps above. Do not upload the outer repository
ZIP or recompress the data ZIP. Folder mode does not require this upload.

Generate a larger dataset instead
---------------------------------

Set ``INPUT_MODE = 'generate'`` in notebook 01's configuration cell.
The supplied logic selects the full profile (5,000 learners). Run validate,
then explicitly authorize publish as above. This path generates source
history and features rather than reading the included folder or ZIP. It uses
the same publication and exploration workflow but takes longer and uses more compute.
The three input modes are alternatives; do not start them concurrently.

Rebuild the included ZIP locally only when needed::

    .\.venv\Scripts\python.exe tools\build_data_bundle.py --replace
    .\.venv\Scripts\python.exe tools\build_notebooks.py

The builder writes ZIP and manifest checksums into ``sample-data/bundle-info.json``.
Notebooks embed those checksums, so data and notebooks must be rebuilt together.
Do not distribute an independently modified ZIP with old notebooks.

Replacement, preservation and recovery
--------------------------------------

Notebook 00 is a non-overwriting import. Existing table conflicts are reported
before any new tables are created. If a later operation fails, some new
tables can remain; rerunning verifies and skips matching tables. Preserve
the error and review a mismatch, or use a separate empty development Lakehouse.
Do not delete existing data just to pass the guard. Use one writer at a time.

The remainder of this section describes notebook 01's authorized publication,
not notebook 00. Its archive behavior applies even after a table-only import.

The source contract names the ``canvas`` and ``canvas_logs`` tables.
Each mutable source table has a ``sim_history.<table>_revisions`` table.
``sim_support`` contains ``course_withdrawal_events`` and
``gradebook_qualification``. The supported outputs are:

* ``silver.historical_weekly_features``
* ``silver.current_weekly_features``
* ``gold.student_course_predictions``
* ``gold.prediction_explanations``
* ``gold.model_evaluation``
* ``gold.data_quality_summary``

Before the first table replacement, every existing named table is copied to
``Files/_archive/<publication>/tables/<schema>/<table>``. Schema, row count
and bidirectional row equivalence are checked. An existing current pointer is
also preserved. Nothing is silently deleted to resolve an archive conflict.

Raw source Parquet has a fresh path below ``Files/data`` for each publication.
New result snapshots and the immutable manifest are below
``Files/model_artifacts/<model>/publications/<publication>/``. A content-derived
model version includes code, training, calibration/decision data, configuration,
selected candidate and library provenance. Repeated data generation does not
overwrite a prior publication's files.

The pointer ``Files/model_artifacts/current.json`` changes last. The reader
uses the six immutable snapshot paths listed there. Stable tables are useful
for inspection, but are not a transactional view during a run. If a run fails,
stop other writers, preserve the error, and inspect the previous manifest and
archive. Do not delete archives or claim the partial run succeeded. A reviewed
manual recovery can restore a complete publication; no automatic rollback
utility is supplied. Rerunning a corrected full job preserves the existing
tables again before writing.

Read-only exploration still consumes compute and loads a trusted serialized
MLflow model. It does not modify tables, Files, model registry or predictions.

Custom names
------------

* In block 3 of each notebook you use, edit the ``SOURCE_CONFIG`` values for
  ``intended_workspace``, ``intended_lakehouse`` and ``experiment_name``.
  Keep the names consistent across notebooks and attach the correct default
  Lakehouse. Set ``SOURCE_FOLDER`` before its derived path is initialized.
  In notebook 00, also update ``defaultLakehouse.name`` in the first
  ``%%configure`` cell. Keep that cell first and stop the old session before rerunning.
* Alternatively, edit ``config/simulation.json`` locally and rebuild all three
  notebooks with ``python tools\build_notebooks.py`` before importing them.
  Editing configuration files alone does not change an imported notebook.
* Do not bypass target or runtime guards. Restart the notebook session after
  changing its default Lakehouse.

Troubleshooting
---------------

* **The exact workspace.Lakehouse.dbo namespace is not present:** an older
  notebook incorrectly required a particular ``SHOW SCHEMAS`` display format.
  Use the updated notebooks, which check the configured Lakehouse's ``dbo``
  schema directly with ``SHOW TABLES IN`` after validating the session target.
  No table needs to exist yet. Missing schemas and access errors still stop
  the run; do not remove the target checks or recreate your data.
* **PackageNotFoundError for mlflow or delta-spark in notebook 00:** import the
  updated loader and rerun all cells. It does not require MLflow and checks
  Fabric-provided Delta classes without assuming pip package metadata exists.
  Do not install those packages into the session just to satisfy the old guard.
* **Session not associated with a Lakehouse:** adding it to the explorer is not
  enough. In updated notebook 00, stop the session and run from its first
  ``%%configure`` cell. For notebooks 01 and 02, right-click the intended
  Lakehouse -> **Set as default lakehouse**, then stop/restart the session.
  This does not require restarting or resizing the Fabric capacity.
* **Folder not found or missing manifest:** set ``SOURCE_FOLDER`` to the exact
  uploaded root, including ``-fixed`` if present. Check for double nesting.
  Its direct children must include ``source_parquet``, ``features``,
  ``manifest.json`` and ``summary.json``.
* **Existing table mismatch in notebook 00:** it stops before creating new
  tables. Review the conflict or use a separate empty development Lakehouse
  with matching configuration. Do not delete data to bypass the guard.
* **TIMESTAMP(NANOS,true):** use the corrected data and matching notebooks from
  the same repository revision. Older feature files used incompatible timestamp
  encoding. Selecting Parquet in the UI does not convert that encoding.
* **Checksum mismatch:** keep all extracted files unchanged and use the data
  shipped with your notebooks. Do not recompress the ZIP alternative. After a
  deliberate dataset rebuild, rebuild all notebooks so their checksums match.
* **ZIP not found (ZIP alternative):** upload the inner dataset ZIP to
  ``Files/uploads``, not the outer repository ZIP. Names are case-sensitive.
* **Target mismatch:** use `Custom names`_ and attach the configured Lakehouse
  as default. Restart the session after changing its default Lakehouse.
* **Runtime mismatch:** choose Runtime 1.3 and review the reported package
  versions. Do not bypass guards or install packages mid-session.
* **No current publication:** finish notebook 01 in publish mode before using
  notebook 02. ``tables_loaded`` only confirms the input tables.
* **Paused or throttled capacity:** resume paused capacity according to your
  policy, or wait for existing work to finish. Do not start concurrent copies.
* **Poor model metrics:** preserve the finding. Synthetic results are
  illustrative, not institutional performance guarantees.

Model artifacts and cost controls
---------------------------------

* Only load the trusted workspace's own MLflow model artifacts. Model loading
  deserializes code; never point the reader at an untrusted model.
* Publication is not a multi-table transaction. A failed run can leave partial
  stable-table updates; preserve the previous manifest and immutable snapshots.
* No automatic locking, rollback, archive retention, budget alerts, scheduling
  or capacity pause/resume is supplied. Stop idle sessions and apply your own
  approved cost controls. Retained data and storage can continue to incur charges.
* The project is MIT-licensed (see ``LICENSE``). Exclude outputs, environments,
  caches and credentials when sharing; include the version-matched synthetic input ZIP.

Public product references
-------------------------

* `Fabric Runtime 1.3 <https://learn.microsoft.com/fabric/data-engineering/runtime-1-3>`_
* `Lakehouse schemas <https://learn.microsoft.com/fabric/data-engineering/lakehouse-schemas>`_
* `Configure the default Lakehouse in code <https://learn.microsoft.com/fabric/data-engineering/author-execute-notebook#spark-session-configuration-magic-command>`_
* `Canvas DAP documentation <https://developerdocs.instructure.com/services/dap>`_

These references describe products, not successful execution in your tenant.
Authentication, resource provisioning and customer approval are deliberately
outside this portable starter.
The new portable folder workflow has not been run end to end in Fabric.
Local checks do not validate cloud permissions, catalog behavior, capacity
or MLflow integration, and are not a cloud execution claim.
