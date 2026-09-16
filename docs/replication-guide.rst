Replication guide
=================

Start here
----------

Follow the root ``README.rst`` Quickstart to upload the included dataset and
run the prebuilt notebooks. It does not require local data generation or a
local Python installation when using the default names.

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
   experiment names. Both notebooks use the same configuration. Keep the
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

5. Confirm both notebooks contain no executed output. They embed the required
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
   MLflow 2.12.2. The guard refuses mismatched versions. Investigate runtime
   changes rather than bypassing the guard or installing packages mid-session.

8. Create ``Files/uploads`` in the target Lakehouse and upload the still-zipped
   ``sample-data/academic-early-warning-simulated-data.zip`` there.
   Manually import ``notebooks/01_build_and_publish.ipynb`` and
   ``notebooks/02_explore_and_explain.ipynb``. Attach the intended Lakehouse
   as the default for both notebooks. They must run in that same workspace.
   No tenant-specific metadata or default-Lakehouse identifier is prefilled.

9. Open the first notebook, start the existing Fabric-managed Spark session,
   and run top to bottom with the default ``RUN_MODE = 'validate'``.
   Leave ``INPUT_MODE = 'uploaded'`` and
   ``ALLOW_DEMO_TABLE_REPLACEMENT = False``. Validation checks package
   versions, active session, workspace and default-Lakehouse names, the Files
   mount, exact ``workspace.Lakehouse.dbo`` namespace, and the uploaded ZIP's
   release hash, member checksums, source/feature schemas and counts. It creates no
   tables or model experiment. A mismatch must be corrected before proceeding.

10. Review the scope below. Only for the intended synthetic development
    Lakehouse, set ``RUN_MODE = 'publish'`` and
    ``ALLOW_DEMO_TABLE_REPLACEMENT = True``. Keep ``INPUT_MODE = 'uploaded'``
    and registration disabled. The dataset profile is truthfully recorded as
    ``small``. No source generation or feature recomputation is performed:
    the prepared features shipped with the source histories are used.
    Run top to bottom with a single writer.
    This is a separate, explicitly authorized cloud workload.

11. Wait for ``publication_completed``. Review the printed actual source
    counts, source assertions, unknown-label exclusions, calibration/decision
    sample sizes, per-class metrics and the current manifest. Do not invent
    accuracy expectations or alter outcomes to make a chart look better.

12. Open the second notebook only after the writer has finished. Attach the
    same Lakehouse and run top to bottom. It reads ``current.json``, verifies
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

13. Stop notebook sessions when finished. Apply your approved capacity,
    archive-retention and budget controls manually. Storage and previously
    consumed compute can remain billable. This starter does not pause capacity
    or implement a scheduled operations service.

Generate a larger dataset instead
---------------------------------

Set ``INPUT_MODE = 'generate'`` in the first notebook's configuration cell.
The supplied logic selects the full profile (5,000 learners). Run validate,
then explicitly authorize publish as above. This path generates source
history and features rather than reading the included ZIP. It uses the same
publication and exploration workflow but takes longer and uses more compute.
The two input paths are alternatives; do not start them concurrently.

Rebuild the included ZIP locally only when needed::

    .\.venv\Scripts\python.exe tools\build_data_bundle.py --replace
    .\.venv\Scripts\python.exe tools\build_notebooks.py

The builder writes new checksums into ``sample-data/bundle-info.json``.
Notebooks embed those checksums, so data and notebooks must be rebuilt together.
Do not distribute an independently modified ZIP with old notebooks.

Replacement, preservation and recovery
--------------------------------------

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

Public product references
-------------------------

* `Fabric Runtime 1.3 <https://learn.microsoft.com/fabric/data-engineering/runtime-1-3>`_
* `Lakehouse schemas <https://learn.microsoft.com/fabric/data-engineering/lakehouse-schemas>`_
* `Canvas DAP documentation <https://developerdocs.instructure.com/services/dap>`_

These references describe products, not successful execution in your tenant.
Authentication, resource provisioning and customer approval are deliberately
outside this portable starter.
