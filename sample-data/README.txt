ACADEMIC EARLY WARNING SYSTEM - INCLUDED SIMULATED DATA

REPLICATION QUICKSTART: DO NOT USE LOAD TO TABLES.
Extract only the outer repository ZIP. Upload the inner data ZIP unchanged to
Lakehouse Files/uploads. The first notebook creates the separate tables.
Start with the repository README.rst for the full self-service steps.

File: academic-early-warning-simulated-data.zip
Upload this ZIP unchanged to:
  Files/uploads/academic-early-warning-simulated-data.zip
inside your configured, schema-enabled Fabric Lakehouse.

Extract the outer repository ZIP on your computer first. For the automated
notebook workflow, do not unzip the inner simulated-data ZIP in the Lakehouse
or use "Load to tables" on a ZIP. The build notebook verifies and unpacks it.
No local Python is needed for that path.

Expected Lakehouse layout before running the notebook:
  Files/
    uploads/
      academic-early-warning-simulated-data.zip
This is one ZIP file, not an extracted folder. No CSV/Parquet selection and
no table name is needed when uploading the file.

Advanced manual inspection only (skip for notebook replication):
1. Extract this simulated-data ZIP on your computer using Extract All.
2. Upload one table folder (with all its part-*.parquet files) to Lakehouse
   Files, then select that folder/file, Load to tables, and Parquet.
3. Never combine different source-table folders into one table. Load
   features/historical.parquet and features/current.parquet separately.
   manifest.json and summary.json are metadata, not Parquet tables.
4. Manual loading does not train or publish a model. The automated notebooks
   still require the original ZIP at the exact upload path above.
Manual loading is not the replication workflow. Use the notebook Quickstart
to train and publish the model and its explanations.

The corrected data files use microsecond Parquet timestamps, not nanoseconds.
If an old extracted feature file reports TIMESTAMP(NANOS,true), replace it
with the corrected file and use the matching notebooks for automated loading.

The data is entirely fictional, generated with seed 271828 using the small
profile in this repository. It includes:
  400 fictional learners and 8 fictional instructors
  5 terms, 40 course offerings, 120 course sections
  3,000 student enrollments, plus 40 instructor enrollments
  800 assignments and 60,000 submission records
  111,217 web activity records
  314 supplemental confirmed course-withdrawal events
  2,336 historical feature rows and 583 current feature rows

Prepared feature tables were computed from the included synthetic histories
during packaging. Uploaded mode uses those features, then fits and scores the
model in your environment. It does not generate the data again. No pretrained
model, previous predictions, credentials or real student records are included.
Current feature files contain no future label columns.

Archive contents:
  manifest.json: every member's checksum, size and Parquet schema/row count
  summary.json: generation validation, counts and limitations
  source_parquet/canvas: selected public-schema source records
  source_parquet/canvas_logs: synthetic activity
  source_parquet/sim_history: immutable source revisions
  source_parquet/sim_support: outcome/gradebook evidence
  features/historical.parquet: early features with later known outcomes
  features/current.parquet: early features only, no future labels

bundle-info.json beside the ZIP contains its release checksum and exact size.
The included archive has 117 files, approximately 15.6 MB compressed and
24.6 MB expanded. Use the exact byte values in bundle-info.json for integrity.

Keep the dataset and notebooks from the same repository version. If you
deliberately rebuild the dataset, rebuild the notebooks so checksums match.
See docs/replication-guide.rst for those developer commands.

Synthetic results illustrate the implementation, not institutional validity.
The independent real-data adapter and acceptance work remain in
docs/adapting-real-data.rst and docs/model-card.rst.
