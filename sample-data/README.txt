ACADEMIC EARLY WARNING SYSTEM - INCLUDED SIMULATED DATA

REPLICATION QUICKSTART: DO NOT USE LOAD TO TABLES.
Extract the repository ZIP, then extract the simulated-data ZIP on your computer.
In Lakehouse Files choose Upload folder and upload the complete extracted folder.
Run 00_load_lakehouse_tables.ipynb once to create all 27 input tables.
No per-table wizard, Parquet/CSV selection or table naming is needed.
Start with the repository README.rst for the full self-service steps.

File: academic-early-warning-simulated-data.zip
Use Extract All, then upload the resulting folder into your configured,
schema-enabled Fabric Lakehouse. Preserve all files, filenames and nesting.
No local Python is needed. Do not create a second dataset root inside the first.

Expected Lakehouse layout before running the notebook:
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

Attach the configured default Lakehouse to notebook 00. In block 3, set:
  SOURCE_FOLDER = 'Files/academic-early-warning-simulated-data'
Use the exact uploaded root. For example, if the folder ends in "-fixed", use
  SOURCE_FOLDER = 'Files/academic-early-warning-simulated-data-fixed'
Already uploaded this folder? Reuse it. Do not upload the ZIP again.

Run all once. The loader verifies the active Runtime 1.3 session, target names,
trusted manifest hash, file bytes/checksums, schemas and row counts. It creates
25 source/history/support tables and two silver feature tables. Identical
existing tables are verified and skipped; mismatches stop before new tables
are created. Per-table results report created or already_matches and verified
Spark row counts. tables_loaded appears only after all 27 tables are complete.
No files or tables are deleted or overwritten. No model or gold placeholders
are created. Use one writer; after a failed run, a rerun verifies any tables
already created and skips matching ones.

The README.rst "Parquet file-to-table names" reference preserves the exact map
of all 115 Parquet files to the 27 tables. Notebook 00 applies it automatically.
Different table folders remain separate; all five parts of one table stay
together. manifest.json and summary.json are metadata, not Parquet tables.

Optional model workflow:
Use 01_build_and_publish.ipynb with INPUT_MODE = 'folder' and the same
SOURCE_FOLDER. First validate with ALLOW_DEMO_TABLE_REPLACEMENT = False.
After reviewing scope, explicitly authorize publication as described in the
root README. Notebook 01 archives and verifies existing named input tables
before replacing them, including tables loaded by 00, along with existing
named gold outputs. It uses the included features without regeneration.
Only after publication_completed can 02_explore_and_explain.ipynb read the
model and explanations. tables_loaded from notebook 00 alone is not enough.

Supported ZIP alternative:
Keep the inner simulated-data ZIP compressed and unchanged. In Files/uploads,
choose Upload files and upload:
  Files/uploads/academic-early-warning-simulated-data.zip
Skip notebook 00. In notebook 01 choose INPUT_MODE = 'uploaded', then follow
the same validation and authorized publication steps. The notebook verifies
and unpacks the ZIP. This alternative is not required for an uploaded folder.

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
during packaging. Notebook 01's folder and uploaded modes use those features,
then fit and score the model in your environment without generating data again.
No pretrained model, previous predictions, credentials or real student records are included.
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

bundle-info.json beside the ZIP contains its release checksum, trusted
manifest checksum and exact size. It stays beside the ZIP in the repository;
the notebooks embed it, so it need not be added to the uploaded data folder.
The included archive has 117 files, approximately 15.6 MB compressed and
24.6 MB expanded. Use the exact byte values in bundle-info.json for integrity.

Keep the dataset and notebooks from the same repository version. If you
deliberately rebuild the dataset, rebuild the notebooks so checksums match.
See docs/replication-guide.rst for those developer commands.

Synthetic results illustrate the implementation, not institutional validity.
The new portable folder workflow has not been run end to end in Fabric;
local tests are not a cloud execution claim.
The independent real-data adapter and acceptance work remain in
docs/adapting-real-data.rst and docs/model-card.rst.
