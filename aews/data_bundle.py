"""Verify and load the included synthetic dataset without generating new records."""
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from aews.core.contract import dumps, frame, validate_table
from aews.core.execution import SUPPORT_SCHEMA, arrow_schema
from aews.core.features import CLASSES, FEATURES, FEATURE_VERSION, PARTITIONS
from aews.parquet import normalize_timestamp_units

FILENAME = "academic-early-warning-simulated-data.zip"
PURPOSE = "academic_early_warning_synthetic_input"
MAX_COMPRESSED = 100 * 1024 * 1024
MAX_UNCOMPRESSED = 512 * 1024 * 1024
MAX_MEMBER = 128 * 1024 * 1024
MAX_FILES = 512
FEATURE_METADATA = [
    "enrollment_id", "user_id", "course_id", "term", "cutoff_week", "as_of_time",
    "observation_start", "observation_end", "feature_version", "partition", "data_quality_flag",
]
LABEL_FIELDS = ["label", "label_status", "label_available_at"]


def sha256_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _hash_stream(handle)


def _hash_stream(handle):
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def generation_settings(config):
    return {key: config[key] for key in ("seed", "terms", "term_days", "cutoff_weeks", "current_week")} | {
        "profile": "small", "size": config["profiles"]["small"],
    }


def source_schemas(contract):
    result = {}
    for table, definition in contract["tables"].items():
        result[definition["namespace"] + "." + table] = arrow_schema(table, contract)
        if table not in ("accounts", "users", "enrollment_terms"):
            result["sim_history." + table + "_revisions"] = arrow_schema(table, contract, history=True)
    result.update({"sim_support." + name: schema for name, schema in SUPPORT_SCHEMA.items()})
    return result


def safe_member_name(name):
    path = PurePosixPath(name)
    if (not name or name.startswith("/") or "\\" in name or ":" in name or
            any(part in ("", ".", "..") for part in name.split("/")) or path.is_absolute()):
        raise ValueError("Unsafe dataset archive path.")
    return name


def validate_feature_frames(historical, current):
    for name, data, labels in (("historical", historical, LABEL_FIELDS), ("current", current, [])):
        if set(data.columns) != set(FEATURES + FEATURE_METADATA + labels):
            raise ValueError(f"Unexpected {name} feature columns, including prohibited future-label fields.")
        if data.empty or len(data) > 100000:
            raise ValueError("Feature population is empty or exceeds the bounded reader.")
        if data.duplicated(["enrollment_id", "cutoff_week"]).any() or not data.cutoff_week.eq(4).all():
            raise ValueError("Duplicate enrollment features or unsupported observation cutoff.")
        if not data.feature_version.eq(FEATURE_VERSION).all():
            raise ValueError("Feature definition version does not match this code.")
        if not data.partition.eq(data.term.map(PARTITIONS)).all():
            raise ValueError("Temporal partition mismatch.")
        if np.isinf(data[FEATURES].to_numpy(dtype=float)).any():
            raise ValueError("Infinite feature value.")
        if not data.observation_end.eq(data.as_of_time).all() or (data.observation_start > data.observation_end).any():
            raise ValueError("Invalid observation interval.")
    if not set(historical.term) <= {1, 2, 3, 4} or not current.term.eq(5).all():
        raise ValueError("Historical/current term separation failed.")
    if set(historical.enrollment_id) & set(current.enrollment_id):
        raise ValueError("An enrollment crosses historical/current datasets.")
    if not set(historical.label.dropna()) <= set(CLASSES):
        raise ValueError("Unknown target outcome.")
    for terms in ([1, 2], [3]):
        if set(historical.loc[historical.term.isin(terms), "label"].dropna()) != set(CLASSES):
            raise ValueError("Training or development labels lack an outcome class.")
    known = historical.loc[historical.label.notna()]
    if known.label_available_at.isna().any() or (known.label_available_at <= known.as_of_time).any():
        raise ValueError("Historical labels must become known after the prediction cutoff.")


@dataclass
class DatasetBundle:
    archive: Path
    manifest: dict
    features: pd.DataFrame
    summary: dict

    def extract_sources(self, destination):
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError("Dataset extraction requires a fresh staging directory.")
        if sha256_file(self.archive) != self.manifest["verified_zip_sha256"]:
            raise ValueError("The uploaded archive changed after validation.")
        destination.mkdir(parents=True)
        with zipfile.ZipFile(self.archive) as archive:
            for member in self.manifest["members"]:
                name = member["path"]
                if not name.startswith("source_parquet/"):
                    continue
                safe_member_name(name)
                target = destination.joinpath(*PurePosixPath(name).parts[1:])
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                if sha256_file(target) != member["sha256"]:
                    raise ValueError("Extracted source checksum mismatch.")


def load_bundle(path, info, config, contract):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Upload the included data ZIP to the configured Lakehouse Files path first.")
    if path.stat().st_size > MAX_COMPRESSED or path.stat().st_size != info["zip_bytes"]:
        raise ValueError("Dataset ZIP size does not match the included release.")
    digest = sha256_file(path)
    if not re.fullmatch(r"[0-9a-f]{64}", info["zip_sha256"]) or digest != info["zip_sha256"]:
        raise ValueError("Dataset ZIP checksum mismatch. Use the data ZIP shipped with these notebooks.")
    schemas = source_schemas(contract)
    tables, all_members = {}, []
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > MAX_FILES or len({e.filename for e in entries}) != len(entries):
            raise ValueError("Too many archive entries or duplicate member names.")
        for entry in entries:
            safe_member_name(entry.filename)
            if entry.is_dir() or entry.flag_bits & 1 or stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError("Unexpected directory, encrypted entry or symbolic link.")
            if entry.file_size > MAX_MEMBER:
                raise ValueError("Dataset member exceeds the uncompressed-size limit.")
        if sum(e.file_size for e in entries) > MAX_UNCOMPRESSED:
            raise ValueError("Dataset exceeds the total uncompressed-size limit.")
        if "manifest.json" not in archive.namelist() or archive.getinfo("manifest.json").file_size > 1024 * 1024:
            raise ValueError("Dataset manifest is absent or oversized.")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("purpose") != PURPOSE or manifest.get("version") != 1 or manifest.get("synthetic") is not True:
            raise ValueError("Wrong dataset type.")
        if manifest["generation_settings"] != generation_settings(config):
            raise ValueError("Included data does not match the configured synthetic profile/seed.")
        if manifest["contract_sha256"] != hashlib.sha256(dumps(contract).encode()).hexdigest():
            raise ValueError("Source contract does not match the bundled data.")
        listed = {item["path"] for item in manifest["members"]}
        if len(listed) != len(manifest["members"]) or set(archive.namelist()) != listed | {"manifest.json"}:
            raise ValueError("Missing, duplicated or unlisted dataset members.")
        summary = None
        for item in manifest["members"]:
            name = safe_member_name(item["path"])
            entry = archive.getinfo(name)
            if entry.file_size != item["bytes"]:
                raise ValueError("Dataset member size mismatch.")
            raw = archive.read(name)
            if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise ValueError("Dataset member checksum mismatch.")
            if name == "summary.json":
                summary = json.loads(raw)
                continue
            if not name.endswith(".parquet"):
                raise ValueError("Unexpected non-Parquet dataset member.")
            if name.startswith("source_parquet/"):
                parts = name.split("/")
                if len(parts) != 4 or not re.fullmatch(r"part-\d{2}\.parquet", parts[-1]):
                    raise ValueError("Unexpected source path.")
                table = parts[1] + "." + parts[2]
                if table not in schemas or item["table"] != table:
                    raise ValueError("Unexpected source table.")
            elif name in ("features/historical.parquet", "features/current.parquet"):
                table = name
            else:
                raise ValueError("Unexpected feature path.")
            arrow = pq.read_table(io.BytesIO(raw))
            arrow.validate(full=True)
            if arrow.num_rows != item["rows"] or str(arrow.schema.remove_metadata()) != item["schema"]:
                raise ValueError("Parquet schema or row-count mismatch.")
            if table in schemas and not arrow.schema.equals(schemas[table], check_metadata=False):
                raise ValueError("Parquet does not match its declared Canvas projection.")
            tables.setdefault(table, []).append(arrow)
            all_members.append(name)
    if set(tables) != set(schemas) | {"features/historical.parquet", "features/current.parquet"} or summary is None:
        raise ValueError("Incomplete source/feature/summary bundle.")
    if summary.get("synthetic") is not True or summary.get("profile") != "small":
        raise ValueError("Bundle is not the included small synthetic population.")
    references = {}
    for table, definition in contract["tables"].items():
        data = pa.concat_tables(tables[definition["namespace"] + "." + table]).to_pandas()
        for column, spec in definition["columns"].items():
            if spec["type"] == "array<string>":
                data[column] = data[column].map(lambda value: list(value) if value is not None else None)
        references[table] = frame(table, data.to_dict("records"), contract)
    for table, data in references.items():
        validate_table(table, data, contract, references)
        if len(data) != summary["observed_source_counts"][table]:
            raise ValueError("Source count differs from the verified generation summary.")
    historical = normalize_timestamp_units(tables["features/historical.parquet"][0].to_pandas())
    current = normalize_timestamp_units(tables["features/current.parquet"][0].to_pandas())
    validate_feature_frames(historical, current)
    enrollment_ids = set(references["enrollments"].id)
    if not (set(historical.enrollment_id) | set(current.enrollment_id)) <= enrollment_ids:
        raise ValueError("Feature enrollment lacks a source record.")
    current = current.copy()
    current["label"] = None
    current["label_status"] = "current_term_no_truth_published"
    current["label_available_at"] = pd.NaT
    features = pd.concat([historical, current], ignore_index=True)
    features["label_available_at"] = pd.to_datetime(features.label_available_at, utc=True)
    if len(features) != summary["feature_snapshots"]:
        raise ValueError("Prepared feature count mismatch.")
    manifest["verified_zip_sha256"] = digest
    summary["input_mode"] = "uploaded_verified_synthetic_bundle"
    summary["prepared_features"] = "Computed from the included synthetic history during dataset packaging, not regenerated on upload."
    return DatasetBundle(path, manifest, features, summary)
