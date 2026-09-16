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


class _DirectoryArchive:
    """Expose a bounded extracted dataset through the same reader as a ZIP."""

    def __init__(self, root):
        self.root = Path(root)
        if not self.root.is_dir() or self.root.is_symlink():
            raise ValueError("The dataset must be a real extracted folder.")
        self.entries = {}
        for path in self.root.rglob("*"):
            if path.is_symlink():
                raise ValueError("Symbolic links are not allowed in the dataset folder.")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError("Unexpected non-file dataset entry.")
            name = safe_member_name(path.relative_to(self.root).as_posix())
            entry = zipfile.ZipInfo(name)
            entry.file_size = path.stat().st_size
            self.entries[name] = entry
            if len(self.entries) > MAX_FILES:
                raise ValueError("Too many files in the dataset folder.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def infolist(self):
        return list(self.entries.values())

    def namelist(self):
        return list(self.entries)

    def getinfo(self, name):
        return self.entries[safe_member_name(name)]

    def open(self, name):
        entry = self.getinfo(name)
        path = self.root
        for part in ("", *PurePosixPath(name).parts):
            path = path / part
            if path.is_symlink():
                raise ValueError("Dataset path changed to a symbolic link.")
        if not path.resolve().is_relative_to(self.root.resolve()) or path.stat().st_size != entry.file_size:
            raise ValueError("Dataset file changed after folder inspection.")
        return path.open("rb")

    def read(self, name):
        if self.getinfo(name).file_size > MAX_MEMBER:
            raise ValueError("Dataset member exceeds the uncompressed-size limit.")
        with self.open(name) as handle:
            raw = handle.read(MAX_MEMBER + 1)
        if len(raw) != self.getinfo(name).file_size:
            raise ValueError("Dataset file size changed while reading.")
        return raw


def load_lakehouse_input(root, input_mode, input_path, info, config, contract):
    if input_mode not in ("uploaded", "folder"):
        raise ValueError("Choose uploaded for a ZIP or folder for an extracted dataset.")
    if not isinstance(input_path, str) or not input_path.startswith("Files/"):
        raise ValueError("The input path must start with Files/ in the default Lakehouse.")
    safe_member_name(input_path)
    root = Path(root).resolve()
    path = root.joinpath(*PurePosixPath(input_path).parts[1:])
    if not path.resolve().is_relative_to(root) or path.resolve() == root:
        raise ValueError("Dataset path must be a child of the default Lakehouse Files folder.")
    if input_mode == "uploaded" and input_path != info["expected_lakehouse_path"]:
        raise ValueError("Upload the included archive to the exact documented Files path.")
    if input_mode == "folder" and not path.is_dir():
        raise FileNotFoundError("Upload the extracted dataset folder and set SOURCE_FOLDER to its Files path.")
    if input_mode == "uploaded" and not path.is_file():
        raise FileNotFoundError("Upload the included data ZIP to Files/uploads first.")
    return load_bundle(path, info, config, contract)


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
    tables: dict

    def extract_sources(self, destination):
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError("Dataset extraction requires a fresh staging directory.")
        folder = self.manifest["verified_input_kind"] == "folder"
        if not folder and sha256_file(self.archive) != self.manifest["verified_zip_sha256"]:
            raise ValueError("The uploaded archive changed after validation.")
        with (_DirectoryArchive(self.archive) if folder else zipfile.ZipFile(self.archive)) as archive:
            if hashlib.sha256(archive.read("manifest.json")).hexdigest() != self.manifest["verified_manifest_sha256"]:
                raise ValueError("The input manifest changed after validation.")
            destination.mkdir(parents=True)
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
    folder = path.is_dir()
    digest = None
    if not folder:
        if not path.is_file():
            raise FileNotFoundError("Upload the included ZIP or extracted dataset folder first.")
        if path.stat().st_size > MAX_COMPRESSED or path.stat().st_size != info["zip_bytes"]:
            raise ValueError("Dataset ZIP size does not match the included release.")
        digest = sha256_file(path)
        if not re.fullmatch(r"[0-9a-f]{64}", info["zip_sha256"]) or digest != info["zip_sha256"]:
            raise ValueError("Dataset ZIP checksum mismatch. Use the data ZIP shipped with these notebooks.")
    schemas = source_schemas(contract)
    tables, all_members = {}, []
    with (_DirectoryArchive(path) if folder else zipfile.ZipFile(path)) as archive:
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
        manifest_bytes = archive.read("manifest.json")
        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
        if folder and (not re.fullmatch(r"[0-9a-f]{64}", info.get("manifest_sha256", "")) or
                       manifest_digest != info["manifest_sha256"]):
            raise ValueError("Dataset manifest checksum mismatch. Use the extracted data shipped with this notebook.")
        manifest = json.loads(manifest_bytes)
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
    manifest["verified_input_kind"] = "folder" if folder else "uploaded"
    manifest["verified_manifest_sha256"] = manifest_digest
    if digest is not None:
        manifest["verified_zip_sha256"] = digest
    summary["input_mode"] = "folder_verified_synthetic_bundle" if folder else "uploaded_verified_synthetic_bundle"
    summary["prepared_features"] = "Computed from the included synthetic history during dataset packaging, not regenerated on upload."
    return DatasetBundle(path, manifest, features, summary,
                         {name: pa.concat_tables(parts) for name, parts in tables.items()})
