"""Rebuild the included synthetic dataset; never connect to a cloud service."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pyarrow.parquet as pq
from aews.configuration import load_configuration
from aews.core.contract import dumps
from aews.core.execution import ParquetSourceSink, prepare
from aews.execution import split_features
from aews.data_bundle import FILENAME, PURPOSE, generation_settings, load_bundle, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true", help="Explicitly replace the shipped bundle, then rebuild notebooks.")
    args = parser.parse_args()
    output = ROOT / "sample-data"
    output.mkdir(exist_ok=True)
    target = output / FILENAME
    if target.exists() and not args.replace:
        raise FileExistsError("The dataset is already included. Use --replace only for a deliberate rebuild.")
    source, _, contract = load_configuration(ROOT / "config")
    source["mode"] = "small"
    with tempfile.TemporaryDirectory(prefix="aews-data-") as temporary:
        stage = Path(temporary)
        features, summary = prepare(source, contract, ParquetSourceSink(stage / "source_parquet", contract))
        historical, current = split_features(features)
        (stage / "features").mkdir()
        historical.to_parquet(stage / "features" / "historical.parquet", index=False)
        current.to_parquet(stage / "features" / "current.parquet", index=False)
        (stage / "summary.json").write_text(dumps(summary), encoding="ascii")
        members = []
        for path in sorted(stage.rglob("*")):
            if not path.is_file():
                continue
            name = path.relative_to(stage).as_posix()
            item = {"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            if path.suffix == ".parquet":
                arrow = pq.read_table(path)
                item.update(rows=arrow.num_rows, schema=str(arrow.schema.remove_metadata()))
                if name.startswith("source_parquet/"):
                    item["table"] = ".".join(name.split("/")[1:3])
            members.append(item)
        manifest = {"purpose": PURPOSE, "version": 1, "synthetic": True,
                    "generation_settings": generation_settings(source),
                    "contract_sha256": hashlib.sha256(dumps(contract).encode()).hexdigest(),
                    "members": members}
        (stage / "manifest.json").write_text(dumps(manifest), encoding="ascii")
        pending = output / (FILENAME + ".pending")
        with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    entry = zipfile.ZipInfo(path.relative_to(stage).as_posix(), (2024, 1, 1, 0, 0, 0))
                    entry.compress_type = zipfile.ZIP_DEFLATED
                    entry.external_attr = 0o100644 << 16
                    archive.writestr(entry, path.read_bytes())
        info = {"purpose": PURPOSE, "filename": FILENAME, "profile": "small", "synthetic": True,
                "zip_sha256": sha256_file(pending), "zip_bytes": pending.stat().st_size,
                "member_count": len(members) + 1,
                "uncompressed_bytes": sum(p.stat().st_size for p in stage.rglob("*") if p.is_file()),
                "seed": source["seed"], "counts": summary["observed_source_counts"],
                "historical_features": len(historical), "current_features": len(current),
                "expected_lakehouse_path": "Files/uploads/" + FILENAME,
                "prepared_features_included": True, "pretrained_model_included": False}
        loaded = load_bundle(pending, info, source, contract)
        if len(loaded.features) != len(features):
            raise RuntimeError("Bundle round trip changed the feature population.")
        pending.replace(target)
        (output / "bundle-info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="ascii")
        print(json.dumps(info))


if __name__ == "__main__":
    main()
