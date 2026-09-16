import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aews.configuration import load_configuration
from aews.execution import run_local


def main():
    parser = argparse.ArgumentParser(description="Generate and validate synthetic data locally; never calls Fabric.")
    parser.add_argument("--profile", choices=("small", "full"), default="small")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "config")
    parser.add_argument("--features-only", action="store_true", help="Skip raw Parquet staging, retaining feature/model validation.")
    args = parser.parse_args()
    source, model, contract = load_configuration(args.config_dir)
    source["mode"] = args.profile
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    output = ROOT / "outputs" / args.profile / stamp
    run_local(ROOT, source, model, contract, output, write_sources=not args.features_only)
    print("Local output: " + str(output.relative_to(ROOT)))


if __name__ == "__main__":
    main()
