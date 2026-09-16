"""Conservative release hygiene scan; not a substitute for release review."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {".git", ".venv", "__pycache__", "outputs"}
TEXT_SUFFIXES = {".py", ".json", ".ipynb", ".rst", ".txt", ".svg", ".excalidraw"}
PATTERNS = {
    "absolute user/workspace path": re.compile(r"(?i)[a-z]:[\\/]+(?:users|repo)[\\/]+"),
    "literal resource identifier": re.compile(r"(?i)\b[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\b"),
    "email address": re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b"),
    "bearer credential": re.compile(r"(?i)bearer\s+[a-z0-9._~-]{15,}"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "JWT credential": re.compile(r"\beyJ[a-zA-Z0-9_-]{15,}\.[a-zA-Z0-9_-]{15,}\.[a-zA-Z0-9_-]+"),
    "development naming": re.compile(r"(?i)\b(?:sm[o]ke|canvas_[d]emo|css-v[12])\b"),
    "embedded cloud item URL": re.compile(r"https://(?:app\.fabric\.microsoft\.com|api\.fabric\.microsoft\.com)/"),
}


def scan_text(text):
    return [name for name, pattern in PATTERNS.items() if pattern.search(text)]


def scan(root):
    findings, scanned = [], 0
    for path in sorted(Path(root).rglob("*")):
        relative = path.relative_to(root)
        if any(part in IGNORED for part in relative.parts) or not path.is_file():
            continue
        if relative.as_posix() == "sample-data/academic-early-warning-simulated-data.zip":
            info = json.loads((Path(root) / "sample-data" / "bundle-info.json").read_text(encoding="utf-8"))
            if path.stat().st_size != info["zip_bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != info["zip_sha256"]:
                findings.append((str(relative), "included synthetic dataset checksum mismatch"))
            continue
        if path.suffix in {".pkl", ".pickle", ".parquet", ".joblib", ".log", ".zip"}:
            findings.append((str(relative), "generated or private artifact"))
            continue
        findings += [(str(relative), issue) for issue in scan_text(str(relative))]
        if path.suffix not in TEXT_SUFFIXES and path.name not in {".gitignore", ".gitattributes"}:
            continue
        scanned += 1
        text = path.read_text(encoding="utf-8")
        if not text.isascii():
            findings.append((str(relative), "non-ASCII text"))
        findings += [(str(relative), issue) for issue in scan_text(text)]
    return scanned, findings


def main():
    parser = argparse.ArgumentParser(description="Scan shareable sources, ignoring local generated outputs and environments.")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    scanned, findings = scan(args.root)
    for path, issue in findings:
        print(path + ": " + issue)
    print(f"Scanned {scanned} text files; {len(findings)} findings.")
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
