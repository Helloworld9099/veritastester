"""Pinned public research sources. Dry-run by default; never executes dataset code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def acquire(item, directory):
    path = directory / item["file"]
    if path.exists():
        if path.stat().st_size != item["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Existing download differs from pinned source: {path.name}")
        return "verified-existing"
    temporary = path.with_suffix(path.suffix + ".part")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(item["url"], timeout=60) as response:
            n = 0
            while block := response.read(1024 * 1024):
                n += len(block)
                if n > item["bytes"]:
                    raise ValueError("Download exceeds pinned size")
                out.write(block)
        if temporary.stat().st_size != item["bytes"] or hashlib.sha256(temporary.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Checksum or byte size differs from pinned source")
        temporary.rename(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return "downloaded-verified"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--download", action="store_true")
    args = p.parse_args()
    manifest = json.loads((ROOT / "docs/dataset_downloads.json").read_text())
    directory = ROOT / "data/private/raw"
    if args.download:
        directory.mkdir(parents=True, exist_ok=True)
    for item in manifest:
        status = acquire(item, directory) if args.download else "dry-run"
        print(json.dumps({"file": item["file"], "bytes": item["bytes"], "status": status}))


if __name__ == "__main__":
    main()
