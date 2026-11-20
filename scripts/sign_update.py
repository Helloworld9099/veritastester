"""Sign a manifest for one or more release artifacts without exposing the private key."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bridgecaption.updates import MAX_PACKAGE, PACKAGE_SUFFIXES, _https, version_tuple


def signed_document(version, specs, private_key):
    version_tuple(version)
    artifacts = {}
    for target, path, url in specs:
        _https(url)
        os_name, _, machine = target.partition("-")
        if os_name not in PACKAGE_SUFFIXES or machine not in {"x86_64", "arm64"}:
            raise ValueError(f"Unsupported target: {target}")
        if target in artifacts:
            raise ValueError(f"Duplicate target: {target}")
        src = Path(path)
        if src.suffix != PACKAGE_SUFFIXES[os_name]:
            raise ValueError(f"Invalid package format for {target}: {src.name}")
        size = src.stat().st_size
        if not 0 < size <= MAX_PACKAGE:
            raise ValueError(f"Invalid size for {src.name}")
        with src.open("rb") as f:
            sha = hashlib.sha256()
            for block in iter(lambda: f.read(128 * 1024), b""):
                sha.update(block)
            digest = sha.hexdigest()
        artifacts[target] = {"url": url, "sha256": digest, "size": size}
    if not artifacts:
        raise ValueError("At least one artifact is required")
    payload = {"version": version.removeprefix("v"), "artifacts": artifacts}
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return {"payload": payload, "signature": base64.b64encode(private_key.sign(data)).decode("ascii")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument("--artifact", action="append", nargs=3, required=True, metavar=("TARGET", "FILE", "URL"))
    args = parser.parse_args()
    key = Ed25519PrivateKey.from_private_bytes(
        base64.b64decode(os.environ["BRIDGECAPTION_UPDATE_PRIVATE_KEY"], validate=True))
    print(json.dumps(signed_document(args.version, args.artifact, key), indent=2))
