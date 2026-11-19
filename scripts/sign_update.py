"""Generate a signed update manifest from an existing AppImage.

Usage:
  BRIDGECAPTION_UPDATE_PRIVATE_KEY=<base64-raw-ed25519-private-key> \
  python scripts/sign_update.py 0.2.0 BridgeCaption.AppImage https://example.com/BridgeCaption.AppImage

Never commit signing private keys. Public key must be embedded in packaged clients.
"""
import base64
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def main():
    if len(sys.argv) != 4:
        raise SystemExit("Usage: sign_update.py VERSION APPIMAGE HTTPS_URL")
    version, filename, url = sys.argv[1:]
    if not url.startswith("https://"):
        raise SystemExit("Update URL must use HTTPS")
    from bridgecaption.updates import version_tuple
    version_tuple(version)
    source = Path(filename)
    key = Ed25519PrivateKey.from_private_bytes(
        base64.b64decode(os.environ["BRIDGECAPTION_UPDATE_PRIVATE_KEY"], validate=True))
    payload = {
        "version": version,
        "artifacts": {
            f"{platform.system().lower()}-{platform.machine().lower()}": {
                "url": url,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "size": source.stat().st_size,
            }
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    print(json.dumps({"payload": payload,
        "signature": base64.b64encode(key.sign(canonical)).decode()}, indent=2))


if __name__ == "__main__":
    main()
