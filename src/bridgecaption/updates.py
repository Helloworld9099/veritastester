"""Signed update manifest and streaming package download.

The signing public key is baked into official builds as BRIDGECAPTION_UPDATE_PUBLIC_KEY.
No unsigned executable may be installed.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import re
import tempfile
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

UPDATE_PUBLIC_KEY = ""  # Replaced at build time; never read trust roots from runtime environment.\nMAX_MANIFEST = 64 * 1024
MAX_PACKAGE = 500 * 1024 * 1024
DEFAULT_FEED = "https://github.com/Helloworld9099/veritastester/releases/latest/download/manifest.json"


@dataclass(frozen=True)
class Update:
    version: str
    url: str
    sha256: str
    size: int


def current_version() -> str:
    try:
        return version("bridgecaption")
    except PackageNotFoundError:
        return "0.1.0"


def version_tuple(value: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"v?(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", value)
    if not match:
        raise ValueError("Invalid semantic version")
    return tuple(map(int, match.groups()))


def _https(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Update URLs must be HTTPS without embedded credentials")
    return url


def _public_key():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    value = UPDATE_PUBLIC_KEY.strip()
    if not value:
        raise RuntimeError("This build has no trusted update signing key")
    return Ed25519PublicKey.from_public_bytes(base64.b64decode(value, validate=True))


def verify_manifest(document: dict) -> dict:
    """Verify signature over canonical payload, not arbitrary remote instructions."""
    if not isinstance(document, dict) or set(document) != {"payload", "signature"}:
        raise ValueError("Malformed signed manifest")
    payload = document["payload"]
    if not isinstance(payload, dict):
        raise ValueError("Malformed update payload")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    _public_key().verify(base64.b64decode(document["signature"], validate=True), canonical)
    return payload


def find_update(current: str | None = None) -> Update | None:
    feed = _https(os.getenv("BRIDGECAPTION_UPDATE_ENDPOINT", DEFAULT_FEED))
    req = Request(feed, headers={"User-Agent": "BridgeCaption-Updater/1"})
    with urlopen(req, timeout=10) as response:
        raw = response.read(MAX_MANIFEST + 1)
    if len(raw) > MAX_MANIFEST:
        raise ValueError("Update manifest too large")
    payload = verify_manifest(json.loads(raw))
    latest = payload["version"]
    version_tuple(latest)
    if version_tuple(latest) <= version_tuple(current or current_version()):
        return None
    system = platform.system().lower()
    machine = platform.machine().lower()
    key = f"{system}-{machine}"
    artifact = payload["artifacts"].get(key)
    if artifact is None:
        raise RuntimeError(f"No signed update for {key}")
    url = _https(artifact["url"])
    digest = artifact["sha256"]
    size = artifact["size"]
    if not isinstance(digest, str) or not re.fullmatch("[a-f0-9]{64}", digest):
        raise ValueError("Invalid package checksum")
    if type(size) is not int or not 0 < size <= MAX_PACKAGE:
        raise ValueError("Invalid package size")
    return Update(latest, url, digest, size)


def download_update(update: Update, progress, cancel=None) -> Path:
    """Download a verified package without running it. Caller owns returned temp path."""
    req = Request(_https(update.url), headers={"User-Agent": "BridgeCaption-Updater/1"})
    fd, name = tempfile.mkstemp(prefix="bridgecaption-update-", suffix=".package")
    destination = Path(name)
    received = 0
    digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as output, urlopen(req, timeout=30) as response:
            while True:
                if cancel and cancel():
                    raise RuntimeError("Update cancelled")
                chunk = response.read(128 * 1024)
                if not chunk:
                    break
                received += len(chunk)
                if received > update.size:
                    raise ValueError("Package exceeds declared size")
                digest.update(chunk)
                output.write(chunk)
                progress(received, update.size)
        if received != update.size or digest.hexdigest() != update.sha256:
            raise ValueError("Package integrity verification failed")
        return destination
    except BaseException:
        destination.unlink(missing_ok=True)
        raise

def install_appimage_on_exit(update_file: Path) -> None:
    """Stage verified AppImage; a detached helper replaces it after this process exits.

    A failed replacement leaves the existing application intact.
    """
    import subprocess
    import sys
    current = os.getenv("APPIMAGE")
    if platform.system() != "Linux" or not current:
        raise RuntimeError("Automatic installation requires the official AppImage build")
    target = Path(current).resolve()\n    if update_file.stat().st_dev != target.parent.stat().st_dev:\n        raise RuntimeError("Update staging must be on the same filesystem as the AppImage")
    if not target.is_file() or not os.access(target.parent, os.W_OK):
        raise RuntimeError("Cannot update this AppImage location; move it to a writable folder")
    # Package was checked against a signed manifest before arriving here.
    update_file.chmod(0o755)
    # Use a system shell: the bundled Python executable disappears when AppImage unmounts.
    helper = r"""
pid="$1"; source="$2"; destination="$3"
count=0
while kill -0 "$pid" 2>/dev/null; do
  count=$((count + 1))
  [ "$count" -lt 150 ] || exit 1
  sleep 0.2
done
backup="${destination}.previous"
if [ -e "$destination" ]; then
  mv -f -- "$destination" "$backup" || exit 1
fi
if ! mv -f -- "$source" "$destination"; then
  [ ! -e "$backup" ] || mv -f -- "$backup" "$destination"
  exit 1
fi
exec "$destination"
"""
    subprocess.Popen(
        ["/bin/sh", "-c", helper, "bridgecaption-update", str(os.getpid()), str(update_file), str(target)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
