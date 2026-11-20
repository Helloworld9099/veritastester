"""Install the old build and apply a real native upgrade from a signed test fixture."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from urllib.request import Request

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from sign_update import signed_document
from bridgecaption import updates

OLD, NEW = "0.1.0", "0.1.1"
SETUP_FLAGS = ["/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"]


def worker(executable, package):
    # This subprocess exits to let the same detached helper used by the app
    # replace the installed bundle/files. The initial path is a real install.
    sys.frozen = True
    sys.executable = str(executable)
    updates.install_update_on_exit(package)


def read_version(executable, report):
    report.unlink(missing_ok=True)
    subprocess.run([str(executable), "--version-file", str(report)], check=True, timeout=30)
    return report.read_text().strip()


def smoke(previous, candidate):
    key = updates.platform_key()
    system = platform.system()
    with tempfile.TemporaryDirectory(prefix="BridgeCaption update test ") as temporary:
        root = Path(temporary)
        report = root / "version.txt"
        if system == "Windows":
            install_root = root / "Installed App"
            subprocess.run([str(previous), *SETUP_FLAGS, "/DIR=" + str(install_root)],
                           check=True, timeout=120)
            executable = install_root / "BridgeCaption.exe"
        elif system == "Darwin":
            subprocess.run(["/usr/bin/ditto", "-x", "-k", str(previous), str(root)], check=True)
            executable = root / "BridgeCaption.app/Contents/MacOS/BridgeCaption"
        else:
            raise RuntimeError("Expected a Windows or macOS runner")
        assert read_version(executable, report) == OLD
        sentinel = root / "preserve.txt"
        sentinel.write_text("keep")
        signing_key = Ed25519PrivateKey.generate()
        updates.UPDATE_PUBLIC_KEY = base64.b64encode(
            signing_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()
        document = signed_document(NEW, [(key, candidate, "https://update.invalid/package")], signing_key)
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps(document))
        # Synthetic transport only; signed manifest, package bytes, updater,
        # installation, and post-upgrade executable are all real.
        def transport(request: Request, *, timeout):
            return (candidate if request.full_url == "https://update.invalid/package" else manifest).open("rb")
        updates.urlopen = transport
        update = updates.find_update(OLD)
        assert update is not None and update.version == NEW
        downloaded = updates.download_update(update, lambda done, total: None)
        try:
            subprocess.run([sys.executable, __file__, "--worker", str(executable), str(downloaded)],
                           check=True, timeout=180)
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                try:
                    if read_version(executable, report) == NEW:
                        break
                except (OSError, subprocess.SubprocessError):
                    pass
                time.sleep(3)
            else:
                raise RuntimeError("Upgraded install never reported the new version")
            assert sentinel.read_text() == "keep"
            assert updates.find_update(NEW) is None
            print(f"{key}: installed {OLD}, applied verified {NEW} in place, and relaunched")
        finally:
            # Relaunch is part of the production path; close it in the disposable
            # runner before cleanup (Windows keeps loaded libraries mapped).
            if system == "Windows":
                subprocess.run(["taskkill", "/IM", "BridgeCaption.exe", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            elif system == "Darwin":
                subprocess.run(["/usr/bin/pkill", "-f", str(executable)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            updates.discard_update(downloaded)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    (worker if args.worker else smoke)(args.first.resolve(), args.second.resolve())
