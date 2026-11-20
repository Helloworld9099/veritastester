"""Build native desktop packages from the current source tree."""
from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bridgecaption.updates import platform_key, version_tuple


def build(version, output):
    version_tuple(version)
    version = version.removeprefix("v")
    output.mkdir(parents=True, exist_ok=True)
    target = platform_key()
    system = platform.system()
    public_key = os.environ.get("BRIDGECAPTION_UPDATE_PUBLIC_KEY", "")
    if public_key and len(base64.b64decode(public_key, validate=True)) != 32:
        raise ValueError("Update public key must be a base64-encoded 32-byte Ed25519 key")
    build_module = ROOT / "src/bridgecaption/_update_build.py"
    original = build_module.read_bytes()
    build_module.write_text(f"BUILD_VERSION = {version!r}\nUPDATE_PUBLIC_KEY = {public_key!r}\n", encoding="utf-8")
    try:
        with tempfile.TemporaryDirectory(prefix="bridgecaption-build-") as temporary:
            temp = Path(temporary)
            dist = temp / "dist"
            args = [
                "--name", "BridgeCaption", "--collect-all", "faster_whisper",
                "--collect-all", "ctranslate2", "--collect-all", "_sounddevice_data",
                "--paths", str(ROOT / "src"), "--specpath", str(temp),
                "--windowed",
            ]
            if system == "Darwin":
                args += ["--osx-bundle-identifier", "org.bridgecaption.desktop"]
                if identity := os.environ.get("BRIDGECAPTION_MACOS_SIGNING_IDENTITY"):
                    args += ["--codesign-identity", identity]
            args += [str(ROOT / "desktop_entry.py")]
            makespec = Path(sysconfig.get_path("scripts"), "pyi-makespec" + (".exe" if system == "Windows" else ""))
            subprocess.run([str(makespec), *args], check=True)
            spec = temp / "BridgeCaption.spec"
            if system == "Darwin":
                # The generated spec emits a single macOS BUNDLE call. Put version
                # and microphone permission in Info.plist before any signing.
                text = spec.read_text(encoding="utf-8")
                point = "bundle_identifier="
                if text.count(point) != 1:
                    raise RuntimeError("Cannot find PyInstaller's macOS bundle metadata")
                plist = {"CFBundleShortVersionString": version, "CFBundleVersion": version,
                         "NSMicrophoneUsageDescription": "BridgeCaption uses your microphone to caption speech."}
                spec.write_text(text.replace(point, f"info_plist={plist!r},\n    {point}"), encoding="utf-8")
            subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--distpath",
                            str(dist), "--workpath", str(temp / "build"), str(spec)], cwd=ROOT, check=True)
            if system == "Windows":
                iscc = shutil.which("ISCC.exe") or str(Path(os.environ["ProgramFiles(x86)"], "Inno Setup 6", "ISCC.exe"))
                subprocess.run([iscc, f"/DBridgeVersion={version}", f"/DOutputDirectory={output}",
                                f"/DDistDirectory={dist}", str(ROOT / "packaging/windows/BridgeCaption.iss")],
                               check=True)
            elif system == "Darwin":
                bundle = dist / "BridgeCaption.app"
                subprocess.run(["/usr/bin/codesign", "--verify", "--deep", "--strict", str(bundle)], check=True)
                subprocess.run(["/usr/bin/ditto", "-c", "-k", "--keepParent", str(bundle),
                                str(output / f"BridgeCaption-{target}.zip")], check=True)
            elif system == "Linux":
                shutil.make_archive(str(output / f"BridgeCaption-{target}-portable"), "gztar",
                                    root_dir=dist, base_dir="BridgeCaption")
            else:
                raise RuntimeError(f"No package format for {target}")
    finally:
        build_module.write_bytes(original)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.version, args.output.resolve())
