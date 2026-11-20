"""Build native desktop packages from the current source tree."""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bridgecaption.updates import _https, platform_key, version_tuple


def official_settings(system, public_key):
    if system == "Linux":
        raise ValueError("Official Linux distribution requires a separately packaged AppImage")
    if not public_key:
        raise ValueError("BRIDGECAPTION_UPDATE_PUBLIC_KEY is required for official builds")
    if system == "Windows":
        fingerprint = os.environ.get("BRIDGECAPTION_WINDOWS_CERT_THUMBPRINT", "").replace(" ", "")
        if not re.fullmatch(r"[0-9a-fA-F]{40}", fingerprint):
            raise ValueError("Set BRIDGECAPTION_WINDOWS_CERT_THUMBPRINT to an installed code-signing certificate thumbprint")
        signtool = os.environ.get("BRIDGECAPTION_SIGNTOOL", "") or shutil.which("signtool.exe")
        if not signtool or not Path(signtool).is_file():
            raise ValueError("Install Windows SDK SignTool and set BRIDGECAPTION_SIGNTOOL to signtool.exe")
        timestamp = _https(os.environ.get("BRIDGECAPTION_TIMESTAMP_URL", "https://timestamp.digicert.com"))
        return (signtool, fingerprint, timestamp)
    if system == "Darwin":
        identity = os.environ.get("BRIDGECAPTION_MACOS_SIGNING_IDENTITY", "")
        profile = os.environ.get("BRIDGECAPTION_NOTARY_PROFILE", "")
        if not identity.startswith("Developer ID Application:") or not profile:
            raise ValueError("Set BRIDGECAPTION_MACOS_SIGNING_IDENTITY and an existing BRIDGECAPTION_NOTARY_PROFILE")
        return (identity, profile)
    raise ValueError("The official desktop builder requires Windows or macOS")


def sign_windows(path, settings):
    signtool, fingerprint, timestamp = settings
    subprocess.run([signtool, "sign", "/fd", "SHA256", "/sha1", fingerprint, "/tr", timestamp,
                    "/td", "SHA256", str(path)], check=True, timeout=120)
    subprocess.run([signtool, "verify", "/pa", "/all", "/tw", str(path)], check=True, timeout=120)


def archive_macos(bundle, archive, notary_profile=None):
    def make_zip():
        subprocess.run(["/usr/bin/ditto", "-c", "-k", "--keepParent", str(bundle), str(archive)], check=True)
    subprocess.run(["/usr/bin/codesign", "--verify", "--deep", "--strict", str(bundle)], check=True)
    make_zip()
    if notary_profile:
        submission = subprocess.run(
            ["/usr/bin/xcrun", "notarytool", "submit", str(archive), "--wait",
             "--keychain-profile", notary_profile, "--output-format", "json"],
            check=True, capture_output=True, text=True, timeout=1800,
        )
        result = json.loads(submission.stdout)
        if result.get("status") != "Accepted":
            raise RuntimeError(f"Apple notarization {result.get('status')}: {result.get('id')}")
        subprocess.run(["/usr/bin/xcrun", "stapler", "staple", str(bundle)], check=True)
        subprocess.run(["/usr/bin/xcrun", "stapler", "validate", str(bundle)], check=True)
        archive.unlink()
        make_zip()


def build(version, output, *, official=False):
    version_tuple(version)
    version = version.removeprefix("v")
    target = platform_key()
    system = platform.system()
    public_key = os.environ.get("BRIDGECAPTION_UPDATE_PUBLIC_KEY", "")
    if public_key and len(base64.b64decode(public_key, validate=True)) != 32:
        raise ValueError("Update public key must be a base64-encoded 32-byte Ed25519 key")
    release = official_settings(system, public_key) if official else None
    output.mkdir(parents=True, exist_ok=True)
    build_module = ROOT / "src/bridgecaption/_update_build.py"
    original = build_module.read_bytes()
    build_module.write_text(f"BUILD_VERSION = {version!r}\nUPDATE_PUBLIC_KEY = {public_key!r}\n", encoding="utf-8")
    try:
        with tempfile.TemporaryDirectory(prefix="bridgecaption-build-") as temporary:
            temp = Path(temporary)
            dist = temp / "dist"
            package_stage = temp / "package"
            package_stage.mkdir()
            args = [
                "--name", "BridgeCaption", "--collect-all", "faster_whisper",
                "--collect-all", "ctranslate2", "--collect-all", "_sounddevice_data",
                "--paths", str(ROOT / "src"), "--specpath", str(temp),
                "--windowed",
            ]
            if system == "Darwin":
                args += ["--osx-bundle-identifier", "org.bridgecaption.desktop"]
                if release:
                    args += ["--codesign-identity", release[0], "--osx-entitlements-file",
                             str(ROOT / "packaging/macos/BridgeCaption.entitlements")]
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
                artifact = package_stage / "BridgeCaption-windows-x86_64-setup.exe"
                if release:
                    sign_windows(dist / "BridgeCaption/BridgeCaption.exe", release)
                iscc = shutil.which("ISCC.exe") or str(Path(os.environ["ProgramFiles(x86)"], "Inno Setup 6", "ISCC.exe"))
                subprocess.run([iscc, f"/DBridgeVersion={version}", f"/DOutputDirectory={package_stage}",
                                f"/DDistDirectory={dist}", str(ROOT / "packaging/windows/BridgeCaption.iss")],
                               check=True)
                if release:
                    sign_windows(artifact, release)
            elif system == "Darwin":
                bundle = dist / "BridgeCaption.app"
                artifact = package_stage / f"BridgeCaption-{target}.zip"
                archive_macos(bundle, artifact, release[1] if release else None)
            elif system == "Linux":
                artifact = Path(shutil.make_archive(str(package_stage / f"BridgeCaption-{target}-portable"), "gztar",
                                                   root_dir=dist, base_dir="BridgeCaption"))
            else:
                raise RuntimeError(f"No package format for {target}")
            destination = output / artifact.name
            with artifact.open("rb") as source, destination.open("xb") as target_file:
                shutil.copyfileobj(source, target_file)
    finally:
        build_module.write_bytes(original)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official", action="store_true",
                        help="Sign and verify using existing credentials; notarize macOS. Never publish.")
    args = parser.parse_args()
    build(args.version, args.output.resolve(), official=args.official)
