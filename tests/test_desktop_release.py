import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_desktop.py"


def test_release_preflight_requires_trust_root(monkeypatch, tmp_path):
    monkeypatch.delenv("BRIDGECAPTION_UPDATE_PUBLIC_KEY", raising=False)
    build = runpy.run_path(str(SCRIPT))["build"]
    monkeypatch.setattr(build.__globals__["platform"], "system", lambda: "Darwin")
    output = tmp_path / "release"
    with pytest.raises(ValueError, match="UPDATE_PUBLIC_KEY"):
        build("0.2.0", output, official=True)
    assert not output.exists()


def test_notarization_staples_before_final_archive(monkeypatch, tmp_path):
    archive_macos = runpy.run_path(str(SCRIPT))["archive_macos"]
    runner = archive_macos.__globals__["subprocess"]
    archive = tmp_path / "app.zip"
    app = tmp_path / "BridgeCaption.app"
    events = []

    def run(args, **kwargs):
        if args[0] == "/usr/bin/codesign":
            events.append("verify")
        elif args[0] == "/usr/bin/ditto":
            events.append("zip")
            archive.write_bytes(b"stapled" if "staple" in events else b"signed")
        elif args[1] == "notarytool":
            assert archive.read_bytes() == b"signed"
            events.append("notarize")
            return SimpleNamespace(stdout=json.dumps({"id": "test", "status": "Accepted"}))
        else:
            events.append(args[2])

    monkeypatch.setattr(runner, "run", run)
    archive_macos(app, archive, "test-profile")
    assert events == ["verify", "zip", "notarize", "staple", "validate", "zip"]
    assert archive.read_bytes() == b"stapled"


def test_windows_signer_requires_trusted_signature_verification(monkeypatch, tmp_path):
    sign_windows = runpy.run_path(str(SCRIPT))["sign_windows"]
    runner = sign_windows.__globals__["subprocess"]
    calls = []
    monkeypatch.setattr(runner, "run", lambda args, **kw: calls.append((args, kw)))
    sign_windows(tmp_path / "BridgeCaption.exe", ("signtool.exe", "a" * 40, "https://example.com"))
    assert calls[0][0][:4] == ["signtool.exe", "sign", "/fd", "SHA256"]
    assert calls[1][0][:5] == ["signtool.exe", "verify", "/pa", "/all", "/tw"]
    assert all(kwargs["check"] for _, kwargs in calls)
