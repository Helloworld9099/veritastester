import pytest

from bridgecaption.updates import current_version, version_tuple, find_update


def test_version_comparison():
    assert version_tuple("v0.2.0") > version_tuple("0.1.9")
    assert version_tuple("1.10.0") > version_tuple("1.9.9")


@pytest.mark.parametrize("invalid", ["", "v1.2.beta", "1.2.3-rc1", "../bad", "1.2"])
def test_reject_invalid_versions(invalid):
    with pytest.raises(ValueError):
        version_tuple(invalid)


def test_reject_insecure_endpoint(monkeypatch):
    monkeypatch.setenv("BRIDGECAPTION_UPDATE_ENDPOINT", "http://example.com/latest")
    with pytest.raises(ValueError, match="HTTPS"):
        find_update("0.1.0")


def test_current_version_is_valid():
    assert version_tuple(current_version())


def test_platform_aliases():
    from bridgecaption.updates import platform_key
    assert platform_key("Windows", "AMD64") == "windows-x86_64"
    assert platform_key("Darwin", "aarch64") == "darwin-arm64"


def test_signed_manifest_and_verified_download(monkeypatch):
    import base64
    import hashlib
    import io
    import json
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    from bridgecaption import updates

    content = b"native-package"
    key = Ed25519PrivateKey.generate()
    monkeypatch.setattr(updates, "UPDATE_PUBLIC_KEY",
                        base64.b64encode(key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode())
    monkeypatch.setattr(updates, "platform_key", lambda: "windows-x86_64")
    payload = {"version": "0.2.0", "artifacts": {"windows-x86_64": {
        "url": "https://example.com/artifact", "sha256": hashlib.sha256(content).hexdigest(),
        "size": len(content)}}}
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    document = {"payload": payload, "signature": base64.b64encode(key.sign(data)).decode()}
    monkeypatch.setattr(updates, "urlopen", lambda request, timeout: io.BytesIO(json.dumps(document).encode()))
    update = find_update("0.1.0")
    assert update is not None and update.platform == "windows-x86_64"
    assert find_update("0.2.0") is None
    payload["version"] = "0.3.0"
    with pytest.raises(InvalidSignature):
        find_update("0.1.0")
    monkeypatch.setattr(updates, "urlopen", lambda request, timeout: io.BytesIO(content))
    path = updates.download_update(update, lambda done, total: None)
    try:
        assert path.suffix == ".exe"
        assert path.read_bytes() == content
    finally:
        updates.discard_update(path)
    assert not path.parent.exists()
    monkeypatch.setattr(updates, "urlopen", lambda request, timeout: io.BytesIO(b"x" * len(content)))
    with pytest.raises(ValueError, match="integrity"):
        updates.download_update(update, lambda done, total: None)


def test_redirect_cannot_downgrade_to_http():
    from bridgecaption.updates import HTTPSOnlyRedirect
    from urllib.request import Request
    with pytest.raises(ValueError, match="HTTPS"):
        HTTPSOnlyRedirect().redirect_request(Request("https://example.com"), None, 302, "", {}, "http://example.com")
