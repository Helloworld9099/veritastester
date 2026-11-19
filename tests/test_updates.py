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
