import pytest

from bridgecaption.diagnostics import check_provider


class Translator:
    def __init__(self):
        self.calls = []

    def translate(self, req):
        self.calls.append(req)
        return "Hello, good morning."


def test_check_provider_uses_only_public_phrase(monkeypatch):
    monkeypatch.setenv("AZURE_TRANSLATOR_KEY", "dummy-test-key")
    model = Translator()
    report = check_provider("azure", factory=lambda name: model)
    assert report["engine"] == "azure"
    assert report["synthetic_only"] is True
    assert report["destination"] == "cloud"
    assert model.calls[0].text == "Hola, buenos días."
    assert model.calls[0].context_enabled is False
    assert model.calls[0].previous == []


def test_missing_credentials_do_not_expose_value(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Missing groq credential"):
        check_provider("groq", factory=lambda _: Translator())


def test_unrecognized_provider_never_calls_factory():
    with pytest.raises(ValueError, match="Unsupported"):
        check_provider("invalid", factory=lambda name: (_ for _ in ()).throw(
            AssertionError("must not create provider")))


def test_cli_dry_run_does_not_contact_provider(monkeypatch, capsys):
    import sys
    from bridgecaption import diagnostics
    monkeypatch.setattr(sys, "argv", ["bridgecaption-check", "--engine", "azure"])
    monkeypatch.setattr(diagnostics, "check_provider", lambda *_: (_ for _ in ()).throw(
        AssertionError("dry run must not call remote service")))
    diagnostics.main()
    assert "DRY RUN" in capsys.readouterr().out

def test_cli_success_is_human_readable(monkeypatch, capsys):
    import sys
    from bridgecaption import diagnostics
    monkeypatch.setattr(sys, "argv", [
        "bridgecaption-check", "--engine", "azure", "--execute"])
    monkeypatch.setattr(diagnostics, "check_provider", lambda *_: {
        "engine": "azure", "destination": "cloud",
        "result": "Hello, good morning.", "elapsed_seconds": 0.25,
        "synthetic_only": True})
    diagnostics.main()
    output = capsys.readouterr().out
    assert "PASS" in output
    assert "Hello, good morning." in output
    assert not output.lstrip().startswith("{")

def test_cli_json_remains_opt_in(monkeypatch, capsys):
    import json
    import sys
    from bridgecaption import diagnostics
    monkeypatch.setattr(sys, "argv", [
        "bridgecaption-check", "--engine", "azure", "--execute", "--json"])
    monkeypatch.setattr(diagnostics, "check_provider", lambda *_: {
        "engine": "azure", "destination": "cloud",
        "result": "Hello", "elapsed_seconds": 0.25, "synthetic_only": True})
    diagnostics.main()
    assert json.loads(capsys.readouterr().out)["result"] == "Hello"
