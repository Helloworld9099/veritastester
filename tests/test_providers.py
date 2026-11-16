import pytest
from urllib.error import HTTPError

from bridgecaption import providers
from bridgecaption.provider import TranslationRequest


def request():
    return TranslationRequest("Hola!", source="es", target="en",
                              previous=["private meeting"], topic="business")


def test_provider_factory_cloud_boundaries(monkeypatch):
    monkeypatch.setenv("BRIDGE_TRANSLATOR", "ollama")
    assert isinstance(providers.create_translator(), providers.OllamaTranslator)
    assert providers.needs_cloud("azure")
    assert not providers.needs_cloud("ollama")
    with pytest.raises(ValueError):
        providers.create_translator("unknown")


def test_groq_request_schema(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "local-test-key")
    captured = {}
    def fake(url, payload, headers, timeout):
        captured.update(url=url, payload=payload)
        return {"choices": [{"message": {"content": "Hello!"}}]}
    monkeypatch.setattr(providers, "request_json", fake)
    assert providers.GroqTranslator().translate(request()) == "Hello!"
    assert captured["url"].startswith("https://api.groq.com/")
    assert captured["payload"]["messages"][0]["role"] == "system"
    assert "private meeting" in captured["payload"]["messages"][1]["content"]


def test_ollama_never_uses_external_endpoint(monkeypatch):
    captured = {}
    def fake(url, payload, headers, timeout):
        captured.update(url=url, payload=payload)
        return {"message": {"content": "Hello!"}}
    monkeypatch.setattr(providers, "request_json", fake)
    assert providers.OllamaTranslator().translate(request()) == "Hello!"
    assert captured["url"].startswith("http://127.0.0.1:")
    assert captured["payload"]["think"] is False


def test_azure_baseline_does_not_send_context(monkeypatch):
    monkeypatch.setenv("AZURE_TRANSLATOR_KEY", "local-test-key")
    captured = {}
    def fake(url, payload, headers, timeout):
        captured.update(url=url, payload=payload)
        return [{"translations": [{"text": "Hello!"}]}]
    monkeypatch.setattr(providers, "request_json", fake)
    assert providers.AzureTranslator().translate(request()) == "Hello!"
    assert captured["payload"] == [{"Text": "Hola!"}]
    assert "private meeting" not in str(captured)


def test_error_does_not_leak_secret(monkeypatch):
    def fail(*args, **kwargs):
        raise HTTPError("https://example.test/?key=secret123", 429,
                        "secret123", {}, None)
    monkeypatch.setattr(providers, "urlopen", fail)
    with pytest.raises(RuntimeError, match="429") as exc:
        providers.request_json("https://api.groq.com/openai/v1/chat/completions",
                               {}, {}, 4)
    assert "secret123" not in str(exc.value)


def test_timeout_range(monkeypatch):
    monkeypatch.setenv("TRANSLATION_TIMEOUT_SECONDS", "60")
    with pytest.raises(ValueError):
        providers.request_timeout()

def test_request_echo_is_not_mistaken_for_translation():
    import json
    text = json.dumps({
        "source_language": "es", "target_language": "en",
        "current_transcript": "Hola", "optional_context": {}})
    with pytest.raises(RuntimeError, match="repeated the request"):
        providers.translation_text(text)
    with pytest.raises(RuntimeError, match="repeated the request"):
        providers.message_content({"choices": [{"message": {"content": text}}]})
    assert providers.translation_text('{"Hello": "world"}') == '{"Hello": "world"}'
    assert providers.translation_text("Hello") == "Hello"
