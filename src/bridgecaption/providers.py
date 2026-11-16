"""Explicit provider selection; never silently change transcript destinations.

Gemini, Groq and Ollama accept contextual prompts. Azure is a conventional,
context-free translation baseline, not a reasoning model.
"""
from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .provider import (GeminiTranslator, TranslationRequest, SYSTEM_TRANSLATION_RULES,
                       prompt_for)

PROVIDERS = ("ollama", "groq", "azure", "gemini")
CLOUD_PROVIDERS = {"groq", "azure", "gemini"}


def needs_cloud(provider: str) -> bool:
    return provider in CLOUD_PROVIDERS


def key_available(provider: str) -> bool:
    if provider not in PROVIDERS:
        raise ValueError("Unsupported translation provider")
    key = {"groq": "GROQ_API_KEY", "azure": "AZURE_TRANSLATOR_KEY",
           "gemini": "GEMINI_API_KEY"}.get(provider)
    return not key or bool(os.getenv(key, ""))


def request_json(url: str, payload: object, headers: dict[str, str], timeout: float):
    """Bound response size and suppress secret/transcript text in errors."""
    req = Request(url, json.dumps(payload, ensure_ascii=False).encode("utf-8"), method="POST",
                  headers={"Content-Type": "application/json", **headers})
    try:
        with urlopen(req, timeout=timeout) as response:
            raw = response.read(256_001)
    except HTTPError as exc:
        raise RuntimeError(f"Translation API returned HTTP {exc.code}; check model, account and quota") from None
    except (URLError, TimeoutError, OSError):
        raise RuntimeError("Translation API is unreachable") from None
    if len(raw) > 256_000:
        raise RuntimeError("Translation response exceeded size limit")
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        raise RuntimeError("Translation API returned invalid JSON") from None


def request_timeout() -> float:
    try:
        seconds = float(os.getenv("TRANSLATION_TIMEOUT_SECONDS", "15"))
    except ValueError:
        raise ValueError("TRANSLATION_TIMEOUT_SECONDS must be a number") from None
    if not 3 <= seconds <= 30:
        raise ValueError("TRANSLATION_TIMEOUT_SECONDS must be between 3 and 30")
    return seconds


def translation_text(text: str) -> str:
    """Reject a model echoing our structured request instead of translating."""
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("The translation model returned empty content")
    result = text.strip()
    candidate = result
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(candidate)
    except (ValueError, TypeError):
        return result
    if isinstance(parsed, dict) and (
            {"current_transcript", "source_language", "target_language"} <= parsed.keys()
            or "optional_context" in parsed and "current_transcript" in parsed):
        raise RuntimeError(
            "The model repeated the request data instead of translating. "
            "Try another model or engine.")
    return result


def message_content(body: dict) -> str:
    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("No translation was returned by the model") from None
    return translation_text(text)


class GroqTranslator:
    supports_context = True

    def __init__(self):
        self.key = os.getenv("GROQ_API_KEY", "")
        self.model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    def translate(self, request: TranslationRequest) -> str:
        if not self.key:
            raise RuntimeError("GROQ_API_KEY is not configured")
        payload = {"model": self.model,
                   "messages": [{"role": "system", "content": SYSTEM_TRANSLATION_RULES},
                                {"role": "user", "content": prompt_for(request)}],
                   "stream": False, "max_completion_tokens": 512}
        body = request_json("https://api.groq.com/openai/v1/chat/completions", payload,
                            {"Authorization": f"Bearer {self.key}"}, request_timeout())
        return message_content(body)


class OllamaTranslator:
    supports_context = True

    def __init__(self):
        self.model = os.getenv("OLLAMA_MODEL", "qwen3:4b")

    def translate(self, request: TranslationRequest) -> str:
        payload = {"model": self.model, "stream": False, "think": False,
                   "messages": [{"role": "system", "content": SYSTEM_TRANSLATION_RULES},
                                {"role": "user", "content": prompt_for(request)}],
                   "options": {"temperature": 0.1, "num_predict": 512}}
        body = request_json("http://127.0.0.1:11434/api/chat", payload, {}, request_timeout())
        try:
            content = body["message"]["content"]
            if content and content.strip():
                return translation_text(content)
        except (KeyError, TypeError, AttributeError):
            pass
        raise RuntimeError(f"Ollama returned no translation; is '{self.model}' installed?")


class AzureTranslator:
    """Text-only baseline; previous utterances and glossary are NOT sent."""
    supports_context = False

    def __init__(self):
        self.key = os.getenv("AZURE_TRANSLATOR_KEY", "")
        self.region = os.getenv("AZURE_TRANSLATOR_REGION", "")

    def translate(self, request: TranslationRequest) -> str:
        if not self.key:
            raise RuntimeError("AZURE_TRANSLATOR_KEY is not configured")
        if not request.target:
            raise ValueError("Target language required")
        # Azure uses zh-Hans and nb where Whisper may return zh/no.
        language_map = {"zh": "zh-Hans", "no": "nb"}
        source = language_map.get(request.source, request.source)
        target = language_map.get(request.target, request.target)
        params = {"api-version": "3.0", "to": target}
        if source != "auto":
            params["from"] = source
        headers = {"Ocp-Apim-Subscription-Key": self.key}
        if self.region:
            headers["Ocp-Apim-Subscription-Region"] = self.region
        body = request_json(
            "https://api.cognitive.microsofttranslator.com/translate?" + urlencode(params),
            [{"Text": request.text[:1500]}], headers, request_timeout())
        try:
            value = body[0]["translations"][0]["text"]
            if isinstance(value, str) and value.strip():
                return value.strip()
        except (KeyError, IndexError, TypeError):
            pass
        raise RuntimeError("Azure returned no translated text")


def create_translator(name: str | None = None):
    name = (name or os.getenv("BRIDGE_TRANSLATOR", "ollama")).lower().strip()
    if name == "ollama":
        return OllamaTranslator()
    if name == "groq":
        return GroqTranslator()
    if name == "azure":
        return AzureTranslator()
    if name == "gemini":
        return GeminiTranslator()
    raise ValueError(f"Unknown provider {name!r}; choose {', '.join(PROVIDERS)}")
