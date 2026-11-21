"""Headless app-controller baseline smoke: no Tk rendering, network, or capture device."""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from bridgecaption import app
from bridgecaption.audio import RATE
from bridgecaption.pipeline import Pipeline
from bridgecaption.provider import TranslationRequest
from bridgecaption.providers import GeminiTranslator
from bridgecaption.session import LiveSession


class Value:
    """Minimal Tk variable double, without a Tcl interpreter or display."""

    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


@pytest.fixture
def desktop_shell(monkeypatch, tmp_path):
    # Discard all provider variables the research machine might otherwise hold.
    for name in ("BRIDGE_TRANSLATOR", "GEMINI_API_KEY", "GOOGLE_API_KEY",
                 "GROQ_API_KEY", "AZURE_TRANSLATOR_KEY", "AZURE_TRANSLATOR_REGION",
                 "BRIDGE_OVERLAY_OPACITY", "AUDIO_CHUNK_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    denied = MagicMock(side_effect=AssertionError("offline smoke cannot connect or launch a process"))
    monkeypatch.setattr("socket.socket.connect", denied)
    monkeypatch.setattr("socket.create_connection", denied)
    monkeypatch.setattr("subprocess.Popen", denied)
    monkeypatch.setattr(app, "load_dotenv", MagicMock())  # Do not read a real .env.
    enumeration = MagicMock(return_value=[("Synthetic fixture", "test-only-source")])
    monkeypatch.setattr(app, "sources", enumeration)
    for key in ("BooleanVar", "DoubleVar", "StringVar"):
        monkeypatch.setattr(app.tk, key, Value)

    live = []
    root = MagicMock()
    monkeypatch.setattr(app.tk, "Tk", MagicMock(return_value=root))

    def build_double_widgets(instance):
        live.append(instance)
        for field in ("status_var", "summary_var", "opacity_label"):
            setattr(instance, field, Value())
        for field in ("source_text", "translated_text", "start_button", "source_box",
                      "source_picker", "target_picker", "topic_box", "engine_note", "test_text",
                      "live_indicator"):
            setattr(instance, field, MagicMock())
        instance.source_box.current.return_value = 0
        instance.source_picker.get.return_value = "Spanish [es]"
        instance.target_picker.get.return_value = "English [en]"
        instance.topic_box.get.return_value = "general"
        instance.test_text.get.return_value = "Hola, texto escrito."

    monkeypatch.setattr(app.Application, "_ui", build_double_widgets)
    monkeypatch.setattr(app.Application, "_apply_overlay_mode", MagicMock())
    monkeypatch.setattr(app.Application, "_apply_opacity", MagicMock())
    consent = MagicMock(return_value=False)
    monkeypatch.setattr(app.messagebox, "askokcancel", consent)
    yield root, live, consent, enumeration
    for instance in live:
        if instance.live_session is not None:
            instance.live_session.stop()
            deadline = time.monotonic() + 3
            while instance.live_session is not None and time.monotonic() < deadline:
                instance.poll()
                time.sleep(.01)
        if not instance.closed:
            instance.close()
    denied.assert_not_called()


def _poll_until(instance, predicate):
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        instance.poll()
        if predicate():
            return
        time.sleep(.01)
    pytest.fail("Headless app did not receive the expected event")


def test_application_entrypoint_is_idle_and_fails_closed_without_provider_key(desktop_shell):
    root, live, consent, enumeration = desktop_shell
    app.main()
    root.mainloop.assert_called_once_with()
    enumeration.assert_called_once_with()
    assert len(live) == 1
    instance = live[0]
    assert isinstance(instance.pipeline, Pipeline)
    assert isinstance(instance.pipeline.translator, GeminiTranslator)
    assert instance.engine_selected == "gemini"
    assert instance.pipeline.translator.key == ""
    assert instance.worker is None and instance.live_session is None
    assert instance.compact and instance.context_enabled.get()
    consent.assert_not_called()

    instance.test_translate()
    assert "Configure the key for gemini" in instance.status_var.get()
    instance.toggle_capture()
    assert "Configure the API key for gemini" in instance.status_var.get()
    instance.check_engine()
    assert "Configure the gemini key" in instance.status_var.get()
    assert instance.worker is None and instance.live_session is None
    consent.assert_not_called()
    instance.close()
    root.destroy.assert_called_once_with()


def test_consent_typed_and_synthetic_listening_use_original_application_pipeline(
    desktop_shell, monkeypatch,
):
    root, live, consent, _ = desktop_shell
    local_pcm = b"\x00\x20" * RATE  # Non-silent 16 kHz signed 16-bit little-endian PCM.
    instances = []
    audio_started = threading.Event()
    stopped = threading.Event()
    actual_asr_inputs = []

    class FakeCapture:
        def __init__(self, source, seconds):
            assert (source, seconds) == ("test-only-source", 4)
            self.finished = threading.Event()

        def start(self):
            audio_started.set()
            return self

        def chunks(self):
            yield local_pcm
            self.finished.wait(6)

        def stop(self):
            self.finished.set()
            stopped.set()

    class FakeASR:
        def transcribe(self, pcm, requested_language):
            actual_asr_inputs.append((len(pcm), requested_language))
            assert pcm == local_pcm
            return "Hola, cambiamos a las seis.", "es"

    class FakeProvider:
        supports_context = True

        def __init__(self):
            self.requests = []

        def translate(self, request):
            assert isinstance(request, TranslationRequest)
            assert not any(isinstance(v, bytes) for v in vars(request).values())
            self.requests.append(request)
            return {
                "Hola, texto escrito.": "Hello, typed text.",
                "Hola, cambiamos a las seis.": "Hi, we moved it to six.",
            }[request.text]

    def fake_factory(engine):
        assert engine == "gemini"
        mock = FakeProvider()
        instances.append(mock)
        return mock

    def real_session_with_fakes(**kwargs):
        return LiveSession(**kwargs, audio_factory=FakeCapture, recognizer_factory=FakeASR)

    monkeypatch.setattr(app, "create_translator", fake_factory)
    # Mimics the boolean readiness gate; no key is created or read.
    monkeypatch.setattr(app, "key_available", lambda name: name == "gemini")
    monkeypatch.setattr(app, "LiveSession", real_session_with_fakes)
    app.main()
    instance = live[0]
    assert not audio_started.is_set()
    instance.toggle_capture()
    consent.assert_called_once()
    assert instance.live_session is None and not audio_started.is_set()
    assert not instances[0].requests

    consent.return_value = True
    instance.test_translate()
    _poll_until(instance, lambda: instance.last_caption is not None)
    assert instance.last_caption == ("Hola, texto escrito.", "Hello, typed text.", "es")
    typed_request, = instances[0].requests
    assert (typed_request.source, typed_request.target, tuple(typed_request.previous)) == ("es", "en", ())
    instance.worker.join(timeout=1)
    assert not instance.worker.is_alive()

    instance.source_picker.get.return_value = "Auto detect [auto]"
    instance.toggle_capture()
    assert isinstance(instance.live_session, LiveSession)
    _poll_until(instance, lambda: instance.last_caption ==
                ("Hola, cambiamos a las seis.", "Hi, we moved it to six.", "es"))
    assert actual_asr_inputs == [(len(local_pcm), "auto")]
    assert audio_started.is_set() and len(instances) == 2
    live_request, = instances[1].requests
    assert (live_request.source, live_request.target, tuple(live_request.previous)) == ("es", "en", ())
    assert instance.translated_text.config.call_args.kwargs["text"] == "Hi, we moved it to six."
    assert consent.call_count == 2  # Denied capture, then granted typed consent for same provider.

    instance.toggle_capture()
    _poll_until(instance, lambda: instance.live_session is None)
    assert stopped.is_set() and instance.last_caption is None
    assert not instance.pipeline.history.snapshot()
    assert instance.start_button.config.call_args.kwargs["text"] == "Start listening"
    instance.close()
    root.destroy.assert_called_once_with()
