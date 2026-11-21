import hashlib
import json

import pytest

from research.unified_models.streaming import (
    PCM_FORMAT, CaptionUpdate, PCMFrame, SingleStreamingSpeechRuntime, SpeechQualification,
    SpeechStream, StreamError, StreamLimits,
)
from research.unified_models.streaming_app_bridge import CaptionEventBridge


@pytest.fixture
def research_stream():
    checkpoint = "a" * 64
    receipt = {"schema_version": 1, "kind": "reviewed_paired_streaming_speech_dev",
               "scope": "research_only", "checkpoint_sha256": checkpoint,
               "pcm_format": PCM_FORMAT, "shared_audio_encoder": True,
               "directions": {"es->en": {"status": "passed", "evidence_sha256": "b" * 64},
                              "fr->en": {"status": "passed", "evidence_sha256": "c" * 64}}}
    pin = hashlib.sha256(json.dumps(receipt, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":")).encode()).hexdigest()
    qualification = SpeechQualification(receipt, expected_sha256=pin)
    calls = []

    class Decoder:
        def __init__(self, source):
            self.source = source
            self.count = 0
        def push(self, _pcm):
            self.count += 1
            calls.append(self.source)
            return [CaptionUpdate(0, "primer texto", "first text", False)] if self.count == 1 else []
        def finish(self):
            return [CaptionUpdate(0, "texto final", "final text", True)] if self.count else []
        def abort(self):
            calls.append("aborted")
    class Backend:
        checkpoint_sha256 = checkpoint
        def open_stream(self, *, source, target):
            return Decoder(source)
    runtime = SingleStreamingSpeechRuntime(qualification=qualification,
                                           load_backend=lambda _: Backend(), opt_in=True)
    return runtime, calls


def test_provisional_stays_separate_and_only_final_matches_app_event_shape(research_stream):
    runtime, _ = research_stream
    final, provisional = [], []
    clock = lambda: 1_150_000_000
    bridge = CaptionEventBridge(runtime.start_stream(source="es", target="en"),
                                on_event=lambda *event: final.append(event),
                                on_provisional=provisional.append, clock_ns=clock)
    bridge.push(PCMFrame(0, b"\0\0"), frame_end_ns=1_000_000_000)
    assert not final
    assert len(provisional) == 1 and provisional[0].translated == "first text"
    assert "first text" not in repr(provisional[0])
    bridge.finish()
    assert final == [("caption", "texto final", "final text", "es", 0.15,
                      "research streaming; context not verified")]
    with pytest.raises(ValueError, match="closed"):
        bridge.finish()


def test_time_validation_happens_before_decoder(research_stream):
    runtime, calls = research_stream
    bridge = CaptionEventBridge(runtime.start_stream(source="es", target="en"),
                                on_event=lambda *event: None, clock_ns=lambda: 3000)
    with pytest.raises(ValueError, match="Frame-end"):
        bridge.push(PCMFrame(0, b"\0\0"), frame_end_ns=3001)
    assert not calls
    bridge.push(PCMFrame(0, b"\0\0"), frame_end_ns=2000)
    with pytest.raises(ValueError, match="Frame-end"):
        bridge.push(PCMFrame(1, b"\0\0"), frame_end_ns=1999)
    assert calls == ["es"]
    bridge.abort()


def test_bad_callback_aborts_and_redacts_error_without_fallback(research_stream):
    runtime, calls = research_stream
    def bad(_):
        raise RuntimeError("a private transcript should not get into the error")
    stream = runtime.start_stream(source="fr", target="en")
    bridge = CaptionEventBridge(stream, on_event=lambda *ev: None,
                                on_provisional=bad, clock_ns=lambda: 3000)
    with pytest.raises(ValueError, match="without fallback") as exc:
        bridge.push(PCMFrame(0, b"\0\0"), frame_end_ns=1000)
    assert "private transcript" not in str(exc.value)
    assert calls == ["fr", "aborted"]
    assert stream.state == "aborted"
    assert runtime.loaded_model_count == 1


def test_requires_an_open_research_stream_and_binds_source_from_it(research_stream):
    runtime, _ = research_stream
    with pytest.raises(ValueError, match="admitted"):
        CaptionEventBridge(object(), on_event=lambda *ev: None)
    stream = runtime.start_stream(source="fr", target="en")
    stream.abort()
    with pytest.raises(ValueError, match="admitted"):
        CaptionEventBridge(stream, on_event=lambda *ev: None)
    events = []
    bridge = CaptionEventBridge(runtime.start_stream(source="fr", target="en"),
                                on_event=lambda *event: events.append(event), clock_ns=lambda: 1000)
    bridge.push(PCMFrame(0, b"\0\0"), frame_end_ns=1000)
    bridge.finish()
    assert events[0][3] == "fr" and runtime.loaded_model_count == 1


def test_reentrant_callback_cannot_push_more_audio(research_stream):
    runtime, calls = research_stream
    stream = runtime.start_stream(source="es", target="en")
    def provisional(_):
        with pytest.raises(ValueError, match="re-enter"):
            bridge.push(PCMFrame(1, b"\0\0"), frame_end_ns=1000)
    bridge = CaptionEventBridge(stream, on_event=lambda *event: None,
                                on_provisional=provisional, clock_ns=lambda: 1000)
    bridge.push(PCMFrame(0, b"\0\0"), frame_end_ns=1000)
    assert calls == ["es"]
    bridge.finish()


def test_empty_stream_cannot_flush_hallucinated_output_and_direct_construction_is_blocked(research_stream):
    from threading import RLock
    runtime, calls = research_stream
    with pytest.raises(StreamError, match="qualified runtime"):
        SpeechStream(decoder=object(), source="es", target="en", lock=RLock(),
                     release=lambda _: None, limits=StreamLimits())
    stream = runtime.start_stream(source="es", target="en")
    events = []
    bridge = CaptionEventBridge(stream, on_event=lambda *event: events.append(event), clock_ns=lambda: 1000)
    bridge.finish()
    assert stream.state == "finished"
    assert not events and calls == ["aborted"]


def test_unverified_locale_cannot_fall_back_to_base_language(research_stream):
    runtime, _ = research_stream
    with pytest.raises(StreamError, match="unverified locale"):
        runtime.start_stream(source="es-MX", target="en")
    assert runtime.loaded_model_count == 0
