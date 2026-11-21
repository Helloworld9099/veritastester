import copy
import hashlib
import json

import pytest

from research.unified_models.incremental_evidence import TraceError, main, validate_trace


PIN = "1" * 64
DIRECTIONS = ["es->en", "fr->en"]


def event(kind, t, **fields):
    return {"kind": kind, "t_ms": t, **fields}


def block(d, offset):
    return [
        event("start", offset, direction=d),
        event("release", offset + 500, end=8000),
        event("encode", offset + 510, start=0, end=8000, cache_in=0, cache_out=1),
        event("draft", offset + 520, channel="source", cache=1),
        event("draft", offset + 530, channel="translation", cache=1),
        event("release", offset + 1000, end=16000),
        event("end", offset + 1000),
        event("encode", offset + 1040, start=8000, end=16000, cache_in=1, cache_out=2),
        event("final", offset + 1050, channel="source", cache=2),
        event("final", offset + 1060, channel="translation", cache=2),
    ]


@pytest.fixture
def trace():
    return {"schema_version": 1, "kind": "bridgecaption_incremental_trace",
            "scope": "open_development", "device": "cpu", "sample_rate": 16000,
            "method_sha256": "a" * 64,
            "events": [event("load", 0, checkpoint_sha256=PIN)] +
            block("es->en", 100) + block("fr->en", 1500)}


def check(trace, directions=DIRECTIONS):
    return validate_trace(trace, expected_checkpoint_sha256=PIN, required_directions=directions)


def test_one_loaded_checkpoint_two_directions(trace):
    result = check(trace)
    assert result["status"] == "trace_structure_passed"
    assert result["benchmark_passed"] is False
    assert result["model_runtime_verified_by_validator"] is False
    assert result["directions"] == dict.fromkeys(DIRECTIONS, {
        "utterances": 1, "encoder_calls": 2, "audio_samples": 16000})


@pytest.mark.parametrize(("index", "patch", "error"), [
    (0, {"checkpoint_sha256": "2" * 64}, "unexpected_checkpoint"),
    (8, {"start": 0}, "overlap_gap_or_future_audio"),
    (8, {"start": 7000}, "overlap_gap_or_future_audio"),
    (8, {"start": 9000}, "overlap_gap_or_future_audio"),
    (8, {"end": 16001}, "overlap_gap_or_future_audio"),
    (8, {"end": 15000}, "final_before_all_audio_is_encoded"),
    (8, {"cache_in": 0}, "encoder_state_not_reused"),
    (8, {"cache_out": 1}, "encoder_state_not_reused"),
    (8, {"cache_out": True}, "encoder_state_not_reused"),
    (5, {"channel": "source"}, "missing_pre_eof_caption_channels"),
    (5, {"cache": 0}, "caption_not_on_current_shared_encoder_state"),
    (5, {"cache": 2}, "caption_not_on_current_shared_encoder_state"),
    (10, {"channel": "source"}, "duplicate_final"),
    (10, {"cache": 1}, "caption_not_on_current_shared_encoder_state"),
    (4, {"t_ms": 0}, "non_monotonic_event_clock"),
    (5, {"transcript": "DO_NOT_ECHO"}, "invalid_event_fields"),
])
def test_negative_traces(trace, index, patch, error):
    trace["events"][index].update(patch)
    with pytest.raises(TraceError, match=f"^{error}$"):
        check(trace)


def test_reload_between_directions_rejected(trace):
    trace["events"].insert(11, event("load", 1200, checkpoint_sha256=PIN))
    with pytest.raises(TraceError, match="checkpoint_reloaded_or_switched"):
        check(trace)


def test_missing_direction_and_unfinished_run(trace):
    reduced = copy.deepcopy(trace)
    reduced["events"] = reduced["events"][:11]
    with pytest.raises(TraceError, match="missing_required_direction"):
        check(reduced)
    trace["events"].pop()
    with pytest.raises(TraceError, match="unfinished_utterance"):
        check(trace)


def test_offline_single_forward_cannot_pass(trace):
    trace["events"] = [trace["events"][0]] + [
        event("start", 100, direction="es->en"),
        event("release", 600, end=8000),
        event("encode", 610, start=0, end=8000, cache_in=0, cache_out=1),
        event("draft", 620, channel="source", cache=1),
        event("draft", 630, channel="translation", cache=1),
        event("end", 700),
        event("final", 705, channel="source", cache=1),
        event("final", 706, channel="translation", cache=1),
    ]
    with pytest.raises(TraceError, match="single_encode_is_not_incremental_evidence"):
        check(trace)


def test_same_source_is_not_a_language_change(trace):
    with pytest.raises(TraceError, match="multiple_distinct_source_languages_required"):
        check(trace, ["es->en", "es->fr"])


def test_cli_pins_bytes_and_redacts(trace, tmp_path, capsys):
    raw = json.dumps(trace).encode()
    path = tmp_path / "trace.json"
    path.write_bytes(raw)
    args = ["--root", str(tmp_path), "--trace", path.name, "--trace-sha256",
            hashlib.sha256(raw).hexdigest(), "--checkpoint-sha256", PIN,
            "--direction", "es->en", "--direction", "fr->en"]
    assert main(args) == 0
    result = capsys.readouterr()
    assert "trace_structure_passed" in result.out
    assert PIN not in result.out
    path.write_bytes(raw + b" ")
    assert main(args) == 2
    assert capsys.readouterr().err == "receipt_raw_sha256_mismatch\n"
