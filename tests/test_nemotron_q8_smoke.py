import io
import json
import wave

import pytest

from research.unified_models import nemotron_q8_smoke as q8


def sample():
    return {"mode": "stream", "chunk_ms": 320, "files": 1, "corpus_audio_seconds": 6,
            "runs": [{"concurrency": 1, "items": 2, "transcript_mismatches": 0,
                      "latency_ms": {"mean": 2400.1234}}]}


def test_bench_accepts_only_expected_streaming_receipt():
    assert q8._read_result(json.dumps(sample()), 6.0, 2) == 2400.123
    s = sample()
    s["mode"] = "offline"
    with pytest.raises(q8.DiagnosticError, match="native_bench_protocol_mismatch"):
        q8._read_result(json.dumps(s), 6.0, 2)


@pytest.mark.parametrize(("section", "field", "value", "error"), [
    ("top", "chunk_ms", 160, "native_bench_protocol_mismatch"),
    ("top", "corpus_audio_seconds", 10, "native_bench_measurement_invalid"),
    ("run", "items", 1, "native_bench_protocol_mismatch"),
    ("run", "transcript_mismatches", 1, "native_bench_protocol_mismatch"),
    ("time", "mean", float("nan"), "native_bench_measurement_invalid"),
])
def test_bench_rejects_incompatible_or_inconsistent_result(section, field, value, error):
    data = sample()
    target = {"top": data, "run": data["runs"][0], "time": data["runs"][0]["latency_ms"]}
    target[section][field] = value
    with pytest.raises(q8.DiagnosticError, match=error):
        q8._read_result(json.dumps(data), 6.0, 2)


def test_canonical_wav_is_pcm16_without_conversion():
    pcm = bytes((0, 1, 2, 3, 4, 5))
    result = q8._canonical_wav(pcm)
    with wave.open(io.BytesIO(result)) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        assert wav.readframes(3) == pcm


def test_pin_rejects_mismatch_before_using_artifact(tmp_path):
    p = tmp_path / "fake"
    p.write_bytes(b"not the native model")
    with pytest.raises(q8.DiagnosticError, match="artifact_pin_mismatch"):
        q8._pin(p, q8.Q8_PIN, 100)
