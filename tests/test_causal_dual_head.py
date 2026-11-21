from hashlib import sha256
import json
from pathlib import Path

import pytest

from research.unified_models import causal_dual_head as probe

PIN = "ba9bb42530cbc70b43edef666922f480e4e275311a8cec5476d1dbab7ac194df"


def observed():
    events = []
    model = probe.CausalProbe(lambda kind, **data: events.append((kind, data)))
    return model, events


def partition(stream, pcm, lengths):
    offset = 0
    for n in lengths:
        stream.push(pcm[offset:offset + n * 2])
        assert 0 <= stream.buffered_samples < probe.FRAME_SAMPLES
        offset += n * 2
    assert offset == len(pcm)
    return stream.finish()


@pytest.mark.parametrize("lengths", [(1287,), (1,) * 1287, (49, 235, 352, 651)])
def test_exact_chunk_invariance_including_final_residue(monkeypatch, lengths):
    calls = []
    original = probe._Encoder.advance

    def observe(self, frame, cache, emit):
        after = original(self, frame, cache, emit)
        calls.append((frame, cache, after))
        return after

    monkeypatch.setattr(probe._Encoder, "advance", observe)
    model, _ = observed()
    pcm = probe._synthetic_pcm(1287, 3)
    whole = partition(model.start("es->en"), pcm, (1287,))
    whole_calls = calls.copy()
    calls.clear()
    split = partition(model.start("es->en"), pcm, lengths)
    assert split == whole
    assert calls == whole_calls  # exact input, incoming and outgoing hidden state
    assert b"".join(frame for frame, _, _ in calls) == pcm
    assert [len(frame) // 2 for frame, _, _ in calls] == [320, 320, 320, 320, 7]
    assert all(old[2] is new[1] for old, new in zip(calls, calls[1:]))


def test_both_heads_receive_the_same_state_object(monkeypatch):
    snapshots = []
    original = probe.CausalProbe._decode

    def observe(self, cache, channel, route, kind):
        snapshots.append((cache, channel, route, kind))
        return original(self, cache, channel, route, kind)

    monkeypatch.setattr(probe.CausalProbe, "_decode", observe)
    model, events = observed()
    for d in probe.DIRECTIONS:
        stream = model.start(d)
        stream.push(probe._synthetic_pcm(320, 5))
        stream.push(probe._synthetic_pcm(319, 8))
        stream.finish()
    assert [kind for kind, _ in events].count("load") == 1
    assert len(snapshots) == 8
    for source, translation in zip(snapshots[::2], snapshots[1::2]):
        assert source[0] is translation[0]
        assert (source[1], translation[1]) == ("source", "translation")
        assert source[2:] == translation[2:]
    assert snapshots[0][0] == snapshots[4][0]  # Same acoustic input, fresh state.
    assert snapshots[0][0] is not snapshots[4][0]


def test_recurrence_uses_prior_frames_and_drafts_cannot_see_suffix(monkeypatch):
    outputs = []
    original = probe._Encoder.advance

    def observe(self, frame, cache, emit):
        state = original(self, frame, cache, emit)
        outputs.append(state.hidden)
        return state

    monkeypatch.setattr(probe._Encoder, "advance", observe)
    m, _ = observed()
    early = probe._synthetic_pcm(320, 7)
    silence = bytes(probe.FRAME_BYTES)
    s = m.start("es->en")
    a = s.push(early)
    s.push(silence)
    s.finish()
    s = m.start("es->en")
    b = s.push(early)
    s.push(probe._synthetic_pcm(320, 11))
    s.finish()
    s = m.start("es->en")
    s.push(silence)
    s.finish()
    assert a == b and not a.final
    assert outputs[1] != outputs[4]  # Same current silence vs different prior audio.


def test_only_subframe_pcm_is_retained_and_routing_is_serial():
    m, _ = observed()
    s = m.start("es->en")
    with pytest.raises(ValueError):
        m.start("fr->en")
    for _ in range(128):
        s.push(bytes(1002))
        assert s.buffered_samples < 320
        assert len(s._cache.hidden) == probe._WIDTH
    end = s.finish()
    assert end.final and end.generation == 201
    assert not s.buffered_samples and not s._cache.generation
    with pytest.raises(ValueError):
        s.finish()
    with pytest.raises(ValueError):
        s.push(b"\0\0")
    with pytest.raises(ValueError):
        m.start("zh->en")
    assert m.start("fr->en").finish() is None


def test_invalid_and_oversized_pcm_fail_before_releasing():
    m, events = observed()
    s = m.start("es->en")
    for pcm in (b"", b"\0", bytearray(2), bytes((probe.MAX_PUSH_SAMPLES + 1) * 2)):
        with pytest.raises(ValueError):
            s.push(pcm)
    assert [kind for kind, _ in events] == ["load", "start"]
    assert s.push(b"\0\0") is None
    assert s.finish().generation == 1


def test_utterance_budget_is_enforced_before_releasing(monkeypatch):
    monkeypatch.setattr(probe, "MAX_UTTERANCE_SAMPLES", 600)
    m, events = observed()
    s = m.start("es->en")
    s.push(bytes(800))
    with pytest.raises(ValueError):
        s.push(bytes(402))
    s.finish()
    assert [e["end"] for kind, e in events if kind == "release"] == [400]


def test_reproducible_redacted_measurement_is_not_benchmark():
    summary = probe.run_probe()
    trace = probe.run_probe(raw_trace=True)
    report = Path(probe.__file__).with_name("causal_dual_head_report.json")
    assert summary == probe.run_probe() == json.loads(report.read_text(encoding="utf-8"))
    assert summary["checkpoint_sha256"] == PIN
    assert summary["method_sha256"] == sha256(Path(probe.__file__).read_bytes()).hexdigest()
    assert summary["model_load_events"] == 1
    assert summary["head_events"] == {"draft": 12, "final": 4}
    assert summary["max_retained_tail_samples"] == 314
    assert trace["events"][0]["checkpoint_sha256"] == PIN
    check = summary["trace_validation"]
    assert check["status"] == "trace_structure_passed"
    assert check["benchmark_passed"] is False
    assert check["model_runtime_verified_by_validator"] is False
    assert check["directions"] == dict.fromkeys(probe.DIRECTIONS, {
        "audio_samples": 1401, "encoder_calls": 5, "utterances": 1})
