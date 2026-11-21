import json

import pytest

from research.unified_models import nemotron_q8_paced as paced


def event(kind, **fields):
    return {"kind": kind, **fields}


def fixture_events(*, late=False, final_text="hola"):
    tick = 320_000_000
    delayed = 12_000_000 if late else 0
    return [
        event("model", load_ns=10_000, api_version="nemo-speech-asr 0.3.0", cpu_affinity_count=2),
        event("clip_start", clip=0, locale="es-US", samples=10240),
        event("step", clip=0, seq=0, start_sample=0, end_sample=5120, due_ns=tick,
              dispatch_ns=tick+100, push_return_ns=tick+200, done_ns=tick*2+delayed if late else tick+500),
        event("result", clip=0, seq=0, ready_ns=tick+300, final=False,
              processed_samples=5120, text="hola"),
        event("step", clip=0, seq=1, start_sample=5120, end_sample=10240, due_ns=2*tick,
              dispatch_ns=2*tick+delayed+100, push_return_ns=2*tick+delayed+200,
              done_ns=2*tick+delayed+500),
        event("result", clip=0, seq=2, ready_ns=2*tick+delayed+700_000, final=True,
              processed_samples=10240, text=final_text),
        event("clip_done", clip=0, released_samples=10240, steps=2, capture_end_ns=2*tick,
              finish_start_ns=2*tick+delayed+800, finish_done_ns=2*tick+delayed+900_000, final_count=1),
    ]


def run(events, frames=10240):
    raw = "\n".join(json.dumps(e) for e in events).encode()
    return paced._summarize(raw, [("es", 0, bytes(frames*2), "hola")])


def test_redacted_first_final_and_no_backlog():
    receipt = run(fixture_events())
    assert receipt["observed_model_loads"] == 1
    clip = receipt["clips"][0]
    assert clip["max_waiting_audio_ms_at_step_completion"] == 0
    assert clip["first_nonempty_partial_ms_from_clip_start"] == 320.0
    assert clip["final_ms_after_scheduled_eof"] == 0.7
    assert clip["word_errors"] == 0
    assert "hola" not in json.dumps(receipt)


def test_detect_an_entire_ready_cohort_waiting_under_late_decode():
    clip = run(fixture_events(late=True))["clips"][0]
    assert clip["max_waiting_audio_ms_at_step_completion"] == 320
    assert clip["max_dispatch_lateness_ms"] == 12
    assert clip["final_ms_after_scheduled_eof"] == 12.7


@pytest.mark.parametrize(("i", "change", "error"), [
    (0, {"cpu_affinity_count": 1}, "inconsistent_native_model_lifetime"),
    (1, {"locale": "fr-FR"}, "unexpected_paced_clip_start"),
    (2, {"start_sample": 3}, "invalid_paced_release_or_completion"),
    (2, {"dispatch_ns": 319999999}, "invalid_paced_release_or_completion"),
    (3, {"ready_ns": 319999999}, "invalid_paced_partial"),
    (5, {"processed_samples": 10239}, "invalid_paced_final"),
    (6, {"released_samples": 7000}, "invalid_paced_clip_done"),
])
def test_pacing_fails_closed_on_bad_events(i, change, error):
    events = fixture_events()
    events[i].update(change)
    with pytest.raises(paced.DiagnosticError, match=error):
        run(events)


def test_missing_final_is_not_treated_as_zero():
    with pytest.raises(paced.DiagnosticError, match="invalid_paced_clip_done"):
        run(fixture_events()[:-2] + fixture_events()[-1:])


def test_missing_partial_is_reported_explicitly():
    events = fixture_events()
    del events[3]
    assert run(events)["clips"][0]["first_nonempty_partial_ms_from_clip_start"] is None


def test_native_decode_frontier_is_not_the_submission_frontier():
    events = fixture_events()
    events[3]["processed_samples"] = 2560
    clip = run(events)["clips"][0]
    assert clip["max_native_unreported_progress_ms_when_partial_emits"] == 160
    assert clip["max_waiting_audio_ms_at_step_completion"] == 0


def test_native_partial_cannot_claim_unreleased_samples():
    events = fixture_events()
    events[3]["processed_samples"] = 5121
    with pytest.raises(paced.DiagnosticError, match="invalid_paced_partial"):
        run(events)
