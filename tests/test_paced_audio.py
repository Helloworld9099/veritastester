import pytest

from research.unified_models.paced_audio import PacePolicy, run_paced_prefix


SECOND = 1_000_000_000
RATE = 16_000


class ManualClock:
    def __init__(self):
        self.ns = 7 * SECOND

    def read(self):
        return self.ns

    def sleep(self, seconds):
        self.ns += round(seconds * SECOND)

    def advance(self, nanoseconds):
        self.ns += nanoseconds


def test_absolute_cadence_never_decodes_future_audio_and_reports_real_timing():
    time = ManualClock()
    started = []

    def decode(prefix, final):
        assert prefix <= (time.ns - 7 * SECOND) * RATE // SECOND
        started.append((time.ns, prefix, final))
        time.advance(SECOND // 4)
        return "PRIVATE"

    run = run_paced_prefix(3 * RATE, PacePolicy(RATE), decode,
                           clock=time.read, sleep=time.sleep)
    assert started == [(8 * SECOND, RATE, False), (9 * SECOND, 2 * RATE, False),
                       (10 * SECOND, 3 * RATE, True)]
    assert [(e.scheduled_ns, e.start_ns, e.finish_ns) for e in run.events] == [
        (s, s, s + SECOND // 4) for s, _, _ in started]
    assert all(e.emitted and e.disposition == "available" for e in run.events)
    summary = run.timing_summary()
    assert summary["first_provisional_from_start_ns"] == 5 * SECOND // 4
    assert summary["final_from_capture_end_ns"] == SECOND // 4
    assert summary["provisionals_available"] == 2
    assert summary["max_release_lag_ns"] == 0
    assert "PRIVATE" not in repr(run) + repr(summary)
    assert run.events[-1].out == "PRIVATE"


def test_slow_decoder_discards_late_output_coalesces_and_prioritizes_full_final():
    time = ManualClock()
    calls = []

    def decode(prefix, final):
        calls.append((prefix, final, time.ns))
        time.advance(5 * SECOND // 2 if not final else SECOND // 2)
        return ("original", "translation")

    run = run_paced_prefix(5 * RATE, PacePolicy(RATE), decode, clock=time.read, sleep=time.sleep)
    assert calls == [(RATE, False, 8 * SECOND), (3 * RATE, False, 21 * SECOND // 2),
                     (5 * RATE, True, 13 * SECOND)]
    assert [e.disposition for e in run.events] == ["stale", "stale", "available"]
    assert run.scheduled_provisionals == 4 and run.skipped_provisionals == 2
    assert run.timing_summary()["first_provisional_from_start_ns"] is None
    assert run.timing_summary()["final_from_capture_end_ns"] == 3 * SECOND // 2
    assert run.timing_summary()["max_release_lag_ns"] == SECOND


def test_overslept_release_is_not_replayed_after_capture_end():
    time = ManualClock()

    def sleep(_):
        time.advance(4 * SECOND)

    run = run_paced_prefix(2 * RATE, PacePolicy(RATE), lambda p, f: (p, f),
                           clock=time.read, sleep=sleep)
    assert run.skipped_provisionals == 1
    assert [(e.final, e.out) for e in run.events] == [(True, (2 * RATE, True))]
    assert run.timing_summary()["final_start_lag_ns"] == 2 * SECOND


def test_exact_sample_timing_when_clip_shorter_than_cadence():
    time = ManualClock()
    run = run_paced_prefix(7, PacePolicy(), lambda n, f: str(n), clock=time.read, sleep=time.sleep)
    assert run.scheduled_provisionals == 0
    assert run.capture_end_ns - run.epoch_ns == 7 * 62_500
    assert run.events[0].prefix_samples == 7
    assert run.events[0].final


def test_empty_caption_and_stale_are_not_latency_successes():
    time = ManualClock()
    run = run_paced_prefix(2 * RATE, PacePolicy(RATE), lambda *_: ("", ""),
                           clock=time.read, sleep=time.sleep, has_caption=all)
    assert [e.disposition for e in run.events] == ["empty", "empty"]
    assert run.outcome == "final_empty"
    assert run.timing_summary()["first_provisional_from_start_ns"] is None
    assert run.timing_summary()["final_from_capture_end_ns"] is None


@pytest.mark.parametrize("on_final", [False, True])
def test_deadline_does_not_count_inflight_or_final_overrun_as_success(on_final):
    time = ManualClock()
    starts = []

    def decode(prefix, final):
        starts.append((prefix, final))
        if final == on_final:
            time.advance(4 * SECOND)
        return "caption"

    run = run_paced_prefix(2 * RATE, PacePolicy(RATE, 4 * SECOND), decode,
                           clock=time.read, sleep=time.sleep)
    assert run.outcome == "deadline_exceeded"
    assert run.events[-1].disposition == "deadline_exceeded"
    assert not run.events[-1].emitted
    assert run.timing_summary()["final_from_capture_end_ns"] is None
    assert len(starts) == (2 if on_final else 1)


def test_oversleep_to_deadline_never_starts_decoder():
    time = ManualClock()
    starts = []
    run = run_paced_prefix(2 * RATE, PacePolicy(RATE, 3 * SECOND),
                           lambda *args: starts.append(args), clock=time.read,
                           sleep=lambda _: time.advance(3 * SECOND))
    assert not starts
    assert run.outcome == "deadline_exceeded"
    assert run.skipped_provisionals == 1


def test_clock_rate_validation_and_policy_are_fail_closed():
    time = ManualClock()
    with pytest.raises(ValueError, match="sample count"):
        run_paced_prefix(True, PacePolicy(), lambda *_: "unused")
    with pytest.raises(ValueError, match="cadence"):
        PacePolicy(cadence_samples=True)
    with pytest.raises(ValueError, match="too many"):
        run_paced_prefix(RATE, PacePolicy(cadence_samples=1), lambda *_: "unused")
    with pytest.raises(ValueError, match="no opportunity"):
        run_paced_prefix(RATE, PacePolicy(deadline_ns=SECOND), lambda *_: "unused")
    with pytest.raises(ValueError, match="regressing"):
        run_paced_prefix(RATE, PacePolicy(), lambda *_: "unused", clock=time.read,
                         sleep=lambda _: time.advance(-1))
    with pytest.raises(ValueError, match="advance"):
        run_paced_prefix(RATE, PacePolicy(), lambda *_: "unused", clock=time.read, sleep=lambda _: None)
