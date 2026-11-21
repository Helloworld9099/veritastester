"""Local prerecorded 16 kHz prefix pacing, not a live or neural streaming runtime.

The capture clock exposes prefixes at absolute release times. A synchronous
decoder can have at most one job in flight. Obsolete provisional releases are
coalesced instead of enqueued; the whole-clip final is never coalesced. No
worker threads, network, model calls or audio file access are built in.
"""
from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic_ns, sleep as _sleep
from typing import Generic, TypeVar

_SAMPLE_RATE = 16_000
_NS_PER_SAMPLE = 1_000_000_000 // _SAMPLE_RATE
_MAX_SAMPLES = 30 * _SAMPLE_RATE
_MAX_PROVISIONALS = 120
_MAX_DEADLINE_NS = 180_000_000_000
T = TypeVar("T")


@dataclass(frozen=True)
class PacePolicy:
    """Cadence in samples; deadline measured from epoch, including simulated capture."""

    cadence_samples: int = 3 * _SAMPLE_RATE
    deadline_ns: int = 60_000_000_000

    def __post_init__(self):
        if type(self.cadence_samples) is not int or not 1 <= self.cadence_samples <= _MAX_SAMPLES:
            raise ValueError("invalid prefix cadence")
        if type(self.deadline_ns) is not int or not 1 <= self.deadline_ns <= _MAX_DEADLINE_NS:
            raise ValueError("invalid paced clip deadline")


@dataclass(frozen=True)
class PaceEvent(Generic[T]):
    scheduled_ns: int
    start_ns: int
    finish_ns: int
    prefix_samples: int
    final: bool
    emitted: bool
    disposition: str
    # Local decoder output, e.g. a pair of strings; never add to public receipts.
    out: T = field(repr=False, compare=False)


@dataclass(frozen=True)
class PaceRun(Generic[T]):
    epoch_ns: int
    capture_end_ns: int
    finish_ns: int
    events: tuple[PaceEvent[T], ...]
    outcome: str
    scheduled_provisionals: int
    skipped_provisionals: int

    def timing_summary(self) -> dict:
        """Redacted timings in integer ns; never return the decoder output.

        'Available' means the local decoder returned a usable, non-stale
        output, not that a UI displayed it. Backpressure is reported as
        skipped/dropped work and release lag (there is no FIFO).
        """
        emitted = [e for e in self.events if not e.final and e.emitted]
        final = next((e for e in self.events if e.final), None)
        return {
            "outcome": self.outcome,
            "planned_capture_ns": self.capture_end_ns - self.epoch_ns,
            "elapsed_ns": self.finish_ns - self.epoch_ns,
            "scheduled_provisionals": self.scheduled_provisionals,
            "provisional_decodes": sum(not e.final for e in self.events),
            "skipped_provisionals": self.skipped_provisionals,
            "stale_provisional_decodes": sum(e.disposition == "stale" for e in self.events),
            "empty_provisional_decodes": sum(not e.final and e.disposition == "empty" for e in self.events),
            "provisionals_available": len(emitted),
            "first_provisional_from_start_ns": (
                emitted[0].finish_ns - self.epoch_ns if emitted else None),
            "final_from_capture_end_ns": (
                final.finish_ns - self.capture_end_ns if final is not None and final.emitted else None),
            "final_start_lag_ns": final.start_ns - final.scheduled_ns if final else None,
            "max_release_lag_ns": max(
                (e.start_ns - e.scheduled_ns for e in self.events), default=None),
            "decode_and_validation_ns": sum(e.finish_ns - e.start_ns for e in self.events),
        }


def run_paced_prefix(
    total_samples: int,
    policy: PacePolicy,
    decode: Callable[[int, bool], T],
    *,
    clock: Callable[[], int] = monotonic_ns,
    sleep: Callable[[float], None] = _sleep,
    has_caption: Callable[[T], bool] = bool,
) -> PaceRun[T]:
    """Pace a sequence of increasingly complete prefixes and one final.

    decode receives (available sample count, is_final). It must consume only
    that prefix and return fully materialized output. has_caption checks that
    output's usability; its runtime is included in completion timing. If a
    provisional returns at or after a newer release (including final), it is
    stale. Final is scheduled at the precise duration; a clip whose length is
    exactly on cadence never schedules a duplicate same-size provisional.

    Deadline enforcement is cooperative: no calls start at/after the
    deadline; a result completing at/after it is not eligible as evidence.
    A deadline cannot interrupt a blocked decoder; use a separate process
    watchdog for hard limits. Exceptions from a supplied decoder propagate.
    """
    if type(total_samples) is not int or not 1 <= total_samples <= _MAX_SAMPLES:
        raise ValueError("invalid paced clip sample count")
    if type(policy) is not PacePolicy:
        raise ValueError("expected a pace policy")
    scheduled = (total_samples - 1) // policy.cadence_samples
    if scheduled > _MAX_PROVISIONALS:
        raise ValueError("too many scheduled provisional releases")
    duration_ns = total_samples * _NS_PER_SAMPLE
    if policy.deadline_ns <= duration_ns:
        raise ValueError("pace deadline leaves no opportunity for final decode")
    cadence_ns = policy.cadence_samples * _NS_PER_SAMPLE

    previous_time = None

    def read_time():
        nonlocal previous_time
        value = clock()
        if type(value) is not int or value < 0 or previous_time is not None and value < previous_time:
            raise ValueError("invalid or regressing monotonic clock")
        previous_time = value
        return value

    epoch = now = read_time()
    end = epoch + duration_ns
    deadline = epoch + policy.deadline_ns
    events = []
    next_release = 1

    def finish(outcome, at):
        # Every provisional release is either attempted once or skipped,
        # including unstarted releases after a deadline miss.
        return PaceRun(epoch, end, at, tuple(events), outcome, scheduled,
                       scheduled - sum(not e.final for e in events))

    while True:
        if now >= deadline:
            return finish("deadline_exceeded", now)
        if now >= end:
            available, release, final = total_samples, end, True
        else:
            due = (now - epoch) // cadence_ns
            if due < next_release:
                release = min(end, epoch + next_release * cadence_ns)
                sleep((release - now) / 1_000_000_000)
                new_now = read_time()
                if new_now == now:
                    raise ValueError("paced sleep did not advance clock")
                now = new_now
                continue
            # Intermediate due releases are never submitted to a worker.
            available, release, final = due * policy.cadence_samples, epoch + due * cadence_ns, False
            next_release = due + 1

        start = now
        out = decode(available, final)
        # Do not perform application validation on an already expired decode.
        now = read_time()
        if now >= deadline:
            disposition = "deadline_exceeded"
        else:
            usable = has_caption(out)
            now = read_time()
            if now >= deadline:
                disposition = "deadline_exceeded"
            elif not final and now >= min(end, epoch + next_release * cadence_ns):
                disposition = "stale"
            else:
                disposition = "available" if usable else "empty"
        events.append(PaceEvent(release, start, now, available, final,
                                disposition == "available", disposition, out))
        if disposition == "deadline_exceeded":
            return finish("deadline_exceeded", now)
        if final:
            return finish("completed" if disposition == "available" else "final_empty", now)
