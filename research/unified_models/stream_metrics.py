"""Summarize synthetic capture timing; no text, audio, models or app calls.

All timestamps use one monotonic millisecond clock and one supplied stream.
An 'utterance' value is a local opaque integer, not a speaker or file ID.
Measured emission times do not prove neural streaming or translation quality.
"""
from collections.abc import Mapping
from dataclasses import dataclass


_EVENT_KEYS = frozenset(('kind', 't_ms', 'utterance'))
_BACKLOG_KEYS = frozenset(('kind', 't_ms', 'pending_ms'))
_KINDS = frozenset(('utterance_start', 'utterance_end', 'provisional', 'stable', 'backlog'))
_MAX_EVENTS = 100_000
_MAX_VALUE = (1 << 63) - 1


class StreamMetricsError(ValueError):
    """Synthetic stream evidence has invalid or ambiguous timing."""


@dataclass
class _Utterance:
    start_ms: int
    end_ms: int | None = None
    first_provisional_ms: int | None = None
    stable_ms: int | None = None
    provisionals: int = 0


def _number(value):
    return type(value) is int and 0 <= value <= _MAX_VALUE


def _require(condition, index, message):
    if not condition:
        # Never include supplied event values or payloads in errors.
        raise StreamMetricsError(f'event {index}: {message}')


def _summary(values):
    """Integer nearest-rank percentiles; omitted observations are not zeros."""
    ordered = sorted(values)
    n = len(ordered)
    return {
        'samples': n,
        'p50_ms': ordered[(n * 50 + 99) // 100 - 1] if n else None,
        'p95_ms': ordered[(n * 95 + 99) // 100 - 1] if n else None,
        'max_ms': ordered[-1] if n else None,
    }


def measure_stream(events):
    """Return redacted aggregate metrics for ordered synthetic event evidence.

    Exact input event keys: `kind`, `t_ms`, and `utterance` for
    utterance_start/utterance_end/provisional/stable; or `kind`, `t_ms`,
    and `pending_ms` for backlog. Global timestamps are nondecreasing,
    while repeated backlog observations require distinct increasing times.
    Multiple utterances can be in flight; stable is terminal and can only
    follow the end of its own capture. Missing endings/emissions are counted
    as incomplete rather than inferred or assigned zero latency.
    """
    utterances = {}
    last_ms = first_ms = None
    backlog_count = 0
    backlog_first_t = backlog_last_t = None
    backlog_first = backlog_last = backlog_min = backlog_peak = backlog_max_rise = None
    count = 0
    for count, event in enumerate(events, 1):
        _require(count <= _MAX_EVENTS, count, 'too many timing events')
        _require(isinstance(event, Mapping), count, 'expected a timing event object')
        kind = event.get('kind')
        _require(type(kind) is str and kind in _KINDS, count, 'unknown event kind')
        expected = _BACKLOG_KEYS if kind == 'backlog' else _EVENT_KEYS
        _require(event.keys() == expected, count, 'unexpected or missing timing fields')
        t = event['t_ms']
        _require(_number(t), count, 'invalid millisecond timestamp')
        _require(last_ms is None or t >= last_ms, count, 'timestamps are not monotonic')
        if first_ms is None:
            first_ms = t
        last_ms = t
        if kind == 'backlog':
            pending = event['pending_ms']
            _require(_number(pending), count, 'invalid pending duration')
            _require(backlog_last_t is None or t > backlog_last_t, count, 'backlog sample timestamps must increase')
            if backlog_count == 0:
                backlog_first_t = t
                backlog_first = backlog_min = backlog_peak = pending
            else:
                backlog_max_rise = max(backlog_max_rise or 0, pending - backlog_min)
                backlog_min = min(backlog_min, pending)
                backlog_peak = max(backlog_peak, pending)
            backlog_count += 1
            backlog_last_t, backlog_last = t, pending
            continue

        uid = event['utterance']
        _require(_number(uid), count, 'invalid opaque utterance number')
        if kind == 'utterance_start':
            _require(uid not in utterances, count, 'duplicate utterance start')
            utterances[uid] = _Utterance(t)
            continue
        state = utterances.get(uid)
        _require(state is not None, count, 'utterance event without start')
        _require(state.stable_ms is None, count, 'utterance event after terminal stable output')
        if kind == 'utterance_end':
            _require(state.end_ms is None, count, 'duplicate utterance end')
            state.end_ms = t
        elif kind == 'provisional':
            if state.first_provisional_ms is None:
                state.first_provisional_ms = t
            state.provisionals += 1
        else:
            _require(state.end_ms is not None, count, 'stable output precedes capture end')
            state.stable_ms = t

    starts = len(utterances)
    ends = 0
    total_capture_ms = 0
    provisional = []
    stable = []
    replacements = 0
    revised = 0
    for state in utterances.values():
        if state.end_ms is not None:
            ends += 1
            total_capture_ms += state.end_ms - state.start_ms
        if state.first_provisional_ms is not None:
            provisional.append(state.first_provisional_ms - state.start_ms)
            replacements += state.provisionals - 1
            revised += state.provisionals > 1
        if state.stable_ms is not None:
            stable.append(state.stable_ms - state.end_ms)

    return {
        'schema_version': 1,
        'scope': 'synthetic_event_timing_only',
        'event_count': count,
        'observed_duration_ms': last_ms - first_ms if count else None,
        'utterances': {
            'started': starts, 'capture_ended': ends, 'stable_emitted': len(stable),
            'without_provisional': starts - len(provisional), 'without_stable': starts - len(stable),
            'completed_capture_duration_ms': total_capture_ms if ends else None,
        },
        'first_provisional_from_start': _summary(provisional),
        'stable_from_end': _summary(stable),
        'revisions': {
            'extra_provisional_events': replacements,
            'utterances_with_extra_provisionals': revised,
        },
        'backlog': {
            'samples': backlog_count,
            'observation_duration_ms': backlog_last_t - backlog_first_t if backlog_count else None,
            'first_pending_ms': backlog_first, 'peak_pending_ms': backlog_peak, 'last_pending_ms': backlog_last,
            'net_growth_ms': backlog_last - backlog_first if backlog_count > 1 else None,
            'max_observed_rise_ms': backlog_max_rise,
        },
    }
