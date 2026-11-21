"""Deterministic synthetic evidence only; no neural inference or microphone."""
import json

import pytest

from research.unified_models.stream_metrics import StreamMetricsError, measure_stream


def _e(kind, t, uid=10):
    return {'kind': kind, 't_ms': t, 'utterance': uid}


def _b(t, pending):
    return {'kind': 'backlog', 't_ms': t, 'pending_ms': pending}


def test_measures_distinct_clock_origins_and_backlog_rise_without_text_or_ids():
    events = [
        _e('utterance_start', 1000100, 42000), _b(1000100, 30),
        _e('provisional', 1000120, 42000), _b(1000130, 10),
        _e('utterance_end', 1000150, 42000), _e('utterance_start', 1000150, 64000), _b(1000150, 90),
        _e('provisional', 1000170, 42000), _e('stable', 1000180, 42000),
        _e('utterance_end', 1000210, 64000), _e('stable', 1000220, 64000), _b(1000230, 25),
    ]
    report = measure_stream(iter(events))
    assert report['scope'] == 'synthetic_event_timing_only'
    assert report['event_count'] == 12 and report['observed_duration_ms'] == 130
    assert report['utterances'] == {
        'started': 2, 'capture_ended': 2, 'stable_emitted': 2, 'without_provisional': 1, 'without_stable': 0,
        'completed_capture_duration_ms': 110,
    }
    assert report['first_provisional_from_start'] == {'samples': 1, 'p50_ms': 20, 'p95_ms': 20, 'max_ms': 20}
    assert report['stable_from_end'] == {'samples': 2, 'p50_ms': 10, 'p95_ms': 30, 'max_ms': 30}
    assert report['revisions'] == {'extra_provisional_events': 1, 'utterances_with_extra_provisionals': 1}
    assert report['backlog'] == {
        'samples': 4, 'observation_duration_ms': 130,
        'first_pending_ms': 30, 'peak_pending_ms': 90, 'last_pending_ms': 25,
        'net_growth_ms': -5, 'max_observed_rise_ms': 80,
    }
    encoded = json.dumps(report, sort_keys=True)
    assert all(str(raw) not in encoded for raw in (42000, 64000, 1000100, 1000150))


def test_unfinished_or_missing_observations_use_counts_and_null_not_fake_zero():
    report = measure_stream([_e('utterance_start', 100), _e('provisional', 100),
                             _e('utterance_start', 103, 20), _b(110, 3)])
    assert report['utterances']['started'] == 2
    assert report['utterances']['capture_ended'] == 0
    assert report['utterances']['stable_emitted'] == 0
    assert report['utterances']['without_stable'] == 2
    assert report['utterances']['completed_capture_duration_ms'] is None
    assert report['first_provisional_from_start']['p95_ms'] == 0  # Actual zero interval.
    assert report['stable_from_end'] == {'samples': 0, 'p50_ms': None, 'p95_ms': None, 'max_ms': None}
    assert report['backlog']['observation_duration_ms'] == 0  # One observed point.
    assert report['backlog']['net_growth_ms'] is None
    assert report['backlog']['max_observed_rise_ms'] is None
    empty = measure_stream([])
    assert empty['event_count'] == 0 and empty['observed_duration_ms'] is None
    assert empty['backlog']['samples'] == 0 and empty['backlog']['observation_duration_ms'] is None


def test_nearest_rank_deterministic_p95_distinct_from_max():
    events = []
    for index in range(1, 21):
        start = index * 100
        events.extend([_e('utterance_start', start, index), _e('provisional', start + index, index),
                       _e('utterance_end', start + index + 1, index), _e('stable', start + index + 1, index)])
    assert measure_stream(events)['first_provisional_from_start'] == {
        'samples': 20, 'p50_ms': 10, 'p95_ms': 19, 'max_ms': 20,
    }


@pytest.mark.parametrize('events, error', [
    ([_e('utterance_start', 3), _e('provisional', 2)], 'monotonic'),
    ([_e('utterance_start', 3), _e('stable', 3)], 'precedes capture end'),
    ([_e('utterance_start', 3), _e('utterance_end', 3), _e('stable', 3), _e('provisional', 3)], 'after terminal'),
    ([_e('utterance_start', 0), _e('utterance_start', 1)], 'duplicate utterance start'),
    ([_e('utterance_start', 1), _e('utterance_end', 2), _e('utterance_end', 3)], 'duplicate utterance end'),
    ([_e('utterance_end', 0)], 'without start'),
    ([_e('utterance_start', True)], 'invalid millisecond'),
    ([_e('utterance_start', 0, 'opaque-private-name')], 'invalid opaque'),
    ([_b(0, 1), _b(0, 2)], 'timestamps must increase'),
    ([_b(0, -1)], 'invalid pending duration'),
    ([_e('utterance_start', 0) | {'text': 'private-input-value'}], 'timing fields'),
])
def test_rejects_invalid_timing_or_content_without_echo(events, error):
    with pytest.raises(StreamMetricsError, match=error) as exc:
        measure_stream(events)
    assert 'private-' not in str(exc.value)
