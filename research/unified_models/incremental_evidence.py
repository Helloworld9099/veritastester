"""Check redacted CPU instrumentation before benchmarking a speech candidate.

This is a structural preflight, not proof of runtime provenance or performance.
Never put text, samples, file paths or user identifiers in the input trace.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

from research.workflow_contract import ContractError, is_sha256, read_pinned_json

_DIRECTION = re.compile(r"([a-z]{2,3})->([a-z]{2,3})\Z")
_MAX = (1 << 63) - 1
_EVENTS = {
    "load": {"checkpoint_sha256"},
    "start": {"direction"},
    "release": {"end"},
    "encode": {"start", "end", "cache_in", "cache_out"},
    "draft": {"channel", "cache"},
    "end": set(),
    "final": {"channel", "cache"},
}
_CHANNELS = ("source", "translation")


class TraceError(ValueError):
    """Invalid structural evidence. Public messages contain no event values."""


def _require(ok, code):
    if not ok:
        raise TraceError(code)


def _nat(value):
    return type(value) is int and 0 <= value <= _MAX


def _match(value):
    m = _DIRECTION.fullmatch(value) if type(value) is str else None
    return m if m and m[1] != m[2] else None


def validate_trace(record, *, expected_checkpoint_sha256, required_directions):
    """Check one process, serial utterances, one checkpoint, both caption channels.

    Encode spans are the *actual raw audio* represented by the neural encoder call
    (exclude padding, include overlapped real input). Cache generations are opaque
    instrumented state IDs, carried across calls within an utterance, not hashes of
    model tensors. Caption events reference the encoder state they actually read.
    The producer must instrument the real model boundary, not the requested chunks.
    """
    _require(type(record) is dict and set(record) == {
        "schema_version", "kind", "scope", "device", "sample_rate", "method_sha256", "events"
    }, "invalid_trace_shape")
    _require(type(record["schema_version"]) is int and record["schema_version"] == 1
             and record["kind"] == "bridgecaption_incremental_trace"
             and record["scope"] == "open_development" and record["device"] == "cpu"
             and type(record["sample_rate"]) is int and record["sample_rate"] == 16000
             and is_sha256(record["method_sha256"]), "unsupported_trace_header")
    _require(is_sha256(expected_checkpoint_sha256), "independent_checkpoint_pin_required")
    directions = tuple(required_directions)
    matches = [_match(d) for d in directions]
    _require(2 <= len(directions) <= 32 and all(matches), "invalid_required_directions")
    _require(len(set(directions)) == len(directions) and len({m[1] for m in matches}) >= 2,
             "multiple_distinct_source_languages_required")
    events = record["events"]
    _require(type(events) is list and 1 <= len(events) <= 20_000, "invalid_event_count")
    summary = {d: {"utterances": 0, "encoder_calls": 0, "audio_samples": 0} for d in directions}
    state = None
    loaded = False
    last = 0
    for event in events:
        _require(type(event) is dict, "invalid_event")
        kind = event.get("kind")
        _require(type(kind) is str and kind in _EVENTS
                 and event.keys() == {"kind", "t_ms"} | _EVENTS[kind], "invalid_event_fields")
        t = event["t_ms"]
        _require(_nat(t) and t >= last, "non_monotonic_event_clock")
        last = t
        if kind == "load":
            _require(not loaded and state is None, "checkpoint_reloaded_or_switched")
            _require(event["checkpoint_sha256"] == expected_checkpoint_sha256,
                     "unexpected_checkpoint")
            loaded = True
            continue
        _require(loaded, "missing_checkpoint_load")
        if kind == "start":
            direction = event["direction"]
            _require(state is None, "unfinished_utterance")
            _require(type(direction) is str and direction in summary, "unexpected_direction")
            state = {"direction": direction, "release": 0, "cursor": 0, "calls": 0,
                     "cache": 0, "eof": False, "drafts": set(), "finals": set()}
            continue
        _require(state is not None, "event_without_active_utterance")
        if kind == "release":
            end = event["end"]
            _require(not state["eof"] and _nat(end) and end > state["release"],
                     "invalid_audio_release")
            state["release"] = end
        elif kind == "encode":
            start, end = event["start"], event["end"]
            cin, cout = event["cache_in"], event["cache_out"]
            _require(_nat(start) and _nat(end) and start == state["cursor"]
                     and start < end <= state["release"], "overlap_gap_or_future_audio")
            _require(_nat(cin) and _nat(cout) and cin == state["cache"] and cout > cin,
                     "encoder_state_not_reused")
            _require(not state["finals"], "encoder_advanced_after_final")
            state.update(cursor=end, cache=cout, calls=state["calls"] + 1)
        elif kind == "end":
            _require(not state["eof"] and state["release"] > 0, "invalid_end_of_capture")
            _require(state["drafts"] == set(_CHANNELS), "missing_pre_eof_caption_channels")
            state["eof"] = True
        else:
            channel, cache = event["channel"], event["cache"]
            _require(channel in _CHANNELS and _nat(cache) and 0 < cache == state["cache"],
                     "caption_not_on_current_shared_encoder_state")
            if kind == "draft":
                _require(not state["eof"], "draft_after_capture_end")
                state["drafts"].add(channel)
            else:
                _require(state["eof"] and state["cursor"] == state["release"],
                         "final_before_all_audio_is_encoded")
                _require(channel not in state["finals"], "duplicate_final")
                state["finals"].add(channel)
                if state["finals"] == set(_CHANNELS):
                    _require(state["calls"] >= 2, "single_encode_is_not_incremental_evidence")
                    counts = summary[state["direction"]]
                    counts["utterances"] += 1
                    counts["encoder_calls"] += state["calls"]
                    counts["audio_samples"] += state["cursor"]
                    state = None
    _require(state is None, "unfinished_utterance")
    _require(all(counts["utterances"] for counts in summary.values()), "missing_required_direction")
    return {"status": "trace_structure_passed", "directions": summary,
            "model_runtime_verified_by_validator": False, "benchmark_passed": False}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--trace", required=True, help="Relative path within the evidence root")
    p.add_argument("--trace-sha256", required=True)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--direction", action="append", required=True)
    args = p.parse_args(argv)
    try:
        record = read_pinned_json(args.root, args.trace, expected_sha256=args.trace_sha256)
        result = validate_trace(record, expected_checkpoint_sha256=args.checkpoint_sha256,
                                required_directions=args.direction)
    except (TraceError, ContractError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
