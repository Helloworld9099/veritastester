"""Local, untrained CPU probe: one causal recurrent encoder, two numeric heads.

No token means a word or translated text. The probe does not implement the
qualified SpeechBackend interface, load pretrained weights, or accept files.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import struct

from .incremental_evidence import validate_trace

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 320
MAX_PUSH_SAMPLES = SAMPLE_RATE
MAX_UTTERANCE_SAMPLES = 30 * SAMPLE_RATE
FRAME_BYTES = FRAME_SAMPLES * 2
DIRECTIONS = ("es->en", "fr->en")  # Routing labels, not language support.
_WIDTH = 8
_CLASSES = 5
_FEATURES = 4


def _matrix(rows, columns, seed):
    # Portable, fixed diagnostic coefficients; not trained or fitted.
    return tuple(tuple((((i * columns + j + 1) * 104_729 + seed * 7919) ** 2
                         % 65_521) % 31 - 15 for j in range(columns)) for i in range(rows))


@dataclass(frozen=True)
class _Coefficients:
    input: tuple = _matrix(_WIDTH, _FEATURES, 11)
    recurrent: tuple = _matrix(_WIDTH, _WIDTH, 23)
    source: tuple = _matrix(_CLASSES, _WIDTH, 37)
    translation: tuple = _matrix(_CLASSES, _WIDTH, 53)
    routes: tuple = _matrix(len(DIRECTIONS), _CLASSES, 67)

    def digest(self):
        raw = json.dumps({
            "input": self.input, "recurrent": self.recurrent, "source": self.source,
            "translation": self.translation, "route_bias": dict(zip(DIRECTIONS, self.routes)),
        }, sort_keys=True, separators=(",", ":")).encode("ascii")
        return sha256(raw).hexdigest()


@dataclass(frozen=True)
class _Cache:
    generation: int = 0
    cursor: int = 0
    hidden: tuple = (0,) * _WIDTH


class _Encoder:
    def __init__(self, coefficients):
        self._c = coefficients

    def advance(self, real_frame, cache, emit):
        """Encode exactly the new raw frame. A short final frame is not padded."""
        samples = struct.unpack(f"<{len(real_frame) // 2}h", real_frame)
        n = len(samples)
        features = (
            sum(samples) // n,
            sum(map(abs, samples)) // n,
            max(map(abs, samples)),
            sum(abs(a - b) for a, b in zip(samples, samples[1:])) // n,
        )
        hidden = tuple(
            max(-32768, min(32767, (
                sum(w * x for w, x in zip(row, features))
                + sum(w * h for w, h in zip(history, cache.hidden))
            ) // 64))
            for row, history in zip(self._c.input, self._c.recurrent)
        )
        next_cache = _Cache(cache.generation + 1, cache.cursor + n, hidden)
        # Instrument inside the encoder using the received samples and actual state.
        emit("encode", start=cache.cursor, end=next_cache.cursor,
             cache_in=cache.generation, cache_out=next_cache.generation)
        return next_cache


@dataclass(frozen=True)
class ProbeOutput:
    source_id: int
    translation_id: int
    generation: int
    final: bool


class CausalProbe:
    """Resident coefficient set and encoder; one diagnostic stream at a time."""

    def __init__(self, emit):
        self._emit = emit
        self._coefficients = _Coefficients()
        self._encoder = _Encoder(self._coefficients)
        self._active = None
        self.checkpoint_sha256 = self._coefficients.digest()
        emit("load", checkpoint_sha256=self.checkpoint_sha256)

    def start(self, direction):
        if type(direction) is not str or direction not in DIRECTIONS or self._active is not None:
            raise ValueError("unsupported probe route or unfinished stream")
        self._emit("start", direction=direction)
        self._active = _Utterance(self, DIRECTIONS.index(direction))
        return self._active

    def _decode(self, cache, channel, route, kind):
        c = self._coefficients
        rows = c.source if channel == "source" else c.translation
        # Route is conditioning inside the same resident weights, not model selection.
        bias = c.routes[route]
        scores = tuple(sum(w * x for w, x in zip(row, cache.hidden)) + bias[i] * 128
                       for i, row in enumerate(rows))
        token = max(range(_CLASSES), key=scores.__getitem__)
        self._emit(kind, channel=channel, cache=cache.generation)
        return token


class _Utterance:
    def __init__(self, model, route):
        self._model = model
        self._route = route
        self._cache = _Cache()
        self._tail = b""
        self._released = 0
        self._closed = False

    @property
    def buffered_samples(self):
        return len(self._tail) // 2

    def _pair(self, final):
        kind = "final" if final else "draft"
        cache = self._cache
        return ProbeOutput(self._model._decode(cache, "source", self._route, kind),
                           self._model._decode(cache, "translation", self._route, kind),
                           cache.generation, final)

    def push(self, pcm):
        if self._closed or type(pcm) is not bytes or not pcm or len(pcm) % 2:
            raise ValueError("expected nonempty signed 16-bit PCM in an open stream")
        samples = len(pcm) // 2
        if samples > MAX_PUSH_SAMPLES or self._released + samples > MAX_UTTERANCE_SAMPLES:
            raise ValueError("probe audio budget exceeded")
        self._released += samples
        self._model._emit("release", end=self._released)
        joined = self._tail + pcm
        complete = len(joined) - len(joined) % FRAME_BYTES
        for begin in range(0, complete, FRAME_BYTES):
            frame = joined[begin:begin + FRAME_BYTES]
            self._cache = self._model._encoder.advance(frame, self._cache, self._model._emit)
        self._tail = joined[complete:]
        return self._pair(False) if complete else None

    def finish(self):
        if self._closed:
            raise ValueError("probe stream is already closed")
        self._closed = True
        self._model._emit("end")
        if self._tail:
            self._cache = self._model._encoder.advance(self._tail, self._cache, self._model._emit)
        self._tail = b""
        result = self._pair(True) if self._cache.generation else None
        self._model._active = None
        self._cache = _Cache()
        return result


def _synthetic_pcm(count, phase):
    """Deterministic, generated diagnostic waveform; never human speech."""
    return struct.pack(f"<{count}h", *(
        ((i * 37 + phase * 11) % 701 - 350) * (5 + ((i // 193 + phase) % 19))
        for i in range(count)
    ))


def run_probe(*, raw_trace=False):
    """Execute the encoder and heads, returning redacted summary or raw events."""
    events = []
    clock = 0

    def emit(kind, **fields):
        events.append({"kind": kind, "t_ms": clock, **fields})

    model = CausalProbe(emit)
    max_tail = 0
    for route, phase in zip(DIRECTIONS, (3, 17)):
        stream = model.start(route)
        for n, shift in ((159, 0), (483, 1), (632, 2), (127, 3)):
            # Protocol ticks, not measured timestamps. This is not paced replay.
            clock += 1
            stream.push(_synthetic_pcm(n, phase + shift))
            max_tail = max(max_tail, stream.buffered_samples)
        clock += 1
        stream.finish()
    header = {"schema_version": 1, "kind": "bridgecaption_incremental_trace",
              "scope": "open_development", "device": "cpu", "sample_rate": SAMPLE_RATE,
              "method_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "events": events}
    result = validate_trace(header, expected_checkpoint_sha256=model.checkpoint_sha256,
                            required_directions=DIRECTIONS)
    if raw_trace:
        return header
    return {
        "kind": "bridgecaption_untrained_causal_probe",
        "checkpoint_sha256": model.checkpoint_sha256,
        "method_sha256": header["method_sha256"],
        "scope": "generated_waveform_and_numeric_heads_only",
        "clock": "logical_ticks_not_latency",
        "model_load_events": sum(e["kind"] == "load" for e in events),
        "head_events": {mode: sum(e["kind"] == mode for e in events)
                        for mode in ("draft", "final")},
        "max_retained_tail_samples": max_tail,
        "encoder_hidden_width": _WIDTH,
        "trace_validation": result,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="store_true", help="Print the redacted logical event trace")
    args = parser.parse_args(argv)
    print(json.dumps(run_probe(raw_trace=args.trace), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
