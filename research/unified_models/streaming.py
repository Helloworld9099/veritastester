"""Opt-in speech research transport for a single reviewed model, with no default.

This module neither captures PCM nor downloads, runs, or trains any model by
itself. The Python factory is an explicit trusted boundary, not an OS sandbox.
An independently pinned speech receipt is admission metadata, not proof that
a backend really has a shared encoder or has passed release gates. A real
reviewed loader must independently verify immutable local weights, tokenizer
and config bytes before use and isolate compute from network and private data.
The runtime compares backend-reported identity; it cannot prove loaded bytes
or observe an implementation secretly switching or reloading its weights.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import re
import threading
from typing import Callable, Protocol

SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_BYTES = 2
PCM_FORMAT = "pcm_s16le_mono_16000"
_SHA = re.compile(r"[a-f0-9]{64}\Z")
_ROUTE = re.compile(r"([a-z]{2,3})->([a-z]{2,3})\Z")
_STREAM_ISSUER = object()


class StreamError(ValueError):
    """Malformed input, invalid backend output, or unavailable stream."""


def _require(ok, message):
    if not ok:
        raise StreamError(message)


def _sha(value):
    return type(value) is str and _SHA.fullmatch(value) is not None


def _caption_text(text, limit):
    return (type(text) is str and 0 < len(text) <= limit and bool(text.strip())
            and "\0" not in text and not any(0xD800 <= ord(c) <= 0xDFFF for c in text))


@dataclass(frozen=True, repr=False)
class SpeechQualification:
    """Explicitly reviewed streaming checkpoint and bounded directions.

    Obtain the digest independently of the submitted receipt. A digest proves
    which metadata is being used, not reviewer authority or a release result.
    Existing text-only receipts are not accepted here.
    """

    checkpoint_sha256: str
    directions: frozenset[str]
    receipt_sha256: str

    def __init__(self, receipt: Mapping, *, expected_sha256: str):
        _require(isinstance(receipt, Mapping) and _sha(expected_sha256),
                 "An independently pinned speech qualification receipt is required")
        try:
            # Take a plain byte snapshot before checking mutable caller metadata.
            raw = json.dumps(dict(receipt), sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
            pin = hashlib.sha256(raw).hexdigest()
            value = json.loads(raw)
        except (TypeError, ValueError, UnicodeError, RecursionError):
            raise StreamError("Invalid speech qualification receipt") from None
        _require(pin == expected_sha256, "Speech qualification receipt checksum mismatch")
        _require(len(raw) <= 65_536 and set(value) == {
            "schema_version", "kind", "scope", "checkpoint_sha256", "pcm_format",
            "shared_audio_encoder", "directions",
        } and type(value["schema_version"]) is int and value["schema_version"] == 1
            and value["kind"] == "reviewed_paired_streaming_speech_dev"
            and value["scope"] == "research_only" and value["pcm_format"] == PCM_FORMAT
            and value["shared_audio_encoder"] is True and _sha(value["checkpoint_sha256"]),
            "Only reviewed shared-encoder streaming speech DEV metadata is accepted")
        directions = value["directions"]
        _require(isinstance(directions, dict) and 1 <= len(directions) <= 128,
                 "No bounded reviewed speech directions declared")
        for direction, evidence in directions.items():
            match = _ROUTE.fullmatch(direction)
            _require(match is not None and match[1] != match[2]
                     and isinstance(evidence, dict)
                     and set(evidence) == {"status", "evidence_sha256"}
                     and evidence["status"] == "passed" and _sha(evidence["evidence_sha256"]),
                     "Each supported speech direction requires its own passing evidence")
        object.__setattr__(self, "checkpoint_sha256", value["checkpoint_sha256"])
        object.__setattr__(self, "directions", frozenset(directions))
        object.__setattr__(self, "receipt_sha256", pin)

    def require_direction(self, source, target):
        _require(type(source) is str and type(target) is str,
                 "Explicit source and target language codes are required")
        a, b = source.strip().lower(), target.strip().lower()
        _require(f"{a}->{b}" in self.directions,
                 "Unsupported speech direction; no unverified locale or model fallback")
        return a, b


@dataclass(frozen=True)
class StreamLimits:
    """Defaults keep input and output bounded; no unbounded append-only state."""

    max_frame_samples: int = SAMPLE_RATE
    max_stream_samples: int = 3600 * SAMPLE_RATE
    max_events_per_push: int = 16
    max_utterances: int = 16_384
    max_updates_per_utterance: int = 128
    max_caption_chars: int = 1500

    def __post_init__(self):
        for value, upper in (
            (self.max_frame_samples, SAMPLE_RATE),
            (self.max_stream_samples, 4 * 3600 * SAMPLE_RATE),
            (self.max_events_per_push, 64), (self.max_utterances, 100_000),
            (self.max_updates_per_utterance, 1024), (self.max_caption_chars, 4096),
        ):
            _require(type(value) is int and 0 < value <= upper, "Invalid streaming limit")
        _require(self.max_frame_samples <= self.max_stream_samples,
                 "Frame limit cannot exceed stream audio budget")


@dataclass(frozen=True)
class PCMFrame:
    """Zero-based sequential PCM; fixed 16 kHz signed 16-bit LE mono."""

    sequence: int
    pcm: bytes = field(repr=False)


@dataclass(frozen=True)
class CaptionUpdate:
    """Paired caption, revised together until final; content omitted from repr."""

    utterance_id: int
    original: str = field(repr=False)
    translated: str = field(repr=False)
    final: bool


class DecoderStream(Protocol):
    """One stream's ephemeral decoder state within the model, not another model."""

    def push(self, pcm: bytes) -> list[CaptionUpdate] | tuple[CaptionUpdate, ...]: ...
    def finish(self) -> list[CaptionUpdate] | tuple[CaptionUpdate, ...]: ...
    def abort(self) -> None: ...


class SpeechBackend(Protocol):
    """Trusted implementation with one loaded checkpoint/shared audio encoder.

    Verify immutable files inside the reviewed isolated loader. This reported
    checkpoint SHA is a contract assertion, never a byte attestation.
    """

    checkpoint_sha256: str

    def open_stream(self, *, source: str, target: str) -> DecoderStream: ...


class SpeechStream:
    """Bounded incremental PCM and paired-caption stream; not an audio recorder."""

    def __init__(self, *, decoder: DecoderStream, source: str, target: str,
                 lock, release: Callable[[SpeechStream], None], limits: StreamLimits,
                 _issuer=None):
        _require(_issuer is _STREAM_ISSUER, "Only a qualified runtime can issue a speech stream")
        self._decoder = decoder
        self._direction = (source, target)
        self._lock = lock
        self._release = release
        self._limits = limits
        self._state = "open"
        self._expected_sequence = 0
        self._samples = 0
        self._utterance_id = 0
        self._provisional_count = 0
        self._inside_call = False

    @property
    def direction(self) -> tuple[str, str]:
        return self._direction

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def _event_batch(self, updates, *, terminal=False):
        limit = self._limits
        _require(type(updates) in (tuple, list) and len(updates) <= limit.max_events_per_push,
                 "Invalid or excessive caption events from the speech decoder")
        utterance, provisional = self._utterance_id, self._provisional_count
        output = tuple(updates)
        for item in output:
            _require(type(item) is CaptionUpdate and type(item.utterance_id) is int
                     and item.utterance_id == utterance and utterance < limit.max_utterances
                     and type(item.final) is bool and (not terminal or item.final)
                     and _caption_text(item.original, limit.max_caption_chars)
                     and _caption_text(item.translated, limit.max_caption_chars),
                     "Invalid, out-of-order or retracted paired caption")
            if item.final:
                utterance += 1
                provisional = 0
            else:
                provisional += 1
                _require(provisional <= limit.max_updates_per_utterance,
                         "Too many provisional updates for one utterance")
        _require(not terminal or provisional == 0, "Final flush left a provisional caption unresolved")
        # Only commit the ledger after validating the entire batch.
        self._utterance_id, self._provisional_count = utterance, provisional
        return output

    def _close(self, state, *, abort_decoder):
        self._state = state
        decoder, self._decoder = self._decoder, None
        try:
            if abort_decoder:
                decoder.abort()
        except Exception:
            pass
        finally:
            self._release(self)

    def _run_decoder(self, *, pcm=None, terminal=False):
        self._inside_call = True
        try:
            try:
                batch = self._decoder.finish() if terminal else self._decoder.push(pcm)
            except Exception:
                raise StreamError("Speech decoder failed; no fallback model is available") from None
            return self._event_batch(batch, terminal=terminal)
        except StreamError:
            self._close("failed", abort_decoder=True)
            raise
        finally:
            self._inside_call = False

    def _require_open(self):
        _require(self._state == "open" and not self._inside_call, "Speech stream is not open")

    def push(self, frame: PCMFrame) -> tuple[CaptionUpdate, ...]:
        """Take one bounded sample-aligned frame; gaps and duplicate frames fail."""
        with self._lock:
            self._require_open()
            _require(type(frame) is PCMFrame and type(frame.sequence) is int
                     and frame.sequence == self._expected_sequence,
                     "PCM frame sequence must be consecutive and zero-based")
            pcm = frame.pcm
            _require(type(pcm) is bytes and 0 < len(pcm) <= SAMPLE_BYTES * self._limits.max_frame_samples
                     and len(pcm) % SAMPLE_BYTES == 0,
                     "PCM frame must be bounded nonempty signed 16-bit LE mono audio")
            samples = len(pcm) // SAMPLE_BYTES
            _require(self._samples + samples <= self._limits.max_stream_samples,
                     "Speech stream audio budget exhausted; finish or abort explicitly")
            output = self._run_decoder(pcm=pcm)
            self._expected_sequence += 1
            self._samples += samples
            return output

    def finish(self) -> tuple[CaptionUpdate, ...]:
        """Flush final pairs only. A stream with unresolved provisional text fails."""
        with self._lock:
            self._require_open()
            if self._samples == 0:
                self._close("finished", abort_decoder=True)
                return ()
            output = self._run_decoder(terminal=True)
            self._close("finished", abort_decoder=False)
            return output

    def abort(self) -> None:
        """Discard provisional state without resurrecting already finalized output."""
        with self._lock:
            if self._state == "open":
                self._require_open()
                self._close("aborted", abort_decoder=True)


class SingleStreamingSpeechRuntime:
    """One qualified model for the runtime's lifetime; one active stream at a time.

    To change direction, finish or abort the existing stream and open a new
    ephemeral decoding state on the same model. No factory call gets a route.
    Neither context support nor independent accuracy or performance is implied.
    """

    supports_context = False

    def __init__(self, *, qualification: SpeechQualification | None = None,
                 load_backend: Callable[[str], SpeechBackend] | None = None,
                 limits: StreamLimits | None = None, opt_in: bool = False):
        _require(qualification is None or type(qualification) is SpeechQualification,
                 "Invalid speech qualification")
        _require(limits is None or type(limits) is StreamLimits, "Invalid streaming limits")
        _require(type(opt_in) is bool, "Explicit boolean research opt-in required")
        self._qualification = qualification
        self._load_backend = load_backend
        self._limits = limits if limits is not None else StreamLimits()
        self._opt_in = opt_in
        self._backend: SpeechBackend | None = None
        self._attempted_load = False
        self._identity_failed = False
        self._opening = False
        self._active: SpeechStream | None = None
        self._lock = threading.RLock()

    @property
    def loaded_model_count(self) -> int:
        with self._lock:
            return int(self._backend is not None)

    def status(self) -> dict:
        with self._lock:
            q = self._qualification
            return {
                "scope": "research_only",
                "enabled": (self._opt_in and q is not None and callable(self._load_backend)
                            and not self._identity_failed
                            and not (self._attempted_load and self._backend is None)),
                "qualified_directions": sorted(q.directions) if q else [],
                "loaded_model_count": int(self._backend is not None),
                "production_activated": False,
            }

    def _check_identity(self, backend):
        _require(not self._identity_failed, "Speech backend identity was invalidated; no fallback")
        try:
            reported = backend.checkpoint_sha256
            matches = _sha(reported) and reported == self._qualification.checkpoint_sha256
        except Exception:
            matches = False
        if not matches:
            self._identity_failed = True
            raise StreamError("Reported speech checkpoint identity changed or was invalidated; no fallback")

    def _get_backend(self) -> SpeechBackend:
        if self._backend is not None:
            self._check_identity(self._backend)
            return self._backend
        _require(not self._attempted_load, "Speech checkpoint loading already attempted; no model fallback")
        self._attempted_load = True
        checkpoint = self._qualification.checkpoint_sha256
        try:
            backend = self._load_backend(checkpoint)
            reported = getattr(backend, "checkpoint_sha256", None)
            matches = (_sha(reported) and reported == checkpoint
                       and callable(getattr(backend, "open_stream", None)))
        except Exception:
            raise StreamError("Single reviewed speech checkpoint could not be loaded") from None
        _require(matches, "Loaded speech checkpoint or backend does not match its qualification")
        self._backend = backend
        return backend

    def _release(self, stream):
        if self._active is stream:
            self._active = None

    def start_stream(self, *, source: str, target: str) -> SpeechStream:
        """Only explicit reviewed directions may reach the lazy backend loader."""
        with self._lock:
            _require(self._opt_in and self._qualification is not None and callable(self._load_backend),
                     "No opt-in qualified speech backend is configured")
            a, b = self._qualification.require_direction(source, target)
            _require(self._active is None and not self._opening,
                     "Finish or abort the current stream before changing direction")
            self._opening = True
            try:
                backend = self._get_backend()
                try:
                    decoder = backend.open_stream(source=a, target=b)
                    valid = all(callable(getattr(decoder, name, None)) for name in ("push", "finish", "abort"))
                except Exception:
                    raise StreamError("Could not open speech decoder state; no model fallback") from None
                _require(valid, "Invalid state returned by the speech backend")
                try:
                    self._check_identity(backend)
                except StreamError:
                    try:
                        decoder.abort()
                    except Exception:
                        pass
                    raise
                self._active = SpeechStream(decoder=decoder, source=a, target=b,
                    lock=self._lock, release=self._release, limits=self._limits, _issuer=_STREAM_ISSUER)
                return self._active
            finally:
                self._opening = False
