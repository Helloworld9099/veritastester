"""Project reviewed research outputs onto the incumbent caption event contract.

This module owns no capture device, provider, thread, model loader or UI.
The desktop never imports it. All callers must supply a previously admitted
research stream and audio they are already authorized to process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic_ns
from typing import Callable

from .streaming import PCMFrame, SpeechStream


@dataclass(frozen=True)
class ProvisionalCaption:
    utterance_id: int
    original: str = field(repr=False)
    translated: str = field(repr=False)
    source_language: str


class CaptionEventBridge:
    """A single-consumer adaptor; use off the UI thread.

    `on_event` receives the existing LiveSession five-field finalized-caption
    payload. `on_provisional` receives revisions separately so an incumbent
    consumer never mistakes them for finalized captions. These timings only
    cover the triggering frame's dispatch; independent utterance latency is
    measured separately. The callbacks receive text and must not log it.
    """

    def __init__(self, stream: SpeechStream, *, on_event: Callable,
                 on_provisional: Callable | None = None,
                 clock_ns: Callable[[], int] = monotonic_ns):
        if type(stream) is not SpeechStream or stream.state != "open":
            raise ValueError("An admitted open research speech stream is required")
        self._source = stream.direction[0]
        if (not callable(on_event) or not callable(clock_ns)
                or (on_provisional is not None and not callable(on_provisional))):
            raise ValueError("Caption callbacks must be callable")
        self._stream = stream
        self._sink = on_event
        self._draft_sink = on_provisional
        self._clock_ns = clock_ns
        self._last_frame_end_ns: int | None = None
        self._closed = False
        self._in_operation = False
        self._last_clock_ns: int | None = None

    def _now(self):
        now = self._clock_ns()
        if (type(now) is not int or now < 0
                or (self._last_clock_ns is not None and now < self._last_clock_ns)):
            raise ValueError("Use a nonnegative monotonic nanosecond clock")
        self._last_clock_ns = now
        return now

    def _publish(self, updates, observed_ns: int, frame_end_ns: int):
        # The session has validated the entire batch before any callback runs.
        lag = round((observed_ns - frame_end_ns) / 1_000_000_000, 2)
        for update in updates:
            if self._closed:
                return
            if update.final:
                self._sink("caption", update.original, update.translated, self._source,
                           lag, "research streaming; context not verified")
            elif self._draft_sink is not None:
                self._draft_sink(ProvisionalCaption(
                    update.utterance_id, update.original, update.translated, self._source,
                ))

    def push(self, frame: PCMFrame, *, frame_end_ns: int):
        self._require_available()
        self._in_operation = True
        try:
            if (type(frame_end_ns) is not int or frame_end_ns < 0
                    or (self._last_frame_end_ns is not None and frame_end_ns < self._last_frame_end_ns)
                    or frame_end_ns > self._now()):
                raise ValueError("Frame-end time must be nondecreasing on the supplied monotonic clock")
            self._last_frame_end_ns = frame_end_ns
            try:
                updates = self._stream.push(frame)
                self._publish(updates, self._now(), frame_end_ns)
            except Exception:
                self.abort()
                raise ValueError("Research caption stream ended without fallback") from None
        finally:
            self._in_operation = False

    def _require_available(self):
        if self._closed:
            raise ValueError("Caption event bridge is closed")
        if self._in_operation:
            raise ValueError("Caption callbacks cannot re-enter the same stream")

    def finish(self):
        self._require_available()
        self._in_operation = True
        try:
            updates = self._stream.finish()
            now = self._now()
            end = now if self._last_frame_end_ns is None else self._last_frame_end_ns
            if now < end:
                raise ValueError("Monotonic clock moved backwards")
            self._publish(updates, now, end)
        except Exception:
            self.abort()
            raise ValueError("Research caption stream ended without fallback") from None
        finally:
            self._closed = True
            self._in_operation = False

    def abort(self):
        if not self._closed:
            self._closed = True
            try:
                self._stream.abort()
            except Exception:
                # Do not expose driver errors, audio or backend-local paths.
                pass
