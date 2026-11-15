"""Bounded live-audio pipeline. No GUI state or secrets live in this module.

Capture, speech recognition and translation run independently. If a slow model
falls behind live playback, the oldest queued chunks are discarded and the UI
is notified rather than accumulating unbounded latency.
"""
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import monotonic
from typing import Callable

from .audio import AudioCapture, SpeechRecognizer, audible
from .pipeline import Pipeline


@dataclass(frozen=True)
class AudioChunk:
    index: int
    pcm: bytes
    captured_at: float


@dataclass(frozen=True)
class SpokenChunk:
    index: int
    original: str
    language: str
    captured_at: float


class LiveSession:
    """Owns a single, isolated listening session.

    callback receives events as (kind, *values). These originate on worker
    threads; GUI callers must marshal them to their main/event-loop thread.
    """

    def __init__(self, source: str, source_language: str = "auto", target_language: str = "en",
                 topic: str = "general", glossary: dict[str, str] | None = None,
                 contextual: bool = True, chunk_seconds: float = 4.0,
                 on_event: Callable[..., None] | None = None,
                 audio_factory=None, recognizer_factory=None, pipeline_factory=None,
                 queue_size: int = 3, sound_gate: Callable[[bytes], bool] = audible):
        if not source or not 1 <= chunk_seconds <= 12 or not 1 <= queue_size <= 10:
            raise ValueError("Invalid source, chunk length or queue capacity")
        self.source = source
        self.source_language = source_language
        self.target_language = target_language
        self.topic = topic
        self.glossary = dict(glossary or {})
        self.contextual = contextual
        self.chunk_seconds = chunk_seconds
        self._event = on_event or (lambda *args: None)
        self._audio_factory = audio_factory or AudioCapture
        self._recognizer_factory = recognizer_factory or SpeechRecognizer
        self._pipeline = (pipeline_factory or Pipeline)()
        self._sound_gate = sound_gate
        self._audio: Queue[AudioChunk] = Queue(maxsize=queue_size)
        self._speech: Queue[SpokenChunk] = Queue(maxsize=queue_size)
        self._stop = Event()
        self._reset = Event()
        self._lock = Lock()
        self._reset_generation = 0
        self._reset_at = 0.0
        self._capture = None
        self._started = False
        self._workers: list[Thread] = []
        self._dropped = 0

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    @property
    def dropped_chunks(self) -> int:
        return self._dropped

    def _emit(self, kind: str, *values):
        try:
            self._event(kind, *values)
        except Exception:
            # A UI that has been closed must not strand capture threads.
            pass

    def start(self):
        with self._lock:
            if self._started:
                raise RuntimeError("This listening session has already started")
            self._started = True
            self._workers = [
                Thread(target=self._capture_loop, name="bridgecaption-capture", daemon=True),
                Thread(target=self._asr_loop, name="bridgecaption-asr", daemon=True),
                Thread(target=self._translation_loop, name="bridgecaption-translation", daemon=True),
            ]
            for worker in self._workers:
                worker.start()
            Thread(target=self._finish, name="bridgecaption-finish", daemon=True).start()
        return self

    def stop(self):
        """Request shutdown promptly. Never block a GUI on external processes."""
        self._stop.set()
        with self._lock:
            capture = self._capture
        if capture is not None:
            Thread(target=capture.stop, name="bridgecaption-stop-audio", daemon=True).start()

    def reset_context(self):
        """Forget old context and pending, pre-reset speech without stopping capture."""
        with self._lock:
            self._reset_generation += 1
            self._reset_at = monotonic()
            self._reset.set()
        # Queued frames from the former context must not be sent to Gemini.
        for pending in (self._audio, self._speech):
            while True:
                try:
                    pending.get_nowait()
                except Empty:
                    break

    def _offer(self, queue: Queue, item):
        if self._stop.is_set():
            return
        try:
            queue.put_nowait(item)
            return
        except Full:
            pass
        try:
            queue.get_nowait()
        except Empty:
            pass
        try:
            queue.put_nowait(item)
        except Full:
            return
        self._dropped += 1
        self._emit("backlog", self._dropped)

    def _fail(self, phase: str, error: Exception):
        if not self._stop.is_set():
            self._emit("error", f"{phase}: {error}")
        self._stop.set()
        with self._lock:
            capture = self._capture
        if capture is not None:
            try:
                capture.stop()
            except Exception:
                pass

    def _capture_loop(self):
        capture = None
        try:
            capture = self._audio_factory(self.source, self.chunk_seconds).start()
            with self._lock:
                self._capture = capture
            if self._stop.is_set():
                return
            self._emit("status", "Listening to selected audio source")
            for index, pcm in enumerate(capture.chunks()):
                if self._stop.is_set():
                    break
                if self._sound_gate(pcm):
                    self._offer(self._audio, AudioChunk(index, pcm, monotonic()))
            if not self._stop.is_set():
                self._fail("Audio source disconnected", RuntimeError("Source stopped providing audio"))
        except Exception as exc:
            self._fail("Audio capture", exc)
        finally:
            if capture is not None:
                try:
                    capture.stop()
                except Exception:
                    pass

    def _asr_loop(self):
        try:
            self._emit("status", "Loading speech model")
            model = self._recognizer_factory()
            while not self._stop.is_set():
                try:
                    chunk = self._audio.get(timeout=.2)
                except Empty:
                    continue
                original, lang = model.transcribe(chunk.pcm, self.source_language)
                if self._stop.is_set():
                    break
                with self._lock:
                    stale_audio = chunk.captured_at < self._reset_at
                if stale_audio:
                    # Neither the transcript nor its translation may reappear after reset.
                    continue
                original = " ".join(original.split())[:1500]
                if not original:
                    continue
                self._emit("heard", original, lang)
                self._offer(self._speech, SpokenChunk(chunk.index, original, lang, chunk.captured_at))
        except Exception as exc:
            self._fail("Speech recognition", exc)

    def _translation_loop(self):
        while not self._stop.is_set():
            try:
                speech = self._speech.get(timeout=.2)
            except Empty:
                continue
            with self._lock:
                generation = self._reset_generation
                reset_at = self._reset_at
                reset_pending = self._reset.is_set()
            if reset_pending:
                self._pipeline.reset()
                with self._lock:
                    if self._reset_generation == generation:
                        self._reset.clear()
            if speech.captured_at < reset_at:
                # Speech recorded before CLEAR CONTEXT is no longer permitted.
                continue
            try:
                result = self._pipeline.translate(
                    speech.original, speech.language, self.target_language,
                    self.topic, self.glossary, self.contextual)
            except Exception as exc:
                # A transient remote translation error should not end listening.
                if not self._stop.is_set():
                    self._emit("error", f"Translation: {exc}")
                continue
            with self._lock:
                stale = self._reset_generation != generation
            if stale:
                # The request was already in flight; suppress stale captions
                # and make sure its old context does not survive the reset.
                self._pipeline.reset()
                continue
            if self._stop.is_set():
                break
            self._emit("caption", result.original, result.translated,
                       result.source_language, round(monotonic() - speech.captured_at, 2),
                       getattr(result, "context_info", "unknown"))

    def _finish(self):
        for worker in self._workers:
            worker.join()
        self._pipeline.reset()
        self._emit("stopped", self._dropped)
