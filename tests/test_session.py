"""Pipeline tests use fake audio, ASR and translation. No API key or models required."""
import time
import threading
from bridgecaption.session import LiveSession, AudioChunk
from bridgecaption.pipeline import Caption


class FakeCapture:
    def __init__(self, name, duration, clips=None):
        self.clips = list(clips if clips is not None else [b'a', b'b', b'c'])
        self.stopped = threading.Event()
        self.done = threading.Event()

    def start(self):
        return self

    def chunks(self):
        for clip in self.clips:
            if self.stopped.is_set():
                break
            yield clip
        self.done.wait(2)

    def stop(self):
        self.stopped.set()
        self.done.set()


class FakeASR:
    def transcribe(self, data, source):
        return data.decode(), 'es'


class FakePipeline:
    instances = []
    def __init__(self):
        self.history = []
        self.requests = []
        self.cleared = False
        self.__class__.instances.append(self)

    def translate(self, text, source, target, topic, glossary, contextual):
        self.requests.append((text, self.history[:]))
        self.history.append(text)
        return Caption(text, 'translated:' + text, source, topic)

    def reset(self):
        self.history.clear()
        self.cleared = True


def wait_for(predicate, seconds=2):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(.01)
    return False


def test_capture_asr_translate_workers_and_shutdown():
    events = []
    session = LiveSession('mock', on_event=lambda *v: events.append(v),
        audio_factory=lambda name, duration: FakeCapture(name, duration),
        recognizer_factory=FakeASR, pipeline_factory=FakePipeline,
        sound_gate=lambda b: True).start()
    assert wait_for(lambda: any(event[0] == 'caption' for event in events))
    session.stop()
    assert wait_for(lambda: any(event[0] == 'stopped' for event in events))
    assert FakePipeline.instances[-1].cleared
    assert any(event[0] == 'heard' for event in events)


def test_session_context_is_never_reused():
    first = LiveSession('mock', on_event=lambda *a: None,
        audio_factory=lambda s,d: FakeCapture(s,d,[b'private']),
        recognizer_factory=FakeASR, pipeline_factory=FakePipeline,
        sound_gate=lambda b: True).start()
    assert wait_for(lambda: bool(FakePipeline.instances[-1].requests))
    first.stop()
    assert wait_for(lambda: FakePipeline.instances[-1].cleared)
    second = LiveSession('mock', on_event=lambda *a: None,
        audio_factory=lambda s,d: FakeCapture(s,d,[b'new']),
        recognizer_factory=FakeASR, pipeline_factory=FakePipeline,
        sound_gate=lambda b: True).start()
    assert wait_for(lambda: bool(FakePipeline.instances[-1].requests))
    assert FakePipeline.instances[-1].requests[0][1] == []
    second.stop()


def test_drop_oldest_when_queue_full_is_bounded():
    session = LiveSession('mock', on_event=lambda *a: None,
        audio_factory=FakeCapture, recognizer_factory=FakeASR, pipeline_factory=FakePipeline,
        queue_size=2)
    session._offer(session._audio, AudioChunk(0,b'a',0))
    session._offer(session._audio, AudioChunk(1,b'b',0))
    session._offer(session._audio, AudioChunk(2,b'c',0))
    assert session._audio.qsize() == 2
    assert session._audio.get_nowait().index == 1
    assert session.dropped_chunks == 1


def test_session_start_only_once():
    s = LiveSession('mock', on_event=lambda *a: None,
        audio_factory=lambda n,d: FakeCapture(n,d),
        recognizer_factory=FakeASR, pipeline_factory=FakePipeline,
        sound_gate=lambda b: True).start()
    try:
        s.start()
        assert False
    except RuntimeError:
        pass
    finally:
        s.stop()


def test_invalid_audio_settings():
    import pytest
    with pytest.raises(ValueError):
        LiveSession('', chunk_seconds=4)
    with pytest.raises(ValueError):
        LiveSession('mock', chunk_seconds=0)


def test_continuous_capture_while_translator_is_blocked():
    """The microphone must keep draining even when a remote API call stalls."""
    begin = threading.Event()
    release = threading.Event()

    class StalledPipeline(FakePipeline):
        def translate(self, *args):
            begin.set()
            release.wait(2)
            return super().translate(*args)

    class BusyCapture(FakeCapture):
        def chunks(self):
            yield b'first'
            if not begin.wait(2):
                return
            for i in range(12):
                yield f'clip{i}'.encode()
            self.done.wait(2)

    events = []
    session = LiveSession('mock', on_event=lambda *v: events.append(v),
        audio_factory=BusyCapture, recognizer_factory=FakeASR,
        pipeline_factory=StalledPipeline, queue_size=2,
        sound_gate=lambda b: True).start()
    assert wait_for(lambda: begin.is_set())
    assert wait_for(lambda: session.dropped_chunks > 0)
    assert session._audio.qsize() <= 2 and session._speech.qsize() <= 2
    release.set()
    assert wait_for(lambda: any(e[0] == 'caption' for e in events))
    session.stop()
    assert wait_for(lambda: any(e[0] == 'stopped' for e in events))
