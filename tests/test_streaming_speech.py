"""Synthetic tests for the disabled-by-default speech contract. No real audio/model execution."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import unittest

from research.unified_models.streaming import (
    CaptionUpdate,
    PCMFrame,
    PCM_FORMAT,
    SingleStreamingSpeechRuntime,
    SpeechQualification,
    StreamError,
    StreamLimits,
)

CHECKPOINT = "2" * 64
PCM = b"\0\0"


def receipt():
    return {
        "schema_version": 1, "kind": "reviewed_paired_streaming_speech_dev",
        "scope": "research_only", "checkpoint_sha256": CHECKPOINT,
        "pcm_format": PCM_FORMAT, "shared_audio_encoder": True,
        "directions": {r: {"status": "passed", "evidence_sha256": "3" * 64}
                       for r in ("es->en", "fr->en", "en->es")},
    }


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def admission(value=None):
    value = receipt() if value is None else value
    return SpeechQualification(value, expected_sha256=digest(value))


def cap(ident=0, *, final=False, original="bonjour", translated="hello"):
    return CaptionUpdate(utterance_id=ident, original=original, translated=translated, final=final)


class FakeDecoder:
    def __init__(self, *push_batches, flush=()):
        self.calls = []
        self.push_batches = iter(push_batches)
        self.flush = flush
        self.aborts = 0

    def push(self, pcm):
        self.calls.append(pcm)
        batch = next(self.push_batches, ())
        if isinstance(batch, Exception):
            raise batch
        return batch

    def finish(self):
        if isinstance(self.flush, Exception):
            raise self.flush
        return self.flush

    def abort(self):
        self.aborts += 1


class FakeBackend:
    def __init__(self, *decoders, checkpoint=CHECKPOINT):
        self.checkpoint_sha256 = checkpoint
        self.calls = []
        self.decoders = iter(decoders)

    def open_stream(self, *, source, target):
        self.calls.append((source, target))
        return next(self.decoders)


class StreamingTests(unittest.TestCase):
    def build_runtime(self, *decoders, limits=None):
        backend = FakeBackend(*decoders)
        loaded = []
        def factory(pin):
            loaded.append(pin)
            return backend
        runtime = SingleStreamingSpeechRuntime(qualification=admission(),
            load_backend=factory, opt_in=True, limits=limits)
        return runtime, backend, loaded

    def test_default_and_unreviewed_routes_block_before_loading(self):
        runtime = SingleStreamingSpeechRuntime()
        self.assertEqual(runtime.status(), {
            "scope": "research_only", "enabled": False, "qualified_directions": [],
            "loaded_model_count": 0, "production_activated": False,
        })
        self.assertFalse(runtime.supports_context)
        with self.assertRaisesRegex(StreamError, "No opt-in"):
            runtime.start_stream(source="es", target="en")
        runtime, _, loaded = self.build_runtime(FakeDecoder())
        for route in (("auto", "en"), ("es", "fr"), ("es", "es"), ("cmn", "en")):
            with self.subTest(route=route), self.assertRaises(StreamError):
                runtime.start_stream(source=route[0], target=route[1])
        self.assertEqual(loaded, [])

    def test_admission_is_speech_specific_immutable_and_fail_closed(self):
        data = receipt()
        q = admission(data)
        data["directions"].clear()
        self.assertEqual(q.directions, {"es->en", "fr->en", "en->es"})
        for mutation in (
            lambda r: r.update(kind="independent_bilingual_direction_receipt"),
            lambda r: r.update(scope="production"),
            lambda r: r.update(pcm_format="mp3"),
            lambda r: r.update(shared_audio_encoder=False),
            lambda r: r["directions"]["es->en"].update(status="blocked"),
            lambda r: r["directions"]["es->en"].update(other_model=CHECKPOINT),
            lambda r: r["directions"].update({"fr->fr": r["directions"]["es->en"]}),
        ):
            mutated = deepcopy(receipt())
            mutation(mutated)
            with self.subTest(mutated=mutated), self.assertRaises(StreamError):
                admission(mutated)
        with self.assertRaisesRegex(StreamError, "checksum"):
            SpeechQualification(receipt(), expected_sha256="a"*64)

    def test_paired_provisional_revisions_final_and_explicit_direction_change(self):
        first = FakeDecoder([cap()], (cap(original="bonjour à tous", translated="hello everyone"),),
                            (cap(final=True, original="bonjour à tous", translated="hello everyone"),),
                            flush=(cap(1, final=True),))
        second = FakeDecoder()
        runtime, backend, loaded = self.build_runtime(first, second)
        s = runtime.start_stream(source="fr", target="en")
        self.assertEqual(s.direction, ("fr", "en"))
        self.assertEqual(s.state, "open")
        self.assertEqual(s.push(PCMFrame(0, PCM)), (cap(),))
        self.assertEqual(s.push(PCMFrame(1, PCM))[0].original, "bonjour à tous")
        self.assertTrue(s.push(PCMFrame(2, PCM))[0].final)
        with self.assertRaisesRegex(StreamError, "before changing"):
            runtime.start_stream(source="es", target="en")
        self.assertEqual(s.finish(), (cap(1, final=True),))
        self.assertEqual(s.state, "finished")
        with self.assertRaisesRegex(StreamError, "not open"):
            s.push(PCMFrame(3, PCM))
        next_stream = runtime.start_stream(source="es", target="en")
        self.assertEqual(next_stream.push(PCMFrame(0, PCM)), ())
        next_stream.abort()
        next_stream.abort()
        self.assertEqual(next_stream.state, "aborted")
        self.assertEqual(second.aborts, 1)
        self.assertEqual(first.calls, [PCM]*3)
        self.assertEqual(backend.calls, [("fr", "en"), ("es", "en")])
        self.assertEqual(loaded, [CHECKPOINT])
        self.assertEqual(runtime.loaded_model_count, 1)
        self.assertFalse(runtime.status()["production_activated"])

    def test_invalid_pcm_or_sequence_never_advances_decoder(self):
        decoder = FakeDecoder()
        runtime, _, _ = self.build_runtime(decoder, limits=StreamLimits(max_frame_samples=2, max_stream_samples=3))
        stream = runtime.start_stream(source="en", target="es")
        for bad in (PCMFrame(1, PCM), PCMFrame(True, PCM), PCMFrame(0, b""), PCMFrame(0, b"x"),
                    PCMFrame(0, b"x"*6), PCMFrame(0, bytearray(PCM)), object()):
            with self.subTest(frame=bad), self.assertRaises(StreamError):
                stream.push(bad)
        self.assertEqual(decoder.calls, [])
        stream.push(PCMFrame(0, PCM*2))
        with self.assertRaisesRegex(StreamError, "budget exhausted"):
            stream.push(PCMFrame(1, PCM*2))
        stream.push(PCMFrame(1, PCM))
        self.assertEqual(decoder.calls, [PCM*2, PCM])
        self.assertEqual(stream.finish(), ())

    def test_atomic_rejection_and_permanent_final_ledger(self):
        first = FakeDecoder((cap(final=True),), (cap(1), cap(0, final=True)))
        second = FakeDecoder()
        runtime, _, loaded = self.build_runtime(first, second)
        stream = runtime.start_stream(source="fr", target="en")
        final, = stream.push(PCMFrame(0, PCM))
        self.assertTrue(final.final)
        with self.assertRaisesRegex(StreamError, "retracted"):
            stream.push(PCMFrame(1, PCM))
        self.assertEqual(stream.state, "failed")
        self.assertEqual(first.aborts, 1)
        with self.assertRaisesRegex(StreamError, "not open"):
            stream.finish()
        runtime.start_stream(source="es", target="en").abort()
        self.assertEqual(loaded, [CHECKPOINT])

    def test_invalid_or_unresolved_output_aborts_without_raw_text_leaks(self):
        for batch, flush in (
            ((cap(translated=""),), ()), ((cap(original="\ud800"),), ()),
            ((cap(ident=2, final=True),), ()), ((cap(final=True),cap(1,final=True)), ()),
            (RuntimeError("private text must not appear"), ()),
            ((cap(),), ()), ((), (cap(),)),
        ):
            with self.subTest(kind=type(batch)):
                decoder = FakeDecoder(batch, flush=flush)
                runtime, _, _ = self.build_runtime(decoder, limits=StreamLimits(max_events_per_push=1))
                s = runtime.start_stream(source="es", target="en")
                with self.assertRaises(StreamError) as error:
                    s.push(PCMFrame(0, PCM)); s.finish()
                self.assertNotIn("private text", str(error.exception))
                self.assertEqual(s.state, "failed")
                self.assertEqual(decoder.aborts, 1)
        self.assertNotIn("bonjour", repr(cap()))
        self.assertNotIn("raw audio", repr(PCMFrame(0, b"raw audio")))

    def test_provisional_budget_and_max_utterances(self):
        decoder = FakeDecoder((cap(),), (cap(),))
        runtime, _, _ = self.build_runtime(decoder, limits=StreamLimits(max_updates_per_utterance=1))
        s = runtime.start_stream(source="es", target="en")
        s.push(PCMFrame(0, PCM))
        with self.assertRaisesRegex(StreamError, "Too many provisional"):
            s.push(PCMFrame(1, PCM))
        decoder = FakeDecoder((cap(final=True),), (cap(1, final=True),))
        runtime, _, _ = self.build_runtime(decoder, limits=StreamLimits(max_utterances=1))
        s = runtime.start_stream(source="es", target="en")
        s.push(PCMFrame(0, PCM))
        with self.assertRaises(StreamError):
            s.push(PCMFrame(1, PCM))

    def test_checkpoint_mismatch_or_loader_failure_is_not_a_reload(self):
        for fails in (False, True):
            called = []
            def factory(pin):
                called.append(pin)
                if fails:
                    raise OSError("private location")
                return FakeBackend(checkpoint="a" * 64)
            runtime = SingleStreamingSpeechRuntime(qualification=admission(), opt_in=True, load_backend=factory)
            with self.assertRaises(StreamError) as failure:
                runtime.start_stream(source="es",target="en")
            self.assertNotIn("private location", str(failure.exception))
            with self.assertRaisesRegex(StreamError, "already attempted"):
                runtime.start_stream(source="fr",target="en")
            self.assertEqual(called,[CHECKPOINT])
            self.assertEqual(runtime.loaded_model_count,0)

    def test_reported_checkpoint_change_on_direction_open_invalidates_same_runtime(self):
        runtime, backend, loaded = self.build_runtime(FakeDecoder())
        runtime.start_stream(source="es", target="en").finish()
        backend.checkpoint_sha256 = "a"*64
        with self.assertRaisesRegex(StreamError, "identity changed"):
            runtime.start_stream(source="fr", target="en")
        backend.checkpoint_sha256 = CHECKPOINT
        with self.assertRaisesRegex(StreamError, "was invalidated"):
            runtime.start_stream(source="fr", target="en")
        self.assertEqual(backend.calls, [("es", "en")])
        self.assertEqual(loaded, [CHECKPOINT])

    def test_concurrent_start_does_not_open_multiple_backend_states(self):
        runtime, backend, loaded = self.build_runtime(FakeDecoder())
        def start(args):
            try:
                return runtime.start_stream(source=args[0],target=args[1])
            except StreamError:
                return None
        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(start, [("es", "en"), ("fr", "en")]))
        accepted = [result for result in outcomes if result is not None]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(loaded, [CHECKPOINT])
        accepted[0].abort()


if __name__ == "__main__":
    unittest.main()
