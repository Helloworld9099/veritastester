import csv
import json

import pytest

from bridgecaption import evaluate_audio


def manifest(tmp_path, rows):
    for row in rows:
        (tmp_path / row["file"]).write_bytes(b"fake-wav")
    path = tmp_path / "examples.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["file", "language", "transcript",
                                                  "translation", "previous_json", "topic"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def entry(name="clip.wav", language="es", transcript="Hola mundo"):
    return {"file": name, "language": language, "transcript": transcript,
            "translation": "Hello world", "previous_json": '["An older sentence"]',
            "topic": "general"}


def test_manifest_and_grouped_replay_without_cloud(tmp_path, monkeypatch):
    f = manifest(tmp_path, [entry(), entry("clip2.wav", "en", "Hello world")])
    rows = evaluate_audio.read_manifest(f)
    assert len(rows) == 2
    monkeypatch.setattr(evaluate_audio, "decode_audio", lambda path: b"test-pcm")
    class FakeRecognizer:
        def transcribe(self, raw, lang):
            return ("Hola mundo" if lang == "es" else "Hello world"), lang
    outputs = evaluate_audio.run_replay(rows, FakeRecognizer())
    assert len(outputs) == 2
    assert all(item["asr_errors"] == 0 for item in outputs)
    summary = evaluate_audio.summarize(outputs)
    assert summary["es"]["asr_error_rate"] == 0
    assert summary["en"]["clips"] == 1
    assert all(item["translated"] is None for item in outputs)


def test_replay_translation_does_not_share_history_between_clips(tmp_path, monkeypatch):
    f = manifest(tmp_path, [entry(), entry("clip2.wav")])
    rows = evaluate_audio.read_manifest(f)
    monkeypatch.setattr(evaluate_audio, "decode_audio", lambda _: b"pcm")
    class FakeRecognizer:
        def transcribe(self, raw, lang):
            return "Hola mundo", "es"
    class FakeTranslator:
        supports_context = True
        def __init__(self):
            self.requests = []
        def translate(self, req):
            self.requests.append(req)
            return "Hello world"
    translator = FakeTranslator()
    result = evaluate_audio.run_replay(
        rows, FakeRecognizer(), translator=translator, contextual=True, context_mode="all")
    assert len(result) == 2
    assert len(translator.requests) == 2
    assert translator.requests[0].previous == ["An older sentence"]
    assert translator.requests[1].previous == ["An older sentence"]
    assert evaluate_audio.summarize(result)["es"]["translation_median_seconds"] is not None


def test_reject_absolute_or_parent_traversal(tmp_path):
    p = manifest(tmp_path, [entry()])
    (tmp_path / "examples.csv").write_text(
        "file,language,transcript\n../outside.wav,es,hola\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside dataset folder"):
        evaluate_audio.read_manifest(p)


def test_reject_invalid_context_json(tmp_path):
    p = manifest(tmp_path, [{**entry(), "previous_json": '{"not":"a list"}'}])
    with pytest.raises(ValueError, match="previous_json must contain"):
        evaluate_audio.read_manifest(p)


def test_percentile_one_element():
    assert evaluate_audio.percentile([2], .95) == 2


def test_cloud_replay_requires_explicit_consent_flag(tmp_path, monkeypatch):
    import sys
    f = manifest(tmp_path, [entry()])
    monkeypatch.setattr(sys, "argv", [
        "bridgecaption-eval-audio", "--manifest", str(f), "--execute",
        "--translate", "--engine", "azure"])
    from bridgecaption import audio
    monkeypatch.setattr(audio, "SpeechRecognizer", lambda *_: (_ for _ in ()).throw(
        AssertionError("must block before loading model")))
    with pytest.raises(SystemExit) as exc:
        evaluate_audio.main()
    assert exc.value.code == 2


def test_manifest_dry_run_skips_audio_model_and_translation(tmp_path, monkeypatch, capsys):
    import sys
    f = manifest(tmp_path, [entry()])
    monkeypatch.setattr(sys, "argv", ["bridgecaption-eval-audio", "--manifest", str(f)])
    from bridgecaption import audio
    monkeypatch.setattr(audio, "SpeechRecognizer", lambda *_: (_ for _ in ()).throw(
        AssertionError("dry run must not load model")))
    evaluate_audio.main()
    assert "DRY RUN" in capsys.readouterr().out
