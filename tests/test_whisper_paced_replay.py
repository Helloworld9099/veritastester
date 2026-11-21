import hashlib
import json

import pytest

from research.unified_models import whisper_paced_replay as replay


def _operator_rows():
    prior = json.loads(replay._PILOT_INPUTS.read_bytes())
    return [
        {"id": row["id"], "language": row["source_language"], "wav": f"{row['id']}.wav",
         "source": "synthetic source", "reference": "synthetic reference",
         "sha256": row["normalized_pcm16_wav_sha256"], "duration": row["seconds"]}
        for row in prior["clips"]
    ]


def _write(tmp_path, rows):
    f = tmp_path / "operator.json"
    raw = json.dumps(rows).encode()
    f.write_bytes(raw)
    return f, hashlib.sha256(raw).hexdigest()


def _not_opened(*_args):
    pytest.fail("unverified input must not reach an audio path")


def test_wrong_independent_manifest_pin_rejected_before_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "checked_wav", _not_opened)
    f, _ = _write(tmp_path, _operator_rows())
    with pytest.raises(replay.DiagnosticError, match="fixture_manifest_sha256_mismatch"):
        replay._verified_inputs(f, tmp_path, "0" * 64, 1)


def test_self_pinned_audio_substitution_rejected_before_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "checked_wav", _not_opened)
    rows = _operator_rows()
    rows[-1]["sha256"] = "0" * 64
    f, sha = _write(tmp_path, rows)
    with pytest.raises(replay.DiagnosticError, match="dev_pilot_clip_pin_mismatch"):
        replay._verified_inputs(f, tmp_path, sha, 1)
