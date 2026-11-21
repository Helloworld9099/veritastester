import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.unified_models import nemotron_cache_smoke as smoke


def test_chunk_plan_covers_nonoverlapping_frames_and_only_pads_eof():
    assert list(smoke.chunk_plan(95, 25, 32)) == [(0, 25, 0), (25, 57, 0),
                                                    (57, 89, 0), (89, 95, 26)]
    assert list(smoke.chunk_plan(25, 25, 32)) == [(0, 25, 0)]
    assert list(smoke.chunk_plan(2, 25, 32)) == [(0, 2, 23)]
    with pytest.raises(smoke.DiagnosticError, match="invalid_mel"):
        list(smoke.chunk_plan(True, 25, 32))


def test_manifest_pin_checked_before_reading_wavs(tmp_path, monkeypatch):
    manifest = tmp_path / "wrong.json"
    manifest.write_text('[{"wav": "out-of-scope.wav"}]')

    def forbid(_):
        pytest.fail("Opened WAV inventory before validating pinned manifest")

    monkeypatch.setattr(smoke.json, "loads", forbid)
    with pytest.raises(smoke.DiagnosticError, match="manifest_pin_mismatch"):
        smoke.verify_audio(manifest, tmp_path)


def test_all_model_files_are_required_and_byte_pins_checked(tmp_path, monkeypatch):
    expected = {"config.json": b"config", "model.safetensors": b"test bytes, not weights"}
    monkeypatch.setattr(smoke, "MODEL_PINS", {n: hashlib.sha256(b).hexdigest()
                                             for n, b in expected.items()})
    for n, b in expected.items():
        (tmp_path / n).write_bytes(b)
    assert smoke.verify_model(tmp_path)["model.safetensors"]["bytes"] == len(expected["model.safetensors"])
    (tmp_path / "alternative.bin").write_bytes(b"not allowed")
    with pytest.raises(smoke.DiagnosticError, match="unexpected_model_file_inventory"):
        smoke.verify_model(tmp_path)
    (tmp_path / "alternative.bin").unlink()
    (tmp_path / "config.json").write_bytes(b"mutated")
    with pytest.raises(smoke.DiagnosticError, match="model_file_pin_mismatch"):
        smoke.verify_model(tmp_path)


def test_pinned_model_file_may_not_be_symlink(tmp_path, monkeypatch):
    body = b"config"
    symlink_target = tmp_path / "different"
    symlink_target.write_bytes(body)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").symlink_to(symlink_target)
    monkeypatch.setattr(smoke, "MODEL_PINS", {"config.json": hashlib.sha256(body).hexdigest()})
    with pytest.raises(smoke.DiagnosticError, match="noncanonical_local_path"):
        smoke.verify_model(model_dir)


class Chunk:
    shape = (1, 4, 128)

    def __init__(self, pointer):
        self.pointer = pointer

    def data_ptr(self):
        return self.pointer


def call(witness, *, chunk, cache, padding, next_cache=None, next_padding=None):
    witness(None, (), dict(input_features=chunk, use_cache=True, past_key_values=cache,
                          padding_cache=padding), SimpleNamespace(
        past_key_values=next_cache if next_cache is not None else cache,
        padding_cache=next_padding if next_padding is not None else padding,
    ))


def test_witness_checks_input_identity_cache_reuse_and_progress():
    chunks = [Chunk(33), Chunk(34), Chunk(35)]
    watch = smoke.EncoderWitness(chunks)
    cached = SimpleNamespace(n=1)
    cached.get_seq_length = lambda: cached.n
    causal = object()
    call(watch, chunk=chunks[0], cache=None, padding=None, next_cache=cached, next_padding=causal)
    with pytest.raises(smoke.DiagnosticError, match="not_next_disjoint_mel_slice"):
        call(watch, chunk=chunks[0], cache=cached, padding=causal)
    with pytest.raises(smoke.DiagnosticError, match="not_passed"):
        call(watch, chunk=chunks[1], cache=cached, padding=object())
    with pytest.raises(smoke.DiagnosticError, match="did_not_advance"):
        call(watch, chunk=chunks[1], cache=cached, padding=causal)
    cached.n = 2
    call(watch, chunk=chunks[1], cache=cached, padding=causal)
    with pytest.raises(smoke.DiagnosticError, match="incomplete"):
        watch.summary()
    # Use a fresh witness to ensure failed inputs aren't counted as successful.
    watch = smoke.EncoderWitness(chunks)
    for i, chunk in enumerate(chunks):
        cached.n = 1+i
        call(watch, chunk=chunk, cache=cached if i else None, padding=causal if i else None,
             next_cache=cached, next_padding=causal)
    assert watch.summary() == {"encoder_forward_calls": 3, "acoustic_cache_reuses": 2,
                               "encoder_positions_processed": 3}


def test_report_refuses_existing_file_before_work(tmp_path, monkeypatch):
    report = tmp_path / "evidence.json"
    report.write_text("preserve me")
    monkeypatch.setattr(smoke, "run", lambda _: pytest.fail("Should refuse existing report before model loading"))
    with pytest.raises(smoke.DiagnosticError, match="existing_report_refused"):
        smoke.main(["--model-dir", str(tmp_path), "--manifest", str(tmp_path / "fleurs-dev.json"),
                    "--audio-root", str(tmp_path), "--report", str(report)])
    assert report.read_text() == "preserve me"
