"""Regression checks for source separation and safe acquisition, no downloads."""
import hashlib
from pathlib import Path
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
clean = runpy.run_path(str(ROOT / "scripts/run_offline_research.py"))["decontaminate"]
acquire = runpy.run_path(str(ROOT / "scripts/acquire_datasets.py"))["acquire"]


def row(name, split, doc, text):
    return {"case_id": name, "split": split, "extra": {"document_id": doc}, "utterance": text}


def test_shared_documents_remove_training_rows_without_moving_holdout():
    rows = [row("train", "train", "shared", "Training example words"),
            row("dev", "dev", "shared", "Development example words"),
            row("test", "test", "shared", "Held out example words"),
            row("independent", "train", "independent", "Independent training words")]
    result, report = clean(rows)
    assert [r["case_id"] for r in result] == ["test", "independent"]
    assert result[0]["split"] == "test"
    assert report["retained"] == 2


def test_punctuation_and_near_duplicate_leakage_is_removed():
    text = " ".join("token"+str(i) for i in range(100))
    rows = [row("train", "train", "a", text+" added"),
            row("dev", "dev", "b", text), row("test", "test", "c", text+"!")]
    result, _ = clean(rows)
    assert [r["case_id"] for r in result] == ["test"]


def test_tampered_existing_download_is_rejected_without_network(tmp_path):
    (tmp_path / "source.json").write_bytes(b"changed")
    manifest = {"file": "source.json", "bytes": 4, "sha256": hashlib.sha256(b"real").hexdigest()}
    with pytest.raises(ValueError, match="differs"):
        acquire(manifest, tmp_path)


def test_verified_download_is_reused_without_network(tmp_path):
    (tmp_path / "source.json").write_bytes(b"real")
    manifest = {"file": "source.json", "bytes": 4, "sha256": hashlib.sha256(b"real").hexdigest()}
    assert acquire(manifest, tmp_path) == "verified-existing"
