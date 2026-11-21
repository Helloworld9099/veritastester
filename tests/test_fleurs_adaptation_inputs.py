from hashlib import sha256
import json

import pytest

from research.unified_models import fleurs_adaptation_inputs as prep


def row(i, filename, text):
    return f"{i}\t{filename}.wav\t{text}\tnormalized\tx\t16000\tOTHER\n".encode()


@pytest.fixture
def tables():
    return {
        ("en_us", "train"): row(1, 111, "The cat sat") + row(2, 112, "Trees grow"),
        ("en_us", "dev"): row(3, 113, "A gull flies"),
        ("es_419", "train"): row(1, 211, "El gato se sentó") + row(4, 214, "Sin referencia"),
        ("es_419", "dev"): row(3, 213, "Una gaviota vuela"),
        ("fr_fr", "train"): row(2, 312, "Les arbres poussent"),
        ("fr_fr", "dev"): row(3, 313, "Une mouette vole"),
    }


def test_same_split_parallel_join_preserves_raw_labels(tables):
    bundles, coverage = prep._prepare(tables)
    es, fr = bundles["train"]
    assert (es["audio_member"], es["source_transcript"], es["english_reference"]) == (
        "211.wav", "El gato se sentó", "The cat sat")
    assert (fr["audio_member"], fr["english_reference"]) == ("312.wav", "Trees grow")
    assert len(bundles["dev"]) == 2
    assert coverage["es->en"]["train"] == {
        "audio_rows": 2, "paired_rows": 1, "missing_english_rows": 1}


@pytest.mark.parametrize(("key", "extra", "reason"), [
    (("fr_fr", "dev"), row(1, 390, "Le chat"), "cross_split_parallel_sentence_id"),
    (("fr_fr", "dev"), row(5, 211, "Ce chien"), "cross_split_audio_member_name"),
    (("fr_fr", "dev"), row(5, 399, "The CAT. SAT!"), "cross_split_exact_normalized_text"),
    (("en_us", "dev"), row(3, 114, "The wrong translation"), "ambiguous_sentence_reference"),
    (("en_us", "dev"), row(3, 113, "A gull flies"), "duplicate_audio_member"),
    (("es_419", "dev"), b"1\t../214.wav\tprivate",
     "invalid_fleurs_fields"),
])
def test_bad_rows_fail_closed_without_data(tables, key, extra, reason):
    tables[key] += extra
    with pytest.raises(prep.InputError, match=f"^{reason}$"):
        prep._prepare(tables)


def test_missing_reference_never_falls_back_across_splits(tables):
    tables["fr_fr", "dev"] = row(10, 313, "Une mouette vole")
    with pytest.raises(prep.InputError, match="missing_split_direction"):
        prep._prepare(tables)


def test_extra_holdout_table_rejected(tables):
    tables["en_us", "test"] = b"not read"
    with pytest.raises(prep.InputError, match="only_six_train_dev_tables_allowed"):
        prep._prepare(tables)


def test_bound_export_redacts_receipt_and_does_not_open_holdout(tables, tmp_path, monkeypatch):
    local = tmp_path / "tables"
    local.mkdir()
    monkeypatch.setattr(prep, "_PINS", {key: sha256(raw).hexdigest() for key, raw in tables.items()})
    for (lang, split), raw in tables.items():
        (local / f"{lang}-{split}.tsv").write_bytes(raw)
    (local / "en_us-test.tsv").symlink_to("nonexistent")
    output = tmp_path / "local-output"
    result = prep.run(local, export_dir=output)
    assert result["holdout_opened"] is False
    assert result["audio_bytes_checked"] is False
    assert "gato" not in json.dumps(result)
    assert result["manifest_sha256"]["train"] == sha256((output / "train.jsonl").read_bytes()).hexdigest()
    assert prep.run(local) == result
    with pytest.raises(FileExistsError):
        prep.run(local, export_dir=output)
    (local / "en_us-train.tsv").write_bytes(row(1, 111, "Changed after pin"))
    with pytest.raises(prep.InputError, match="publisher_tsv_pin_mismatch"):
        prep.run(local)


def test_local_export_cannot_write_into_checkout(tmp_path, monkeypatch):
    table_path = tmp_path / "checkout" / "research" / "unified_models" / "tool.py"
    monkeypatch.setattr(prep, "__file__", str(table_path))
    relative_output = tmp_path / "checkout" / "dump"
    monkeypatch.setattr(prep, "_public_tsv", lambda _: None)
    monkeypatch.setattr(prep, "_prepare", lambda _: ({"train": [], "dev": []}, {}))
    with pytest.raises(prep.InputError, match="use_new_local_directory_outside_checkout"):
        prep.run(tmp_path, export_dir=relative_output)
    assert not relative_output.exists()
