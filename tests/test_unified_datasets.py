"""Synthetic fixtures only: no BridgeCaption corpus, benchmark, or model use."""

import csv
import hashlib
import json
from dataclasses import replace

import pytest

from research.unified_models.datasets import (
    DatasetError,
    DatasetSpec,
    audit_split_integrity,
    convert_dataset,
    load_bundle,
    training_rows,
)


BASE_MAP = {"original_id": "legacy_id", "source": "utterance", "target": "translation", "split": "role"}
RICH_MAP = {
    **BASE_MAP,
    "group_id": "session",
    "speaker_id": "speaker_code",
    "dialect": "variety",
    "context": "previous_turns",
    "context_required": "needed",
    "slang": "slang",
    "idiom": "idiom",
    "slang_category": "register",
    "noisy_transcription": "asr_noise",
    "source_basis": "origin",
    "verification_status": "status",
    "holdout": "sealed_role",
    "tags": "labels",
}


def spec(dataset="fixture", version="v1", **kw):
    values = dict(
        dataset_id=dataset,
        version=version,
        source_uri="synthetic M1 fixture",
        license="CC BY 4.0",
        rights="private_research",
        rights_evidence="fixture permission declaration",
        privacy="private",
        source_language="zh",
        target_language="en",
    )
    values.update(kw)
    return DatasetSpec(**values)


def source_row(legacy_id, utterance, translation, role="train", **kw):
    return dict(legacy_id=legacy_id, utterance=utterance, translation=translation, role=role, **kw)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def fixture_bundle(tmp_path, name, rows, settings=None, mapping=BASE_MAP):
    source = write_jsonl(tmp_path / f"{name}.jsonl", rows)
    destination = tmp_path / name
    convert_dataset(source, destination, settings or spec(name), mapping)
    pin = hashlib.sha256((destination / "manifest.json").read_bytes()).hexdigest()
    return load_bundle(destination, original_source=source, expected_manifest_sha256=pin), pin


def train(*bundles_and_pins, purpose="private_research"):
    return training_rows(
        (bundle for bundle, _ in bundles_and_pins),
        purpose=purpose,
        expected_manifest_sha256s=(pin for _, pin in bundles_and_pins),
    )


def test_round_trip_is_deterministic_and_keeps_original_data(tmp_path):
    row = source_row(
        902,
        "  这句带全角标点。\n\u3000下一行！  ",
        " Keep my exact spacing.\t",
        session="pseudonymous-session-22",
        speaker_code=13,
        variety="An explicitly assigned fixture variety",
        previous_turns=[" 第一段原话 ", "第二段\n仍是同一轮"],
        needed=True,
        slang=True,
        idiom=False,
        register="casual",
        asr_noise=False,
        origin="synthetic",
        status="independent_review",
        sealed_role=False,
        labels=["pragmatics", "specific quantity"],
    )
    original = write_jsonl(tmp_path / "raw.jsonl", [row])
    input_bytes = original.read_bytes()
    a, b = tmp_path / "first", tmp_path / "second"
    manifest = convert_dataset(original, a, spec(), RICH_MAP)
    assert convert_dataset(original, b, spec(), RICH_MAP) == manifest
    assert (a / "records.jsonl").read_bytes() == (b / "records.jsonl").read_bytes()
    assert (a / "manifest.json").read_bytes() == (b / "manifest.json").read_bytes()
    assert original.read_bytes() == input_bytes
    assert manifest["conversion"]["source_sha256"] == hashlib.sha256(input_bytes).hexdigest()
    assert manifest["counts"]["by_language_pair"] == {"zh->en": 1}
    assert manifest["counts"]["slang"] == 1
    bundle = load_bundle(a, original_source=original)
    new = bundle.records[0]
    assert new["schema_version"] == 1
    assert type(new["original_id"]) is int and new["original_id"] == 902
    for target, source in RICH_MAP.items():
        assert new[target] == row[source]
    assert new["source_language"] == "zh" and new["target_language"] == "en"
    assert a.stat().st_mode & 0o077 == 0
    assert all((a / p).stat().st_mode & 0o077 == 0 for p in ("records.jsonl", "manifest.json"))


@pytest.mark.parametrize(("ext", "delimiter"), (("csv", ","), ("tsv", "\t")))
def test_delimited_input_keeps_leading_zeros_unicode_and_newlines(tmp_path, ext, delimiter):
    raw = tmp_path / f"input.{ext}"
    with raw.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream, delimiter=delimiter)
        writer.writerow(["legacy_id", "utterance", "translation", "role", "slang", "speaker"])
        writer.writerow(["0007", 'Écoute, "ça"\r\nfin ', " Listen. ", "train", "false", ""])
    directory = tmp_path / "converted"
    convert_dataset(raw, directory, spec(source_language="fr-CA"), {**BASE_MAP, "slang": "slang", "speaker_id": "speaker"})
    row = load_bundle(directory, original_source=raw).records[0]
    assert row["original_id"] == "0007"
    assert row["source"] == 'Écoute, "ça"\r\nfin '
    assert row["target"] == " Listen. "
    assert row["source_language"] == "fr-CA"
    assert row["slang"] is False
    assert row["speaker_id"] is row["dialect"] is row["verification_status"] is None


def test_rejects_missing_split_or_surrogate_and_does_not_overwrite(tmp_path):
    raw = write_jsonl(tmp_path / "raw.jsonl", [source_row("one", "texto", "text")])
    destination = tmp_path / "v1"
    with pytest.raises(DatasetError, match="split explicitly"):
        convert_dataset(raw, destination, spec(source_language="es"), {k: v for k, v in BASE_MAP.items() if k != "split"})
    assert not destination.exists()
    convert_dataset(raw, destination, spec(source_language="es"), BASE_MAP)
    previous = (destination / "manifest.json").read_bytes()
    with pytest.raises(DatasetError, match="already exists"):
        convert_dataset(raw, destination, spec(source_language="es"), BASE_MAP)
    assert (destination / "manifest.json").read_bytes() == previous
    raw.write_text('{"legacy_id":"one","utterance":"\\ud800","translation":"text","role":"train"}\n', encoding="utf-8")
    with pytest.raises(DatasetError, match="source must be nonempty Unicode text"):
        convert_dataset(raw, tmp_path / "bad", spec(source_language="es"), BASE_MAP)
    assert not (tmp_path / "bad").exists()


def test_original_and_converted_checksums_are_independently_verified(tmp_path):
    raw = write_jsonl(tmp_path / "legacy.jsonl", [source_row("old", "¿Qué hora?", "What time?")])
    path = tmp_path / "converted"
    convert_dataset(raw, path, spec(source_language="es"), BASE_MAP)
    with pytest.raises(DatasetError, match="manifest checksum"):
        load_bundle(path, expected_manifest_sha256="0" * 64)
    raw.write_bytes(raw.read_bytes() + b"\n")
    with pytest.raises(DatasetError, match="original source checksum"):
        load_bundle(path, original_source=raw)
    converted = path / "records.jsonl"
    converted.write_bytes(converted.read_bytes() + b" ")
    with pytest.raises(DatasetError, match="record checksum"):
        load_bundle(path)


def test_training_rows_enforces_rights_purpose_and_split_roles(tmp_path):
    private = fixture_bundle(tmp_path, "private", [source_row("tr", "练习", "Exercise")])
    sealed_flag = fixture_bundle(
        tmp_path, "sealed-flag",
        [source_row("hold", "留着", None, "holdout"), source_row("dev", "开发", "Develop", "dev")],
        spec("sealed-flag", rights="unknown", rights_evidence=None),
    )
    denied = fixture_bundle(
        tmp_path, "denied", [source_row("tr", "不要练", "Do not practice")],
        spec("denied", rights="prohibited", rights_evidence=None),
    )
    commercial = fixture_bundle(
        tmp_path, "commercial", [source_row("other", "哪里", "Where")],
        spec("commercial", rights="commercial", privacy="public"),
    )
    assert [r["original_id"] for r in train(private, sealed_flag)] == ["tr"]
    assert [r["original_id"] for r in train(commercial, purpose="commercial")] == ["other"]
    with pytest.raises(DatasetError, match="no eligible train"):
        train(sealed_flag)
    with pytest.raises(DatasetError, match="unknown or prohibited"):
        train(private, denied)
    with pytest.raises(DatasetError, match="commercial training"):
        train(private, purpose="commercial")
    with pytest.raises(DatasetError, match="commercial training conflicts"):
        spec(rights="commercial", license="CC BY-NC 4.0")
    with pytest.raises(DatasetError, match="explicit training permission"):
        spec(license="unknown")
    with pytest.raises(DatasetError, match="explicit training permission"):
        spec(rights_evidence=None)


def test_explicit_holdout_must_not_be_reassigned_and_training_rows_do_not_mutate_bundle(tmp_path):
    raw = write_jsonl(tmp_path / "bad.jsonl", [source_row("x", "题目", "Question", sealed=True)])
    with pytest.raises(DatasetError, match="holdout flag"):
        convert_dataset(raw, tmp_path / "bad", spec(), {**BASE_MAP, "holdout": "sealed"})
    candidate, pin = fixture_bundle(
        tmp_path, "candidate", [
            source_row("train", "好的", "Okay", prior=["就这样"], group="g1", label=["ack"]),
            source_row("holdout", "另一个", "Another one", "holdout", prior=[], group=None, label=[]),
        ],
        mapping={**BASE_MAP, "context": "prior", "group_id": "group", "tags": "label"},
    )
    view = train((candidate, pin))
    view[0]["context"].append("caller edits cannot mutate the verified source")
    view[0]["tags"].clear()
    assert candidate.records[0]["context"] == ["就这样"]
    assert candidate.records[0]["tags"] == ["ack"]
    with pytest.raises(DatasetError, match="one trusted manifest"):
        training_rows([candidate], purpose="private_research", expected_manifest_sha256s=[])
    candidate.records[1]["split"] = "train"
    candidate.records[1]["holdout"] = False
    with pytest.raises(DatasetError, match="training records checksum"):
        train((candidate, pin))
    # Even canonical-looking replacement manifests are not trusted without the original pin.
    forged = replace(candidate, manifest={**candidate.manifest, "records_sha256": "0" * 64})
    with pytest.raises(DatasetError, match="trusted training manifest"):
        train((forged, pin))


@pytest.mark.parametrize(("role", "what"), (("test", "source"), ("dev", "context")))
def test_cross_dataset_split_contamination_uses_normalized_current_and_history(tmp_path, role, what):
    a = fixture_bundle(tmp_path, "alpha", [source_row("a", "ＡＢＣ， 你好！", "First fixture")])
    extras = {"context": "prior", "group_id": "session"}
    if what == "source":
        heldout = source_row("different-id", "abc你好", "Not the same translation", role)
        mapping = BASE_MAP
    else:
        heldout = source_row("different-id", "完全不同", None, role, prior="ABC你好", session="eval-session")
        mapping = {**BASE_MAP, **extras}
    b = fixture_bundle(tmp_path, "beta", [heldout], spec("beta", source_language="zh-Hant"), mapping)
    with pytest.raises(DatasetError, match="source or context"):
        train(a, b)


@pytest.mark.parametrize("kind", ("same_direction_reference", "reversed_direction_source"))
def test_cross_role_target_to_eval_leakage_rejected(tmp_path, kind):
    a = fixture_bundle(
        tmp_path, "alpha", [source_row("a", "练习句", "Keep THAT key!")],
        spec("alpha"),
    )
    if kind == "same_direction_reference":
        row = source_row("b", "并不相同", "keep that key", "test")
        config = spec("beta")
    else:
        row = source_row("b", "keep that key", "另一句", "test")
        config = spec("beta", source_language="en", target_language="zh")
    b = fixture_bundle(tmp_path, "beta", [row], config)
    with pytest.raises(DatasetError, match="training text overlaps"):
        train(a, b)


@pytest.mark.parametrize(("meta", "message"), (("group_id", "conversation/group"), ("speaker_id", "speaker")))
def test_conversation_and_speaker_ids_cannot_move_across_splits_or_versions(tmp_path, meta, message):
    mapping = {**BASE_MAP, meta: "catalog_key"}
    a, _ = fixture_bundle(tmp_path, "old", [source_row("x", "甲", "A", catalog_key=80)], spec("family", "v1"), mapping)
    b, _ = fixture_bundle(tmp_path, "new", [source_row("y", "乙", None, "test", catalog_key=80)], spec("family", "v2"), mapping)
    with pytest.raises(DatasetError, match=message):
        audit_split_integrity((*a.records, *b.records))


def test_original_identity_survives_releases_and_detects_nonconsecutive_same_release_duplicate(tmp_path):
    a, _ = fixture_bundle(tmp_path, "v1a", [source_row("vendor-id", "第一句", "First utterance")], spec("family", "v1"))
    b, _ = fixture_bundle(tmp_path, "v2", [source_row("vendor-id", "第二句", "Second utterance")], spec("family", "v2"))
    c, _ = fixture_bundle(tmp_path, "v1c", [source_row("vendor-id", "第三句", "Third utterance")], spec("family", "v1"))
    d, _ = fixture_bundle(tmp_path, "v3", [source_row("vendor-id", "第四句", None, "test")], spec("family", "v3"))
    with pytest.raises(DatasetError, match="duplicate original ID"):
        audit_split_integrity((*a.records, *b.records, *c.records))
    with pytest.raises(DatasetError, match="original ID appears"):
        audit_split_integrity((*a.records, *d.records))


def test_good_same_split_context_variants_are_not_false_leakage(tmp_path):
    bundle = fixture_bundle(
        tmp_path, "variants", [
            source_row("a", "把它关掉", "Turn the fan off.", prior=["风扇很响"], group="pair"),
            source_row("b", "把它关掉", "Turn it off.", prior=[], group="pair"),
        ],
        mapping={**BASE_MAP, "context": "prior", "group_id": "group"},
    )
    assert len(train(bundle)) == 2
