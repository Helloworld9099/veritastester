"""Offline reviewed-data manifest tests use only artificial caption snippets."""

import copy
import hashlib
import json
import os

import pytest

from research import workflow_data as data


def write_fixture(root, name, rows):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def label(i, source, target, **metadata):
    return dict(id=i, source=source, target=target, **metadata)


def declaration(root, name, role, *, source="zh", target="en", **options):
    path = root / name
    item = {
        "dataset_id": "toy-dataset",
        "version": "v1",
        "role": role,
        "path": name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "2" * 64,
        "source_language": source,
        "target_language": target,
        "origin_uri": "manual artificial provenance",
        "license": "CC BY 4.0",
        "rights": "public_research",
        "rights_evidence": "reviewed permissions receipt",
        "privacy": "public",
        "review_status": "reviewed",
        "review_evidence": "fixture label review receipt",
    }
    item.update(options)
    return item


def config(*items, purpose="public_research"):
    return {"schema_version": 1, "purpose": purpose, "datasets": list(items)}


def create_pair(root):
    write_fixture(root, "corpora/tr.jsonl", [label("train-private-marker", "合成训练句", "Synthetic train caption")])
    write_fixture(root, "corpora/dv.jsonl", [
        label("dev-private-marker", "验证前门", "Check the front door"),
        label(27, "衣服挂在墙上", "The garment hangs on the wall", role="dev"),
    ])
    return config(declaration(root, "corpora/tr.jsonl", "train"), declaration(root, "corpora/dv.jsonl", "dev"))


def test_checksum_and_safe_aggregate_receipt(tmp_path):
    cfg = create_pair(tmp_path)
    first = data.prepare(cfg, root=tmp_path)
    assert first == data.verify(cfg, first, root=tmp_path)
    assert first["kind"] == "bridgecaption_workflow_data"
    assert first["rows"] == {"train": 1, "dev": 2}
    assert first["directions"] == {"zh->en": {"train": 1, "dev": 2}}
    assert first["files"] == 2
    assert all(len(first[key]) == 64 for key in ("config_sha256", "membership_sha256"))
    emitted = json.dumps(first, ensure_ascii=False)
    for text in ("train-private-marker", "dev-private-marker", "corpora/", "合成训练句", "Check the front door",
                 "fixture label review receipt", "toy-dataset"):
        assert text not in emitted
    stale = {**first, "rows": {"train": 2, "dev": 2}}
    with pytest.raises(data.DataError, match="receipt does not match"):
        data.verify(cfg, stale, root=tmp_path)
    (tmp_path / "corpora/tr.jsonl").write_text("modified", encoding="utf-8")
    with pytest.raises(data.DataError, match="SHA-256"):
        data.verify(cfg, first, root=tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "pattern"),
    [
        ("role", "holdout", "only train and development"),
        ("role", "test", "only train and development"),
        ("privacy", "private", "reviewed public"),
        ("review_status", "skipped", "reviewed public"),
        ("review_evidence", "unknown", "concrete origin and review"),
        ("rights", "unknown", "explicit allowed"),
        ("license", "CC BY-ND 4.0", "no-derivatives"),
        ("sha256", "sha256:loose", "explicit SHA-256"),
        ("path", "corpora/../other.jsonl", "unsafe"),
        ("path", "/tmp/not-accessible.jsonl", "relative JSONL"),
        ("path", "corpora/secureHoldout.jsonl", "sealed dataset"),
        ("path", "private/temp.jsonl", "not allowed"),
        ("path", "corpora/Sealed/dev.jsonl", "not allowed"),
        ("target_language", "EN", "explicit lowercase"),
        ("target_language", "zh", "distinct languages"),
    ],
)
def test_no_files_opened_if_any_manifest_declaration_is_disallowed(tmp_path, monkeypatch, field, value, pattern):
    cfg = create_pair(tmp_path)
    cfg["datasets"][1][field] = value
    # All requested datasets are validated before *any* file opens.
    monkeypatch.setattr(data, "_open_public_regular", lambda *args: pytest.fail("unexpected open"))
    with pytest.raises(data.DataError, match=pattern) as err:
        data.prepare(cfg, root=tmp_path)
    assert "corpora/" not in str(err.value)


def test_roles_are_explicit_per_active_language_direction(tmp_path):
    cfg = create_pair(tmp_path)
    cfg["datasets"][1]["source_language"] = "es"
    with pytest.raises(data.DataError, match="each direction needs both"):
        data.prepare(cfg, root=tmp_path)


def test_known_noncommercial_permission_cannot_be_labeled_commercial(tmp_path):
    cfg = create_pair(tmp_path)
    cfg["purpose"] = "commercial"
    with pytest.raises(data.DataError, match="commercial data permission"):
        data.prepare(cfg, root=tmp_path)
    cfg["datasets"][0]["rights"] = "commercial"
    cfg["datasets"][0]["license"] = "CC BY-NC 4.0"
    with pytest.raises(data.DataError, match="declared restriction"):
        data.prepare(cfg, root=tmp_path)


def test_symlink_and_special_file_rejected(tmp_path):
    cfg = create_pair(tmp_path)
    source = tmp_path / "corpora/tr.jsonl"
    linked = tmp_path / "corpora/link.jsonl"
    linked.symlink_to(source)
    cfg["datasets"][0]["path"] = "corpora/link.jsonl"
    with pytest.raises(data.DataError, match="safely open"):
        data.prepare(cfg, root=tmp_path)
    linked.unlink()
    os.mkfifo(linked)
    with pytest.raises(data.DataError, match="regular local"):
        data.prepare(cfg, root=tmp_path)
    cfg["datasets"][0]["path"] = "different-dir/tr.jsonl"
    (tmp_path / "different-dir").symlink_to(tmp_path / "corpora", target_is_directory=True)
    with pytest.raises(data.DataError, match="safely open"):
        data.prepare(cfg, root=tmp_path)


@pytest.mark.parametrize("kind", ("current", "prior", "reference"))
def test_normalized_train_dev_overlap_across_roles_and_language_variants(tmp_path, kind):
    write_fixture(tmp_path, "tr.jsonl", [label("a", "ＡＢＣ， 你好！", "Fixture cafe request.")])
    if kind == "current":
        bad = label("b", "abc你好", "Not the same translation")
    elif kind == "prior":
        bad = label("b", "我现在离开", "A distinct translation", context=["abc你好"], group_id="one")
    else:
        bad = label("b", "真的不同", "FIXTURE cafe request")
    write_fixture(tmp_path, "dv.jsonl", [bad])
    write_fixture(tmp_path, "dv-a.jsonl", [label("c", "另一例", "Different example")])
    write_fixture(tmp_path, "tr-b.jsonl", [label("d", "尚未接触", "Unseen and unrelated")])
    cfg = config(
        declaration(tmp_path, "tr.jsonl", "train", dataset_id="one"),
        declaration(tmp_path, "dv-a.jsonl", "dev", dataset_id="one"),
        declaration(tmp_path, "tr-b.jsonl", "train", dataset_id="two", source="zh-hant"),
        declaration(tmp_path, "dv.jsonl", "dev", dataset_id="two", source="zh-hant"),
    )
    with pytest.raises(data.DataError, match="normalized train/development text overlap"):
        data.prepare(cfg, root=tmp_path)


@pytest.mark.parametrize("meta", ("id", "group_id", "speaker_id"))
def test_publisher_metadata_cannot_cross_split_by_version(tmp_path, meta):
    tr, dv = label("train", "第一句", "A training sample."), label("dev", "隔开的", "A different check.")
    tr[meta] = dv[meta] = "same-opaque-identifier"
    write_fixture(tmp_path, "tr.jsonl", [tr])
    write_fixture(tmp_path, "dv.jsonl", [dv])
    cfg = config(declaration(tmp_path, "tr.jsonl", "train"), declaration(tmp_path, "dv.jsonl", "dev", version="v2"))
    with pytest.raises(data.DataError, match="overlap in original identity, group, or speaker"):
        data.prepare(cfg, root=tmp_path)


@pytest.mark.parametrize(
    ("metadata", "error"),
    [
        ({"sealed": True}, "sealed records"),
        ({"privacy": "private"}, "sealed records"),
        ({"role": "test"}, "role conflicts"),
        ({"split": "holdout"}, "role conflicts"),
        ({"context": ["保留之前的说法"]}, "stable group"),
        ({"unknown": True}, "row fields"),
    ],
)
def test_row_markers_and_context_are_validated(tmp_path, metadata, error):
    cfg = create_pair(tmp_path)
    write_fixture(tmp_path, "corpora/tr.jsonl", [label("train", "独立样本", "Standalone", **metadata)])
    cfg["datasets"][0] = declaration(tmp_path, "corpora/tr.jsonl", "train")
    with pytest.raises(data.DataError, match=error):
        data.prepare(cfg, root=tmp_path)


def test_same_split_context_variants_are_deliberately_allowed(tmp_path):
    cfg = create_pair(tmp_path)
    write_fixture(tmp_path, "corpora/tr.jsonl", [
        label("one", "现在可以关了", "You can turn it off now."),
        label("two", "现在可以关了", "You can turn off the fan now.", context=["风扇一直开着"], group_id="pair"),
    ])
    cfg["datasets"][0] = declaration(tmp_path, "corpora/tr.jsonl", "train")
    assert data.prepare(cfg, root=tmp_path)["rows"]["train"] == 2


def test_duplicate_json_keys_and_original_ids_are_not_silently_dropped(tmp_path):
    cfg = create_pair(tmp_path)
    raw = tmp_path / "corpora/tr.jsonl"
    raw.write_text('{"id":"x","id":"y","source":"row","target":"item"}\n', encoding="utf-8")
    cfg["datasets"][0] = declaration(tmp_path, "corpora/tr.jsonl", "train")
    with pytest.raises(data.DataError, match="invalid JSONL"):
        data.prepare(cfg, root=tmp_path)
    write_fixture(tmp_path, "corpora/tr.jsonl", [label("train", "一", "Nine"), label("train", "二", "Ten")])
    cfg["datasets"][0] = declaration(tmp_path, "corpora/tr.jsonl", "train")
    with pytest.raises(data.DataError, match="duplicate original ID"):
        data.prepare(cfg, root=tmp_path)
