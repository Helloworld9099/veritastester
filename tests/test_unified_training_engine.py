import hashlib
import json

import pytest

from research.unified_training.artifacts import ArtifactError, ArtifactStore, ALL_COMPONENTS
from research.unified_training.config import TrainingConfig
from research.unified_training.engine import BackendResult, EngineError, LocalBackend, run_local


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _json(data):
    return (json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n").encode()


@pytest.fixture
def sandbox(tmp_path):
    base = tmp_path / "base"
    data = tmp_path / "data"
    base.mkdir()
    data.mkdir()
    files = {}
    for name in ("config.json", "source.spm", "target.spm", "vocab.json", "model.safetensors"):
        payload = name.encode()
        (base / name).write_bytes(payload)
        files[name] = _sha(payload)

    def prepare(split, *, output_split=None, sealed=False, source=None):
        records = _json({"split": output_split or split, "source_language": "es",
                         "target_language": "en", "source": source or f"source-{split}",
                         "target": f"target-{split}"})
        (data / f"{split}.jsonl").write_bytes(records)
        manifest = {"schema_version": 1, "kind": "reviewed_research_training_split",
                    "dataset_id": "synthetic", "version": "v1", "split": split,
                    "sealed": sealed, "license": "CC0-1.0", "visibility": "public",
                    "records_path": f"{split}.jsonl", "records_sha256": _sha(records), "row_count": 1}
        raw = _json(manifest)
        path = data / f"{split}.json"
        path.write_bytes(raw)
        return {**{k: manifest[k] for k in ("dataset_id", "version", "split", "license", "visibility")},
                "manifest_sha256": _sha(raw), "local_path": path.name}

    raw = {"schema_version": 1, "direction": "es->en", "architecture": "marian",
           "backend": {"id": "synthetic", "version": "1.0.0"},
           "base": {"model_id": "example/synthetic", "revision": "a" * 40, "files": files},
           "device": {"kind": "cpu", "gpu_opt_in": False, "precision": "fp32", "threads": 1},
           "strategy": {"kind": "lora", "rank": 8, "alpha": 16,
                        "target_modules": ["model.tiny.q_proj"], "protocol_sha256": "b" * 64},
           "optimizer": {"name": "adamw", "learning_rate": 0.001, "weight_decay": 0.01,
                         "beta1": 0.9, "beta2": 0.999, "epsilon": 0.00000001},
           "schedule": {"kind": "constant", "total_updates": 3, "warmup_updates": 0,
                        "micro_batch_size": 1, "gradient_accumulation_steps": 1,
                        "max_source_tokens": 96, "max_target_tokens": 96},
           "seed": 73, "data": [prepare("train"), prepare("dev")],
           "rights": "commercial_candidate", "source": {"mode": "base"}, "output_id": "new-run"}
    return raw, base, data, prepare, ArtifactStore(tmp_path / "checkpoints")


def _driver(callback, *, exact=True):
    return LocalBackend("synthetic", "1.0.0", frozenset({"marian"}), frozenset({"lora"}),
                        frozenset({"cpu"}), exact, callback)


def _run(sandbox, callback, *, authorized=True, exact=True):
    raw, base, data, _, store = sandbox
    return run_local(TrainingConfig.from_mapping(raw), backend=_driver(callback, exact=exact),
                     store=store, base_root=base, data_root=data, authorized=authorized)


def _done(ctx):
    return BackendResult(ctx.target_step, {"adapter": b"candidate-adapter"})


def test_execution_is_opt_in_and_duplicate_id_never_reinvokes_backend(sandbox):
    called = []
    def backend(ctx):
        called.append(ctx)
        return _done(ctx)
    with pytest.raises(EngineError, match="authorization"):
        _run(sandbox, backend, authorized=False)
    artifact = _run(sandbox, backend)
    assert called[0].start_step == 0
    assert [item.split for item in called[0].train_data] == ["train"]
    assert [item.split for item in called[0].development_data] == ["dev"]
    assert artifact.manifest["lineage"]["parent"] is None
    with pytest.raises(ArtifactError):
        _run(sandbox, backend)
    assert len(called) == 1


def test_base_tokenizer_digest_and_all_files_verified_before_backend(sandbox):
    _, base, _, _, _ = sandbox
    backend = lambda ctx: pytest.fail("backend invoked before base was verified")
    (base / "source.spm").write_bytes(b"wrong")
    with pytest.raises(EngineError, match="checksum"):
        _run(sandbox, backend)
    (base / "extra").write_bytes(b"shadow")
    with pytest.raises(EngineError, match="contents"):
        _run(sandbox, backend)


def test_holdout_manifest_and_exact_overlap_are_blocked_before_backend(sandbox):
    raw, _, data, prepare, _ = sandbox
    backend = lambda ctx: pytest.fail("backend invoked on a rejected split")
    raw["data"][1] = prepare("dev", sealed=True)
    # A sealed manifest must reject before accessing even its referenced path.
    (data / "dev.jsonl").unlink()
    with pytest.raises(EngineError, match="unsealed"):
        _run(sandbox, backend)
    raw["data"][1] = prepare("dev", output_split="holdout")
    with pytest.raises(EngineError, match="incompatible"):
        _run(sandbox, backend)
    raw["data"][1] = prepare("dev", source="source-train")
    with pytest.raises(EngineError, match="overlap"):
        _run(sandbox, backend)


@pytest.mark.parametrize("mode", ["warm_start", "exact_resume"])
def test_verified_parent_state_is_passed_only_as_promised(sandbox, mode):
    raw, _, _, _, store = sandbox
    initial = TrainingConfig.from_mapping({**raw, "output_id": "parent"})
    original = {key: ("previous-" + key).encode() for key in ALL_COMPONENTS}
    parent = store.publish("parent", config=initial, components=original, step=1)
    raw["source"] = parent.source_pin(mode)
    def backend(ctx):
        assert ctx.start_step == (1 if mode == "exact_resume" else 0)
        assert dict(ctx.source_components) == (
            original if mode == "exact_resume" else {"adapter": original["adapter"]})
        assert "source-train" not in repr(ctx)
        return BackendResult(ctx.target_step, {key: b"new-" + value for key, value in original.items()},
                             {key: _sha(value) for key, value in original.items()}
                             if mode == "exact_resume" else None)
    if mode == "exact_resume":
        with pytest.raises(EngineError, match="does not support exact"):
            _run(sandbox, backend, exact=False)
    if mode == "exact_resume":
        with pytest.raises(EngineError, match="restore receipt"):
            _run(sandbox, lambda ctx: BackendResult(ctx.target_step, original))
    child = _run(sandbox, backend)
    assert child.manifest["lineage"]["mode"] == mode
    assert child.manifest["lineage"]["parent"]["artifact_id"] == "parent"
    assert store.verify("parent", expected_manifest_sha256=parent.manifest_sha256).component_bytes == original


def test_failed_or_incomplete_backend_never_creates_a_result(sandbox):
    raw, _, _, _, store = sandbox
    def error(_):
        raise RuntimeError("do not expose arbitrary callback text")
    with pytest.raises(EngineError, match="did not complete") as raised:
        _run(sandbox, error)
    assert "arbitrary" not in str(raised.value)
    with pytest.raises(EngineError, match="update target"):
        _run(sandbox, lambda _: BackendResult(2, {"adapter": b"incomplete"}))
    assert not (store.root / raw["output_id"]).exists()


def test_controlled_pause_can_resume_from_its_verified_intermediate_step(sandbox):
    raw, _, _, _, store = sandbox
    payloads = {key: key.encode() for key in ALL_COMPONENTS}
    with pytest.raises(EngineError, match="all five"):
        _run(sandbox, lambda ctx: BackendResult(1, {"adapter": b"partial"}, interrupted=True))
    paused = _run(sandbox, lambda ctx: BackendResult(1, payloads, interrupted=True))
    assert paused.step == 1
    assert paused.manifest["resumability"] == "full_state_snapshot"
    raw.update(output_id="resumed", source=paused.source_pin("exact_resume"))
    def resume(ctx):
        assert (ctx.start_step, ctx.target_step) == (1, 3)
        return BackendResult(ctx.target_step, payloads,
                             {name: _sha(data) for name, data in ctx.source_components.items()})
    assert _run(sandbox, resume).step == 3
    assert store.verify("new-run", expected_manifest_sha256=paused.manifest_sha256).step == 1


def test_artifact_storage_cannot_be_inside_input_snapshot(sandbox):
    raw, base, data, _, _ = sandbox
    with pytest.raises(EngineError, match="separate"):
        run_local(TrainingConfig.from_mapping(raw), backend=_driver(_done),
                  store=ArtifactStore(base / "forbidden-output"), base_root=base, data_root=data,
                  authorized=True)


def test_data_metadata_and_paths_are_not_logged_and_unknown_fields_fail(sandbox):
    raw, _, data, _, _ = sandbox
    train = data / "train.jsonl"
    row = json.loads(train.read_bytes())
    row["final_holdout"] = True
    records = _json(row)
    train.write_bytes(records)
    manifest = data / "train.json"
    meta = json.loads(manifest.read_bytes())
    meta["records_sha256"] = _sha(records)
    manifest.write_bytes(_json(meta))
    raw["data"][0]["manifest_sha256"] = _sha(_json(meta))
    assert "local_path" not in repr(TrainingConfig.from_mapping(raw))
    assert "weights" not in repr(BackendResult(1, {"adapter": b"weights"}))
    with pytest.raises(EngineError, match="schema"):
        _run(sandbox, lambda ctx: pytest.fail("backend used invalid fields"))
