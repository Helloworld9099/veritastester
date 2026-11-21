"""Synthetic-only immutable training checkpoint tests; no model/backend execution."""
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from threading import Barrier
from unittest import TestCase, mock

from research.unified_training.artifacts import ALL_COMPONENTS, ArtifactError, ArtifactStore, preflight_config


def _compatibility():
    return {
        "direction": "es->en",
        "architecture": "marian",
        "backend": {"id": "synthetic", "version": "1.0.0"},
        "base": {"model_id": "fixture/marian", "revision": "b" * 40,
                 "files": {"source.spm": "c" * 64, "target.spm": "c" * 64, "vocab.json": "c" * 64,
                           "config.json": "c" * 64, "model.safetensors": "a" * 64}},
        "strategy": {"kind": "lora", "rank": 8, "alpha": 16,
                     "target_modules": ["decoder.q_proj"], "protocol_sha256": "d" * 64},
    }


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def _parts(suffix=b"A"):
    return {k: k.encode() + b":" + suffix for k in ALL_COMPONENTS}


class _Config:
    """Minimal structural stand-in; no product or backend imports."""
    def __init__(self, *, source=None, rights="commercial_candidate", resume=None, compat=None, output_id=None):
        self.source = {"mode": "base"} if source is None else source
        self.rights = rights
        self.resume_fingerprint = resume or "e" * 64
        self._compat = _compatibility() if compat is None else compat
        self.compatibility_sha256 = _sha(_canonical(self._compat))
        self.output_id = output_id

    def artifact_compatibility(self):
        return self._compat


class ArtifactStoreTests(TestCase):
    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.path = Path(td.name) / "store"
        self.store = ArtifactStore(self.path)

    def _publish(self, output="artifact-a", *, config=None, parts=None, step=4):
        cfg = copy.copy(config) if config is not None else _Config()
        cfg.output_id = output if cfg.output_id is None else cfg.output_id
        return self.store.publish(output, config=cfg, step=step,
                                  components=_parts() if parts is None else parts)

    def test_verifies_canonical_manifest_and_rejects_overwrite_without_touching_old_bytes(self):
        self.store.require_new("artifact-a")
        a = self._publish()
        self.assertEqual(dict(a.component_bytes), _parts())
        raw = (self.path / "artifact-a/manifest.json").read_bytes()
        self.assertEqual(_sha(raw), a.manifest_sha256)
        j = json.loads(raw)
        self.assertEqual(_canonical(j), raw)
        self.assertEqual(j["scope"], "research-only")
        self.assertEqual(j["resumability"], "full_state_snapshot")
        self.assertNotIn(str(self.path), raw.decode())
        self.assertNotIn("production", raw.decode())
        self.assertEqual(a.weights_sha256, _sha(_parts()["adapter"]))
        with self.assertRaisesRegex(ArtifactError, "already exists"):
            self.store.require_new("artifact-a")
        with self.assertRaisesRegex(ArtifactError, "already exists"):
            self._publish(parts=_parts(b"B"))
        self.assertEqual((self.path / "artifact-a/manifest.json").read_bytes(), raw)
        self.assertEqual(dict(self.store.verify("artifact-a", expected_manifest_sha256=a.manifest_sha256).component_bytes),
                         _parts())

    def test_warm_uses_only_immutable_adapter_snapshot_and_records_parent(self):
        a = self._publish()
        warm = _Config(source=a.source_pin("warm_start"), resume="1" * 64, output_id="warm-child")
        locked = self.store.verify_source(warm)
        self.assertEqual(dict(locked.component_bytes), {"adapter": _parts()["adapter"]})
        self.assertEqual(set(locked.component_paths), {"adapter"})
        with self.assertRaises(TypeError):
            locked.component_bytes["rng"] = b"x"
        with self.assertRaises(TypeError):
            locked.manifest["components"]["adapter"]["sha256"] = "f" * 64
        b = self.store.publish("warm-child", config=warm, step=1, parent=locked, components={"adapter": b"new-weights"})
        self.assertEqual(b.step, 1)
        self.assertEqual(b.manifest["lineage"]["mode"], "warm_start")
        self.assertEqual(b.manifest["lineage"]["parent"]["manifest_sha256"], a.manifest_sha256)
        self.assertEqual(b.manifest["lineage"]["parent"]["ancestry_sha256"], a.ancestry_sha256)
        self.assertEqual(b.manifest["resumability"], "weights_only")
        with self.assertRaisesRegex(ArtifactError, "cannot pin exact"):
            b.source_pin("exact_resume")

    def test_exact_resume_uses_all_pinned_state_and_strictly_later_global_step(self):
        a = self._publish()
        rconfig = _Config(source=a.source_pin("exact_resume"))
        verified = self.store.verify_source(rconfig)
        self.assertEqual(set(verified.component_bytes), ALL_COMPONENTS)
        self.assertEqual(verified.component_bytes["sampler"], _parts()["sampler"])
        self.assertEqual(verified.component_bytes["rng"], _parts()["rng"])
        with self.assertRaisesRegex(ArtifactError, "later step"):
            self._publish("not-later", config=rconfig, step=4)
        self.assertFalse((self.path / "not-later").exists())
        b = self._publish("resume-b", config=rconfig, step=5, parts=_parts(b"B"))
        self.assertEqual(b.manifest["lineage"]["parent"]["artifact_id"], a.artifact_id)
        self.assertEqual(b.manifest["lineage"]["parent"]["step"], 4)
        with self.assertRaisesRegex(ArtifactError, "fingerprint"):
            self.store.verify_source(_Config(source=a.source_pin("exact_resume"), resume="3" * 64))

    def test_wrong_manifest_weights_ancestry_state_pins_and_pin_compatibility_block(self):
        a = self._publish()
        with self.assertRaisesRegex(ArtifactError, "manifest SHA-256"):
            self.store.verify(a.artifact_id, expected_manifest_sha256="0" * 64)
        for key in ("weights_sha256", "compatibility_sha256", "ancestry_sha256"):
            pin = a.source_pin()
            pin[key] = "0" * 64
            with self.assertRaisesRegex(ArtifactError, "source pins"):
                self.store.verify_source(_Config(source=pin))
        pin = a.source_pin("exact_resume")
        pin["component_sha256"]["sampler"] = "0" * 64
        with self.assertRaisesRegex(ArtifactError, "source pins"):
            self.store.verify_source(_Config(source=pin))
        for change in ("backend", "base", "strategy"):
            changed = _compatibility()
            if change == "backend":
                changed[change]["version"] = "2.0.0"
            elif change == "base":
                changed[change]["files"]["vocab.json"] = "0" * 64
            else:
                changed[change]["rank"] = 16
            with self.assertRaisesRegex(ArtifactError, "incompatible"):
                self.store.verify_source(_Config(source=a.source_pin(), compat=changed))

    def test_partial_or_tampered_resume_state_is_not_silently_replaced(self):
        incomplete = _parts()
        incomplete.pop("rng")
        with self.assertRaisesRegex(ArtifactError, "all five"):
            self._publish(parts=incomplete)
        a = self._publish()
        rconf = _Config(source=a.source_pin("exact_resume"))
        a.component_paths["rng"].write_bytes(b"rng:B")
        with self.assertRaisesRegex(ArtifactError, "component SHA-256.*rng"):
            self.store.verify_source(rconf)
        a.component_paths["rng"].unlink()
        with self.assertRaisesRegex(ArtifactError, "artifact file"):
            self.store.verify_source(rconf)
        b = self._publish("weights-only-b", parts={"adapter": b"only"})
        with self.assertRaisesRegex(ArtifactError, "weights-only"):
            self.store.verify(b.artifact_id, expected_manifest_sha256=b.manifest_sha256,
                              purpose="exact_resume", config=_Config())

    def test_private_parent_cannot_be_republished_as_commercial_candidate(self):
        prv = self._publish(config=_Config(rights="private_noncommercial"))
        dconf = _Config(source=prv.source_pin(), rights="commercial_candidate")
        with self.assertRaisesRegex(ArtifactError, "rights cannot be broadened"):
            self._publish("incorrect-rights", config=dconf)
        self.assertFalse((self.path / "incorrect-rights").exists())
        source = self._publish("commercial-source")
        child = self._publish("private-child", config=_Config(source=source.source_pin(), rights="private_noncommercial"))
        self.assertEqual(child.rights, "private_noncommercial")
        self.assertEqual(child.manifest["lineage"]["parent"]["rights"], "commercial_candidate")

    def test_exclusive_id_reservation_allows_only_one_concurrent_writer(self):
        start = Barrier(2)

        def attempt(marker):
            start.wait()
            try:
                return self._publish(parts=_parts(marker))
            except ArtifactError as exc:
                return str(exc)

        with ThreadPoolExecutor(max_workers=2) as pool:
            r = list(pool.map(attempt, (b"A", b"B")))
        ok = [x for x in r if not isinstance(x, str)]
        self.assertEqual(len(ok), 1)
        self.assertEqual(len([x for x in r if isinstance(x, str) and "already exists" in x]), 1)
        a = ok[0]
        self.assertEqual(self.store.verify(a.artifact_id, expected_manifest_sha256=a.manifest_sha256).weights_sha256,
                         a.weights_sha256)

    def test_interrupted_publish_is_not_valid_and_reserves_id(self):
        with mock.patch("research.unified_training.artifacts.os.link", side_effect=OSError("mock write failure")):
            with self.assertRaisesRegex(ArtifactError, "not published"):
                self._publish()
        self.assertFalse((self.path / "artifact-a/manifest.json").exists())
        with self.assertRaisesRegex(ArtifactError, "already exists"):
            self._publish()
        with self.assertRaisesRegex(ArtifactError, "artifact file"):
            self.store.verify("artifact-a", expected_manifest_sha256="0" * 64)

    def test_only_configured_output_id_can_be_published(self):
        with self.assertRaisesRegex(ArtifactError, "configured output ID"):
            self._publish("not-approved-id", config=_Config(output_id="approved-id"))
        self.assertFalse(self.path.exists())

    def test_config_compatible_qlora_semver_and_long_target_preflight_then_publish(self):
        compat = _compatibility()
        compat["backend"]["version"] = "1.0.0+build"
        compat["strategy"].update(kind="qlora", load_bits=4, target_modules=["model." + "a" * 145])
        config = _Config(compat=compat, output_id="synthetic-qlora")
        self.assertEqual(preflight_config(config), compat)
        check = self._publish(config.output_id, config=config)
        self.assertEqual(check.manifest["compatibility"]["strategy"]["load_bits"], 4)

    def test_verifies_all_pinned_ancestor_manifests_and_rejects_deceptive_link(self):
        a = self._publish()
        b = self._publish("child", config=_Config(source=a.source_pin()), parts={"adapter": b"B"})
        c = self._publish("grandchild", config=_Config(source=b.source_pin()), parts={"adapter": b"C"})
        self.assertEqual(self.store.verify(c.artifact_id, expected_manifest_sha256=c.manifest_sha256).artifact_id,
                         c.artifact_id)
        path = self.path / "child/manifest.json"
        original = path.read_bytes()
        forged = json.loads(original)
        forged["lineage"]["parent"]["step"] += 1
        forged["ancestry_sha256"] = _sha(_canonical(forged["lineage"]))
        raw = _canonical(forged)
        path.write_bytes(raw)
        with self.assertRaisesRegex(ArtifactError, "ancestor manifest disagrees"):
            self.store.verify(b.artifact_id, expected_manifest_sha256=_sha(raw))
        path.write_bytes(original)
        (self.path / "artifact-a/manifest.json").unlink()
        with self.assertRaisesRegex(ArtifactError, "missing or invalid pinned ancestor"):
            self.store.verify(c.artifact_id, expected_manifest_sha256=c.manifest_sha256)

    def test_refuses_path_aliases_and_external_component_symlink(self):
        for bad in ("../escape", "/outside", "drive:weights", ".hidden", "NUL.txt"):
            with self.subTest(bad=bad), self.assertRaisesRegex(ArtifactError, "invalid.*ID"):
                self.store.require_new(bad)
        a = self._publish()
        target = self.path / "loose"
        target.write_bytes(_parts()["adapter"])
        a.component_paths["adapter"].unlink()
        try:
            a.component_paths["adapter"].symlink_to(target)
        except OSError:
            self.skipTest("symlink creation unavailable")
        with self.assertRaisesRegex(ArtifactError, "symlink"):
            self.store.verify(a.artifact_id, expected_manifest_sha256=a.manifest_sha256)
