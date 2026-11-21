"""Offline schema and verification tests using small synthetic artifacts."""
import copy
import hashlib
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from research.unified_models.historical.registry import (
    MANIFEST, ModelRegistry, compatibility_digest, safe_local_path,
)
from research.unified_models.historical.runtime import HistoricalMarianBaseline

RAW = json.loads(MANIFEST.read_text())
ROUTES = {"en-es", "es-en", "fr-en"}


def route(raw, key):
    return next(x for x in raw["models"] if x["model_id"] == key)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fake_safetensors(protocol, *, dtype="F32", reverse=False, nan=False, gap=False):
    data, indexes = bytearray(), {}
    r, d = protocol["rank"], protocol["projection_size"]
    for layer in protocol["target_modules"]:
        for slot, shape in (("A", [r,d]), ("B", [d,r])):
            start = len(data)
            data.extend(struct.pack("<f", float("nan") if nan else .5) * r * d)
            indexes[layer + ".lora_" + slot] = {
                "dtype": dtype, "shape": shape[::-1] if reverse else shape,
                "data_offsets": [start + int(gap), len(data) + int(gap)]}
    header = json.dumps(indexes).encode()
    return struct.pack("<Q", len(header)) + header + data


class TestM1Registry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = copy.deepcopy(RAW)
        self.manifest = self.root / "manifest.json"

    def load(self):
        self.manifest.write_text(json.dumps(self.raw))
        return ModelRegistry(self.manifest, root=self.root)

    def test_actual_routes_metadata_and_historical_pins(self):
        reg = self.load()
        self.assertEqual(set(reg.entries), ROUTES)
        es = reg.select(" es-MX ", "en-US")
        fr = reg.select("fr-CA")
        en = reg.select("EN", "es-419")
        self.assertEqual(es.active_adapter["sha256"], "3bf849bb9a4377dcb9f937cdc7750a7f26616fc1c58cf703cef0ba8e46430a71")
        self.assertEqual(fr.active_adapter["sha256"], "fba2bd78958bf0bf8a4c14dc55e601e08663633718c4c9c17f91ef9b52dcfa1c")
        self.assertNotEqual(es.files["source.spm"]["sha256"], fr.files["source.spm"]["sha256"])
        self.assertIsNone(en.active_adapter)
        self.assertIsNone(en.rollback_adapter)
        self.assertTrue(all(not v.supports_context for v in reg.entries.values()))
        self.assertEqual(dict(reg.inactive_private),
                         {"direction":"cmn-en", "status":"private_noncommercial_inactive", "routable":False})
        with self.assertRaises(TypeError):
            es.active_adapter["sha256"] = "x"
        with self.assertRaises(ValueError):
            reg.verify_adapter(en, self.root, use_previous=True)
        for src,tgt in (("auto","en"),("en","fr"),("fr","es"),("es","fr"),("cmn-Hans","en"),("de","en")):
            with self.subTest(src=src,tgt=tgt), self.assertRaises(ValueError):
                reg.select(src,tgt)

    def test_dummy_language_and_direction_are_additive_configuration(self):
        self.raw["languages"]["qaa"] = {**self.raw["languages"]["fr"], "name":"Dummy",
                                      "dialects": ["qaa-DE"]}
        demo = copy.deepcopy(route(self.raw,"en-es"))
        # Manifest-only routing fixture. Never a real/available/demonstrated model.
        demo.update(model_id="qaa-en",source="qaa",target="en")
        demo["base"]["model"] = "TestOnly/not-a-real-model"
        demo["base"]["revision"] = "1" * 40
        demo["compatibility_sha256"] = compatibility_digest(demo["base"], None)
        self.raw["models"].append(demo)
        reg = self.load()
        self.assertEqual(reg.select("qaa-DE","en").model_id,"qaa-en")
        self.assertEqual(set(reg.entries), ROUTES | {"qaa-en"})
        with self.assertRaises(FileNotFoundError):
            reg.verify_base(reg.select("qaa-DE"), self.root)

    def test_reject_wrong_route_rights_role_and_pins(self):
        cases = [
            lambda: route(self.raw,"es-en").update(active_adapter="fr-en-current"),
            lambda: route(self.raw,"es-en").update(active_adapter="es-en-rollback"),
            lambda: route(self.raw,"es-en").update(noncommercial_data_in_training=True),
            lambda: route(self.raw,"fr-en").update(supports_context=True),
            lambda: route(self.raw,"en-es").update(human_meaning_verified=True),
            lambda: route(self.raw,"es-en")["base"]["files"]["source.spm"].update(sha256="0"*64),
            lambda: self.raw["checkpoints"]["es-en-current"].update(sha256=self.raw["archived_noncommercial"][0]["sha256"]),
            lambda: self.raw["checkpoints"]["es-en-current"].update(commit="main"),
            lambda: self.raw["checkpoints"]["es-en-current"].update(path="../evil.safetensors"),
            lambda: self.raw["inactive_private"].update(artifact_id="should-not-escape"),
            lambda: self.raw["languages"]["en"].update(dialects=["fr-FR"]),
        ]
        for case in cases:
            self.raw = copy.deepcopy(RAW)
            case()
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.load()

    def fixture(self, *, dtype="F32", reverse=False, nan=False, gap=False):
        row = route(self.raw,"es-en")
        protocol = row["adapter_protocol"]
        protocol.update(projection_size=4,rank=2,alpha=4,
                        target_modules=["model.encoder.layers.0.self_attn.q_proj"])
        configs = {"model_type":"marian","is_encoder_decoder":True,
                   "d_model":4,"encoder_layers":1}
        dest = self.root / "bases" / row["model_id"] / row["base"]["revision"]
        dest.mkdir(parents=True)
        for name, meta in row["base"]["files"].items():
            content = json.dumps(configs).encode() if name == "config.json" else name.encode()
            meta.update(sha256=sha(content),size_bytes=len(content))
            (dest / name).write_bytes(content)
        compat = compatibility_digest(row["base"],protocol)
        row["compatibility_sha256"] = compat
        for slot in ("active_adapter","rollback_adapter"):
            b = fake_safetensors(protocol, dtype=dtype, reverse=reverse, nan=nan, gap=gap)
            if slot == "rollback_adapter":
                # Distinct content with same layout.
                b = b[:-4] + struct.pack("<f",.25)
            record = self.raw["checkpoints"][row[slot]]
            record.update(sha256=sha(b),compatibility_sha256=compat,bytes=len(b))
            path = self.root / "artifacts" / record["path"]
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(b)
        return self.load(), dest

    def test_checks_pinned_base_and_adapter_safetensors(self):
        reg,dest = self.fixture()
        spec = reg.select("es")
        self.assertEqual(reg.verify_base(spec,self.root/"bases"),dest)
        current = reg.verify_adapter(spec,self.root/"artifacts")
        rollback = reg.verify_adapter(spec,self.root/"artifacts",use_previous=True)
        self.assertNotEqual(current,rollback)
        with self.assertRaises(FileNotFoundError):
            reg.verify_adapter(spec,None)
        unpinned = dest / "unexpected.json"
        unpinned.write_text("{}")
        with self.assertRaisesRegex(ValueError,"unpinned"):
            reg.verify_base(spec,self.root/"bases")
        unpinned.unlink()
        (dest / "source.spm").write_bytes(b"untrusted!")
        with self.assertRaisesRegex(ValueError,"checksum"):
            reg.verify_base(spec,self.root/"bases")
        x = self.root / "artifacts" / spec.active_adapter["path"]
        x.write_bytes(current[:-1]+b"7")
        with self.assertRaisesRegex(ValueError,"checksum"):
            reg.verify_adapter(spec,self.root/"artifacts")

    def test_incompatible_adapter_never_loads_model(self):
        for opts in ({"nan":True},{"reverse":True},{"dtype":"F16"},{"gap":True}):
            with tempfile.TemporaryDirectory() as d, self.subTest(opts=opts):
                self.raw = copy.deepcopy(RAW)
                self.root = Path(d)
                self.manifest = self.root / "manifest.json"
                reg,_ = self.fixture(**opts)
                engine = HistoricalMarianBaseline(historical_replay=True, registry=reg,cache_dir=self.root/"bases",
                                                          artifact_root=self.root/"artifacts")
                with patch.object(reg,"verify_base",side_effect=AssertionError("should not reach base")):
                    with self.assertRaises(ValueError):
                        engine.translate(SimpleNamespace(source="es", target="en", text="Hola"))
                self.assertEqual(engine.loaded_model_count,0)

    def test_enabled_unsupported_context_modifiers_fail_explicitly(self):
        engine = HistoricalMarianBaseline(historical_replay=True, registry=self.load())
        for key, value, label in [("previous", ["irrelevant"], "context"),
                                  ("glossary", {"term":"meaning"}, "glossary"),
                                  ("sense_hints", {"term":"topic"}, "sense hints")]:
            request = SimpleNamespace(source="es",target="en",text="Hola",
                                      context_enabled=True, **{key:value})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,label):
                engine.translate(request)
        self.assertEqual(engine.loaded_model_count,0)

    @unittest.skipUnless(os.name == "posix", "symlink test requires POSIX")
    def test_symlink_and_traversal_in_source_are_rejected(self):
        fixture = self.root / "somewhere"
        fixture.mkdir()
        (self.root / "link").symlink_to(fixture,target_is_directory=True)
        for p in ("link/weights","/tmp/x","a//b","a/../b","a/./b","a\\b"):
            with self.subTest(path=p), self.assertRaises(ValueError):
                safe_local_path(self.root,p)


if __name__ == "__main__":
    unittest.main()
