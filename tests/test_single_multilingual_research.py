"""Synthetic contract tests only; no weights, inference, APIs, or holdouts."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from research.unified_models.registry import ModelRegistry
from research.unified_models.runtime import SingleMultilingualResearchTranslator


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def request(direction, text="Some public synthetic sample"):
    source, target = direction.split("->")
    return SimpleNamespace(source=source, target=target, text=text, context_enabled=False)


class SingleModelContract(unittest.TestCase):
    def setUp(self):
        root = TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.root = Path(root.name)
        self.model_root = self.root / "checkpoint"
        self.model_root.mkdir()
        self.manifest = self.root / "registry.json"
        self.evaluation = self.root / "evaluation.json"
        files = {"config.json": b"synthetic config", "tokenizer.json": b"synthetic tokenizer",
                 "model.safetensors": b"synthetic blob; no actual weight loading"}
        for name, data in files.items():
            (self.model_root / name).write_bytes(data)
        self.identity = {"model_id": "fixture-no-real-model", "revision": "1" * 40,
                         "architecture": "fixture-text-stub",
                         "files": {name: digest(data) for name, data in files.items()}}
        self.targets = ["en->es", "es->en", "fr->en"]
        self.result = {"schema_version": 1, "kind": "independent_bilingual_direction_receipt",
                       "checkpoint_sha256": digest(canonical(self.identity)),
                       "directions": {route: {"status": "passed", "case_count": 20,
                                             "evaluation_sha256": "2" * 64} for route in self.targets}}
        self.value = {"schema_version": 1, "scope": "research_only",
                      "architecture": "single_multilingual_model", "target_directions": self.targets,
                      "candidate": {**self.identity, "verified_directions": self.targets,
                                    "evaluation_authority": "independent_bilingual_review",
                                    "evaluation_receipt_sha256": digest(canonical(self.result))}}

    def write_registry(self, data=None, result=None):
        value = deepcopy(self.value if data is None else data)
        result = self.result if result is None else result
        receipt_raw = canonical(result)
        self.evaluation.write_bytes(receipt_raw)
        if isinstance(value.get("candidate"), dict):
            value["candidate"]["evaluation_receipt_sha256"] = digest(receipt_raw)
        manifest_raw = canonical(value)
        self.manifest.write_bytes(manifest_raw)
        return digest(manifest_raw)

    def registry(self, data=None, result=None):
        return ModelRegistry(self.manifest, expected_sha256=self.write_registry(data, result),
                             evaluation_receipt=self.evaluation)

    def test_default_is_disabled_and_never_imports_archive_or_invokes_loader(self):
        loaded = []
        runtime = SingleMultilingualResearchTranslator(
            checkpoint_root=self.model_root, load_model=lambda *a: loaded.append(a))
        state = runtime.status()
        self.assertFalse(state["candidate_configured"])
        self.assertEqual(state["verified_directions"], [])
        self.assertEqual(state["target_directions"], sorted(self.targets))
        with self.assertRaisesRegex(ValueError, "No verified single"):
            runtime.translate(request("es->en"))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            runtime.translate(request("fr->es"))
        self.assertFalse(loaded)
        self.assertEqual(runtime.loaded_model_count, 0)
        self.assertNotIn("use_previous", runtime.translate.__code__.co_varnames)

    def test_interleaved_directions_share_same_session_and_hash(self):
        called = []
        registry = self.registry()
        class Session:
            def translate(self, text, *, source, target):
                return f"{source}->{target}:{text}"
        def load(model, location):
            called.append((model.checkpoint_sha256, location))
            return Session()
        runtime = SingleMultilingualResearchTranslator(registry=registry,
                   checkpoint_root=self.model_root, load_model=load)
        directions = self.targets * 15
        with ThreadPoolExecutor(max_workers=5) as threads:
            outputs = list(threads.map(lambda d: runtime.translate(request(d, "x")), directions))
        self.assertEqual(outputs, [d + ":x" for d in directions])
        self.assertEqual(called, [(registry.candidate.checkpoint_sha256, self.model_root.resolve())])
        self.assertEqual(runtime.loaded_model_count, 1)
        self.assertFalse(runtime.status()["production_activated"])
        self.assertEqual(runtime.status()["checkpoint_sha256"], called[0][0])

    def test_reject_multiple_candidates_selectors_and_missing_independent_pins(self):
        bad = deepcopy(self.value)
        bad["candidate"] = [bad["candidate"], bad["candidate"]]
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.registry(bad)
        bad = deepcopy(self.value)
        bad["candidate"]["models_by_direction"] = {"en->es": "someone"}
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.registry(bad)
        self.write_registry()
        with self.assertRaisesRegex(ValueError, "independently pinned"):
            ModelRegistry(self.manifest, evaluation_receipt=self.evaluation)
        with self.assertRaisesRegex(ValueError, "independent pin"):
            ModelRegistry(self.manifest, expected_sha256="3"*64, evaluation_receipt=self.evaluation)

    def test_same_model_requires_pinned_complete_passing_direction_evidence(self):
        for change in ("missing", "extra", "failed", "wrong checkpoint"):
            alternate = deepcopy(self.result)
            if change == "missing":
                del alternate["directions"]["fr->en"]
            elif change == "extra":
                alternate["directions"]["fr->es"] = alternate["directions"]["fr->en"]
            elif change == "failed":
                alternate["directions"]["fr->en"]["status"] = "failed"
            else:
                alternate["checkpoint_sha256"] = "a"*64
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.registry(result=alternate)
        pin = self.write_registry()
        self.evaluation.write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "receipt does not match"):
            ModelRegistry(self.manifest, expected_sha256=pin, evaluation_receipt=self.evaluation)

    def test_wrong_or_extra_files_and_unsupported_route_never_load(self):
        called = []
        reg = self.registry()
        runtime = lambda: SingleMultilingualResearchTranslator(registry=reg,
                  checkpoint_root=self.model_root, load_model=lambda *a: called.append(a))
        for pair in ("auto->en", "fr->es", "en->en", "cmn->en"):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                runtime().translate(request(pair))
        (self.model_root / "shadow.py").write_bytes(b"nothing executed")
        with self.assertRaisesRegex(ValueError, "files differ"):
            runtime().translate(request("es->en"))
        (self.model_root / "shadow.py").unlink()
        (self.model_root / "model.safetensors").write_bytes(b"wrong")
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            runtime().translate(request("fr->en"))
        self.assertFalse(called)

    def test_failed_loader_never_switches_or_reloads_across_routes(self):
        count = []
        def loader(*args):
            count.append(1)
            raise RuntimeError("do not expose loader error")
        translator = SingleMultilingualResearchTranslator(
            registry=self.registry(), checkpoint_root=self.model_root, load_model=loader)
        with self.assertRaisesRegex(ValueError, "^The single multilingual checkpoint could not be loaded$"):
            translator.translate(request("es->en"))
        with self.assertRaisesRegex(ValueError, "no fallback"):
            translator.translate(request("fr->en"))
        self.assertEqual(count, [1])


if __name__ == "__main__":
    unittest.main()
