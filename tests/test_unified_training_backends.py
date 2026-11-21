"""Verify the inspected legacy trainer against current repository bytes only."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from research.unified_training.backends import (
    BackendSourceError, UnsupportedBackend, backend_for, plan_backend,
)
from research.unified_training.config import TrainingConfig


ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "experiments/cpu_continuation_20261117/train.py"
INTENTIONAL_PRIVATE_LOCATOR = "private-personal-subdir/records.manifest.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def config_dict(direction="es->en"):
    language = "spanish" if direction == "es->en" else "french"
    prior = ROOT / f"research/neural_continuation_20261116/{language}"
    protocol_data = (prior / "protocol.json").read_bytes()
    protocol = json.loads(protocol_data)
    base_manifest_dir = ("translation_training_20261116" if language == "spanish"
                         else "translation_fr_training_20261116")
    files = json.loads((ROOT / "research" / base_manifest_dir / "model/download_manifest.json").read_bytes())
    return {
        "schema_version": 1, "direction": direction, "architecture": "marian",
        "backend": {"id": "marian_hf", "version": "1.0.0"},
        "base": {"model_id": protocol["model"]["model"], "revision": protocol["model"]["revision"],
                 "files": {f["file"]: f["sha256"] for f in files}},
        "device": {"kind": "cpu", "gpu_opt_in": False, "precision": "fp32", "threads": 4},
        "strategy": {"kind": "lora", "rank": 8, "alpha": 16,
                     "target_modules": protocol["target_modules"], "protocol_sha256": sha(protocol_data)},
        "optimizer": {"name": "adamw", "learning_rate": 2.5e-5, "weight_decay": 0.01,
                      "beta1": .9, "beta2": .999, "epsilon": 1e-8},
        "schedule": {"kind": "constant", "total_updates": 96, "warmup_updates": 0,
                     "micro_batch_size": 4, "gradient_accumulation_steps": 4,
                     "max_source_tokens": 96, "max_target_tokens": 96},
        "seed": 20261117, "rights": "private_noncommercial", "output_id": "test-new-id",
        "data": [{"dataset_id": "training-fixture-not-for-receipt", "version": "v1",
                  "manifest_sha256": "a"*64, "split": "train", "visibility": "private",
                  "license": "CC0-1.0", "local_path": INTENTIONAL_PRIVATE_LOCATOR}],
        "source": {"mode": "base"},
    }


def source_weights(raw, mode="warm_start", weights_sha=None):
    prior = ROOT / ("research/neural_continuation_20261116/spanish"
                    if raw["direction"] == "es->en" else "research/neural_continuation_20261116/french")
    base = TrainingConfig.from_mapping(raw)
    return {
        "mode": mode, "artifact_id":"local-prior", "manifest_sha256":"b"*64,
        "weights_sha256": weights_sha or sha((prior/"adapter_selected.safetensors").read_bytes()),
        "compatibility_sha256":base.compatibility_sha256, "ancestry_sha256":"c"*64,
        "rights":"private_noncommercial",
        **({"resume_fingerprint":base.resume_fingerprint,
            "component_sha256":{k:"d"*64 for k in ("optimizer","scheduler","sampler","rng")}}
           if mode == "exact_resume" else {})
    }


class LegacyPlannerTests(unittest.TestCase):
    def test_plan_reads_real_legacy_artifacts_without_a_launch(self):
        for direction, lang in (("es->en","spanish"),("fr->en","french")):
            raw = config_dict(direction)
            # Parse-only source manifest/ancestry placeholders are deliberately
            # not actual execution evidence. The pinned historical weights are.
            raw["source"] = source_weights(raw)
            with patch("subprocess.Popen", side_effect=AssertionError("no subprocess")),                  patch("urllib.request.urlopen", side_effect=AssertionError("no network")):
                plan = plan_backend(TrainingConfig.from_mapping(raw), repo_root=ROOT).to_mapping()
            self.assertEqual(plan["status"],"inspection_only")
            self.assertEqual(plan["legacy_arguments"]["--language"], lang)
            self.assertEqual(plan["entrypoint_sha256"],sha(LEGACY.read_bytes()))
            self.assertFalse(plan["execution_supported"])
            self.assertFalse(plan["capabilities"]["offline_execution_supported"])
            self.assertFalse(plan["capabilities"]["exact_resume_supported"])
            self.assertEqual(plan["observed_recipe"]["checkpoint_component"],"adapter_weights_only")
            self.assertEqual(plan["observed_recipe"]["schedule_total_updates"],96)
            self.assertEqual({x["code"] for x in plan["issues"]},
                             {"legacy_cannot_bind_pinned_local_dataset",
                              "network_fetch_required_by_legacy_call","no_execution_route_in_m3"})
            serialized=json.dumps(plan)
            self.assertNotIn(str(ROOT),serialized)
            self.assertNotIn(INTENTIONAL_PRIVATE_LOCATOR,serialized)
            self.assertNotIn(raw["data"][0]["dataset_id"],serialized)

    def test_mapped_argument_names_exist_in_real_legacy_cli(self):
        source_tree = ast.parse(LEGACY.read_bytes())
        main = next(node for node in source_tree.body
                    if isinstance(node, ast.FunctionDef) and node.name=="main")
        flag_names = {node.args[0].value for node in ast.walk(main)
                      if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                      and node.func.attr == "add_argument" and isinstance(node.args[0],ast.Constant)}
        plan = plan_backend(TrainingConfig.from_mapping(config_dict()),repo_root=ROOT)
        self.assertEqual(set(dict(plan.legacy_arguments)),flag_names)
        self.assertEqual(flag_names,{"--language","--repo-root","--output-dir"})

    def test_gpu_quantization_and_config_overrides_are_unsupported(self):
        raw = config_dict()
        raw["device"].update(kind="gpu", gpu_opt_in=True, precision="bf16", threads=8)
        raw["strategy"].update(kind="qlora", load_bits=4)
        raw["seed"]=4
        raw["schedule"]["total_updates"]=13
        raw["optimizer"]["learning_rate"]=0.001
        config = TrainingConfig.from_mapping(raw)
        result = plan_backend(config,repo_root=ROOT).to_mapping()
        self.assertEqual(result["status"],"blocked")
        self.assertFalse(result["execution_supported"])
        self.assertTrue({"gpu_not_supported","unsupported_strategy","legacy_fixed_seed",
                         "legacy_fixed_schedule_value","legacy_fixed_optimizer_value",
                         "legacy_fixed_cpu_threads","unsupported_precision"} <= {i["code"] for i in result["issues"]})

    def test_warm_weights_only_is_neither_custom_init_nor_exact_resume(self):
        raw = config_dict()
        raw["source"]=source_weights(raw, weights_sha="e"*64)
        plan=plan_backend(TrainingConfig.from_mapping(raw),repo_root=ROOT)
        self.assertIn("legacy_cannot_select_custom_warm_weights", {i.code for i in plan.issues})
        raw["source"]={"mode":"base"}
        raw["source"]=source_weights(raw, mode="exact_resume")
        plan=plan_backend(TrainingConfig.from_mapping(raw),repo_root=ROOT)
        self.assertIn("exact_resume_unsupported_no_state", {i.code for i in plan.issues})
        self.assertFalse(plan.capabilities.exact_resume_supported)

    def test_other_directions_and_unknown_backends_never_fall_back(self):
        raw = config_dict()
        raw["direction"]="cmn->en"
        plan=plan_backend(TrainingConfig.from_mapping(raw),repo_root=ROOT)
        self.assertIn("unsupported_direction",{i.code for i in plan.issues})
        self.assertFalse(plan.legacy_arguments)
        with self.assertRaisesRegex(UnsupportedBackend,"unsupported_backend_identity"):
            backend_for("unknown_user_supplied_module","1.0.0")
        with self.assertRaisesRegex(UnsupportedBackend,"unsupported_backend_identity"):
            backend_for("marian_hf","2.0.0")

    def test_tampered_and_symlinked_legacy_sources_fail_closed(self):
        config = TrainingConfig.from_mapping(config_dict())
        with TemporaryDirectory() as temp:
            root=Path(temp)
            wrong=root/LEGACY.relative_to(ROOT)
            wrong.parent.mkdir(parents=True)
            shutil.copyfile(LEGACY,wrong)
            wrong.write_bytes(wrong.read_bytes()+b"\n")
            with self.assertRaisesRegex(BackendSourceError, "legacy_source_pin_mismatch"):
                plan_backend(config,repo_root=root)
            wrong.unlink()
            wrong.symlink_to(LEGACY)
            with self.assertRaisesRegex(BackendSourceError, "legacy_source_symlink_rejected"):
                plan_backend(config,repo_root=root)


if __name__ == "__main__":
    unittest.main()
