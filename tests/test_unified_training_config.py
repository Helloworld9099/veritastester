"""Structural/offline tests for immutable training metadata, no real inputs."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest

from research.unified_training.config import ConfigError, TrainingConfig, load_config

DIGEST = "a"*64


def specimen():
    return {
        "schema_version":1, "direction":"es->en", "architecture":"marian",
        "backend":{"id":"marian_hf","version":"1.0.0"},
        "base":{"model_id":"TestOnly/marian","revision":"0"*40,
                "files":{name:"1"*64 for name in ("config.json","source.spm","target.spm",
                                                  "vocab.json","model.safetensors")}},
        "device":{"kind":"cpu","gpu_opt_in":False,"precision":"fp32","threads":4},
        "strategy":{"kind":"lora","rank":8,"alpha":16,
                    "target_modules":["model.encoder.layers.0.self_attn.q_proj"],
                    "protocol_sha256":"2"*64},
        "optimizer":{"name":"adamw","learning_rate":0.0001,"weight_decay":0.01,
                     "beta1":0.9,"beta2":0.999,"epsilon":1e-8},
        "schedule":{"kind":"constant","total_updates":96,"warmup_updates":0,
                    "micro_batch_size":4,"gradient_accumulation_steps":4,
                    "max_source_tokens":96,"max_target_tokens":96},
        "seed":20261117,"rights":"commercial_candidate",
        "data":[{"dataset_id":"d0","version":"v1","manifest_sha256":"3"*64,"split":"train",
                 "visibility":"public","license":"CC-BY-4.0","local_path":"test/open_train.json"},
                {"dataset_id":"d0","version":"v1","manifest_sha256":"4"*64,"split":"dev",
                 "visibility":"public","license":"CC-BY-4.0","local_path":"test/open_dev.json"}],
        "source":{"mode":"base"},"output_id":"independent-run",
    }


def sources(config, *, mode="warm_start", rights="commercial_candidate"):
    entry = {"mode":mode,"artifact_id":"prior","manifest_sha256":"5"*64,
             "weights_sha256":"6"*64,"compatibility_sha256":config.compatibility_sha256,
             "ancestry_sha256":"7"*64,"rights":rights}
    if mode == "exact_resume":
        entry.update(resume_fingerprint=config.resume_fingerprint,
                     component_sha256={k:"8"*64 for k in ("optimizer","scheduler","rng","sampler")})
    return entry


class ConfigTest(unittest.TestCase):
    def test_base_declares_no_source_and_metadata_is_deeply_immutable(self):
        raw = specimen()
        conf = TrainingConfig.from_mapping(raw)
        self.assertEqual(conf.direction,"es->en")
        self.assertEqual(conf.source["mode"],"base")
        self.assertEqual(conf.schedule["total_updates"],96)
        with self.assertRaises(TypeError):
            conf.base["files"]["source.spm"] = "9"*64
        with self.assertRaises(TypeError):
            conf.data[0]["license"] = "untrusted"
        with self.assertRaises(AttributeError):
            conf.seed = 100
        raw["data"][0]["manifest_sha256"] = DIGEST
        result = conf.to_mapping()
        result["strategy"]["target_modules"].append("edited")
        self.assertNotEqual(conf.data[0]["manifest_sha256"],DIGEST)
        self.assertEqual(len(conf.strategy["target_modules"]),1)
        self.assertEqual(TrainingConfig.from_mapping(conf.to_mapping()),conf)

    def test_structural_scope_and_backend_only(self):
        edits = (
            lambda r:r.update(direction="auto->en"),
            lambda r:r.update(direction="fr->fr"),
            lambda r:r.update(schema_version=True),
            lambda r:r.update(architecture="mystery"),
            lambda r:r["backend"].update(version="latest"),
            lambda r:r["backend"].update(id="import.some.module"),
            lambda r:r["base"].update(revision="main"),
            lambda r:r["base"]["files"].pop("target.spm"),
            lambda r:r["base"]["files"].update({"../config.json":DIGEST}),
            lambda r:r["base"]["files"].update({"source.spm":"missing"}),
            lambda r:r.update(run_after_load=True),
            lambda r:r.update(output_id="../existing"),
        )
        for mutation in edits:
            bad = specimen()
            mutation(bad)
            with self.subTest(mutation=mutation), self.assertRaises(ConfigError):
                TrainingConfig.from_mapping(bad)

    def test_device_and_lora_strategy_metadata_opt_in(self):
        raw = specimen()
        raw["device"].update(kind="gpu",precision="bf16")
        with self.assertRaisesRegex(ConfigError,"explicit opt-in"):
            TrainingConfig.from_mapping(raw)
        raw["device"]["gpu_opt_in"] = True
        raw["strategy"].update(kind="qlora",load_bits=4)
        self.assertEqual(TrainingConfig.from_mapping(raw).strategy["load_bits"],4)
        raw["strategy"]["load_bits"] = 8
        with self.assertRaises(ConfigError):
            TrainingConfig.from_mapping(raw)
        raw = specimen()
        raw["strategy"].update(kind="qlora",load_bits=4)
        with self.assertRaisesRegex(ConfigError,"GPU"):
            TrainingConfig.from_mapping(raw)
        raw = specimen()
        raw["device"].update(precision="fp16")
        with self.assertRaises(ConfigError):
            TrainingConfig.from_mapping(raw)
        raw = specimen()
        raw["strategy"]["target_modules"] += raw["strategy"]["target_modules"]
        with self.assertRaisesRegex(ConfigError,"Duplicate"):
            TrainingConfig.from_mapping(raw)

    def test_no_sealed_test_or_noncommercial_ancestry_mixing(self):
        raw = specimen()
        for split in ("test","holdout","bootstrap",None):
            raw["data"][0]["split"] = split
            with self.subTest(split=split), self.assertRaises(ConfigError):
                TrainingConfig.from_mapping(raw)
        raw = specimen()
        raw["data"][0]["license"] = "CC-BY-NC-4.0"
        with self.assertRaisesRegex(ConfigError,"commercial"):
            TrainingConfig.from_mapping(raw)
        raw.update(rights="private_noncommercial")
        raw["data"][0]["visibility"] = "private"
        self.assertEqual(TrainingConfig.from_mapping(raw).rights,"private_noncommercial")
        raw["data"][0]["visibility"] = "sealed"
        with self.assertRaises(ConfigError):
            TrainingConfig.from_mapping(raw)
        raw = specimen()
        config = TrainingConfig.from_mapping(raw)
        raw["source"] = sources(config,rights="private_noncommercial")
        with self.assertRaisesRegex(ConfigError,"Noncommercial ancestry"):
            TrainingConfig.from_mapping(raw)

    def test_train_dev_identity_and_manifest_paths(self):
        changes = [
            lambda r:r["data"][1].update(manifest_sha256=r["data"][0]["manifest_sha256"]),
            lambda r:r["data"][1].update(split="train"),
            lambda r:r["data"][1].update(local_path=r["data"][0]["local_path"]),
            lambda r:r["data"][0].update(local_path="open/../train.json"),
            lambda r:r["data"][0].update(local_path="open/train.jsonl"),
            lambda r:r["data"].pop(0),
        ]
        for edit in changes:
            raw = specimen()
            edit(raw)
            with self.subTest(edit=edit), self.assertRaises(ConfigError):
                TrainingConfig.from_mapping(raw)

    def test_warm_start_pins_format_but_not_optimizer_recipe(self):
        raw = specimen()
        original = TrainingConfig.from_mapping(raw)
        raw["source"] = sources(original)
        raw["optimizer"]["learning_rate"] = .0002
        config = TrainingConfig.from_mapping(raw)
        self.assertEqual(config.compatibility_sha256,original.compatibility_sha256)
        self.assertNotEqual(config.resume_fingerprint,original.resume_fingerprint)
        changed_fields = (
            lambda r:r.update(direction="fr->en"),
            lambda r:r["backend"].update(version="1.0.1"),
            lambda r:r["base"]["files"].update({"vocab.json":DIGEST}),
            lambda r:r["strategy"].update(rank=16),
        )
        for edit in changed_fields:
            attempt = deepcopy(raw)
            edit(attempt)
            with self.subTest(edit=edit), self.assertRaisesRegex(ConfigError,"incompatible"):
                TrainingConfig.from_mapping(attempt)
        raw["source"]["component_sha256"] = {}
        with self.assertRaises(ConfigError):
            TrainingConfig.from_mapping(raw)

    def test_exact_resume_requires_each_state_and_pins_ordered_recipe(self):
        raw = specimen()
        original = TrainingConfig.from_mapping(raw)
        raw["source"] = sources(original,mode="exact_resume")
        raw.update(output_id="new-location")
        raw["data"][0]["local_path"] = "copy/train.json"
        resumed = TrainingConfig.from_mapping(raw)
        self.assertEqual(resumed.resume_fingerprint,original.resume_fingerprint)
        changes = (
            lambda r:r["source"]["component_sha256"].pop("optimizer"),
            lambda r:r["source"]["component_sha256"].pop("rng"),
            lambda r:r["source"]["component_sha256"].pop("scheduler"),
            lambda r:r["source"]["component_sha256"].pop("sampler"),
            lambda r:r.update(seed=42),
            lambda r:r["device"].update(threads=2),
            lambda r:r["schedule"].update(total_updates=128),
            lambda r:r["optimizer"].update(learning_rate=.0002),
            lambda r:r["data"][1].update(manifest_sha256=DIGEST),
            lambda r:r["data"].reverse(),
            lambda r:r.update(output_id="prior"),
        )
        for edit in changes:
            block = deepcopy(raw)
            edit(block)
            with self.subTest(edit=edit), self.assertRaises(ConfigError):
                TrainingConfig.from_mapping(block)

    def test_config_file_open_is_metadata_only_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"run.json"
            path.write_text(json.dumps(specimen()))
            self.assertEqual(load_config(path).source["mode"],"base")
            path.write_text(path.read_text().replace('"schema_version": 1',
                                                      '"schema_version": 1, "schema_version": 1'))
            with self.assertRaisesRegex(ConfigError,"Duplicate"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
