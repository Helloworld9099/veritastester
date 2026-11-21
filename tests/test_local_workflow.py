"""End-to-end tests for the actual local CLI, not a model or external workflow."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CANARY = "never-print-private-input-path"
EMPTY_PLAN = {
    "schema_version": 1, "kind": "local_research_workflow",
    "data": None, "checkpoint": None, "comparison": None,
}


def _json_bytes(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


class LocalWorkflowCLI(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=CANARY + "-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.inputs = self.base / "inputs"
        self.inputs.mkdir()
        self.store = self.base / "local-store"
        self.plan = self.inputs / "stage-plan.json"

    def _invoke(self, *args):
        env = os.environ.copy()
        for key in ("PYTHONPATH", "PYTHONHOME", "GITHUB_TOKEN", "GH_TOKEN"):
            env.pop(key, None)
        env.update(PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        result = subprocess.run(
            [sys.executable, "-m", "research.local_workflow", *args], cwd=ROOT,
            env=env, capture_output=True, timeout=20, text=True, check=False,
        )
        self.assertNotIn(CANARY, result.stdout + result.stderr)
        self.assertNotIn(str(self.base), result.stdout + result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        return result

    def _write_plan(self, plan=None):
        payload = _json_bytes(EMPTY_PLAN if plan is None else plan)
        self.plan.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()

    def _run(self, pin, *, run_id="run-001"):
        return self._invoke("run", "--plan", str(self.plan), "--plan-sha256", pin,
                            "--input-root", str(self.inputs), "--store", str(self.store),
                            "--run-id", run_id)

    def _audit(self, run_id="run-001"):
        return self._invoke("audit", "--store", str(self.store), "--run-id", run_id)

    def _snapshot(self):
        return {str(path.relative_to(self.store)): path.read_bytes()
                for path in self.store.rglob("*") if path.is_file()}

    def _decoded(self, result):
        return json.loads(result.stdout)

    def test_omitted_phases_are_not_verified_and_audit_is_read_only(self):
        pin = self._write_plan()
        result = self._run(pin)
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertEqual(self._decoded(result)["status"], "not_verified")
        before = self._snapshot()
        self.assertTrue(before, "valid incomplete run did not leave auditable evidence")
        audit = self._audit()
        self.assertEqual(audit.returncode, 0, audit.stderr)
        observed = self._decoded(audit)
        self.assertIsInstance(observed, dict)
        self.assertEqual(before, self._snapshot(), "audit modified the local run")

    def test_same_run_identity_does_not_overwrite_evidence(self):
        pin = self._write_plan()
        self.assertEqual(self._run(pin).returncode, 3)
        before = self._snapshot()
        self.assertTrue(before)
        # A different valid representation (new byte digest) still cannot claim
        # the earlier run's identity. Whether duplicate exits before reporting
        # its content or shows a blocked status, evidence must remain exact.
        self.plan.write_bytes(self.plan.read_bytes() + b"\n")
        changed = hashlib.sha256(self.plan.read_bytes()).hexdigest()
        self.assertNotEqual(pin, changed)
        duplicate = self._run(changed)
        self.assertNotEqual(duplicate.returncode, 0, duplicate.stdout)
        self.assertEqual(before, self._snapshot())

    def test_wrong_plan_checksum_and_uppercase_digest_do_not_publish(self):
        pin = self._write_plan()
        for bad in ("0" * 64, pin.upper()):
            with self.subTest(digest_kind="malformed" if bad == pin.upper() else "incorrect"):
                result = self._run(bad)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(self._snapshot())
        self.assertNotEqual(self._audit().returncode, 0)

    def test_plan_cannot_claim_sealed_input_or_escape_run_directory(self):
        plan = {**EMPTY_PLAN, "dataset_role": "sealed_holdout"}
        result = self._run(self._write_plan(plan))
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self._snapshot())
        good = self._write_plan()
        result = self._run(good, run_id="../unintended")
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self._snapshot())
        self.assertFalse((self.base / "unintended").exists())

    def _data_plan(self, *, sealed=False):
        rows = [
            ("train", {"id": "omitted-train-id", "source": "synthetic canary A",
                       "target": "synthetic target A"}),
            ("dev", {"id": "omitted-dev-id", "source": "synthetic canary B",
                     "target": "synthetic target B"}),
        ]
        datasets = []
        for role, row in rows:
            filename = f"{role}-tiny-public.jsonl"
            payload = _json_bytes(row)
            if not sealed:
                (self.inputs / filename).write_bytes(payload)
            datasets.append({"dataset_id": "synthetic-local-demo", "version": "v1",
                             "role": "holdout" if sealed and role == "dev" else role,
                             "path": filename, "sha256": hashlib.sha256(payload).hexdigest(),
                             "source_language": "es", "target_language": "en",
                             "origin_uri": "https://example.invalid/synthetic-local-demo",
                             "license": "CC-BY-4.0", "rights": "public_research",
                             "rights_evidence": "synthetic-license-fixture",
                             "privacy": "sealed" if sealed and role == "dev" else "public",
                             "review_status": "reviewed", "review_evidence": "synthetic-review-fixture"})
        return {**EMPTY_PLAN, "data": {"schema_version": 1, "purpose": "public_research",
                                      "datasets": datasets}}

    def test_valid_data_phase_cannot_imply_evaluation_or_training(self):
        plan = self._data_plan()
        result = self._run(self._write_plan(plan))
        self.assertEqual(result.returncode, 3, result.stderr)
        outer = self._decoded(result)
        receipt = outer["receipt"]
        self.assertEqual(receipt["status"], "not_verified")
        self.assertEqual(receipt["phases"]["data"]["status"], "pass")
        self.assertEqual(receipt["phases"]["checkpoint"]["status"], "not_verified")
        self.assertEqual(receipt["phases"]["comparison"]["status"], "not_verified")
        audit = self._audit()
        self.assertEqual(audit.returncode, 0, audit.stderr)
        for raw in (result.stdout.encode(), audit.stdout.encode(), *self._snapshot().values()):
            for secret in ("synthetic canary", "omitted-train-id", "train-tiny-public.jsonl"):
                self.assertNotIn(secret.encode(), raw)

    def test_sealed_data_declaration_never_leaves_a_completed_run(self):
        # No fixture files exist, so this tests the metadata boundary without
        # constructing or reading even a mock holdout payload.
        result = self._run(self._write_plan(self._data_plan(sealed=True)))
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        audit = self._audit()
        self.assertNotEqual(audit.returncode, 0, audit.stdout)

    def _comparison_plan(self, *, changed=False, candidate_fixture=None, checkpoint_sha=None):
        baseline = {
            "schema_version": 1, "kind": "bridgecaption_aggregate_receipt",
            "stage": "eval", "scope": "development", "direction": "es->en",
            "fixture_sha256": "a" * 64, "method_sha256": "b" * 64,
            "checkpoint_sha256": checkpoint_sha or "c" * 64, "output_sha256": "d" * 64,
            "counts": {"paired_rows": 3}, "metrics": {"chrf": 73.0}, "status": "completed",
        }
        candidate = dict(baseline)
        if changed:
            candidate.update(output_sha256="e" * 64, metrics={"chrf": 75.0})
        if candidate_fixture is not None:
            candidate["fixture_sha256"] = candidate_fixture
        plan = {**EMPTY_PLAN, "comparison": {}}
        for typ, obj in (("baseline", baseline), ("candidate", candidate)):
            name = f"{typ}-synthetic-metadata.json"
            payload = _json_bytes(obj)
            (self.inputs / name).write_bytes(payload)
            plan["comparison"][typ] = {"path": name, "sha256": hashlib.sha256(payload).hexdigest()}
        return plan

    def test_pinned_comparison_mismatch_is_a_run_failure(self):
        result = self._run(self._write_plan(self._comparison_plan(changed=True)))
        self.assertEqual(result.returncode, 3, result.stderr)
        output = self._decoded(result)
        self.assertEqual(output["status"], "fail")
        self.assertEqual(output["receipt"]["phases"]["comparison"]["status"], "fail")
        self.assertEqual(output["receipt"]["phases"]["data"]["status"], "not_verified")
        self.assertEqual(self._audit().returncode, 0)

    def test_reported_exact_match_stays_unverified_without_checkpoint_binding(self):
        result = self._run(self._write_plan(self._comparison_plan()))
        self.assertEqual(result.returncode, 3, result.stderr)
        output = self._decoded(result)
        self.assertEqual(output["status"], "not_verified")
        self.assertEqual(output["receipt"]["phases"]["comparison"]["status"], "not_verified")
        self.assertFalse(output["promotion_eligible"])
        self.assertFalse(output["training_dispatched"])

    def test_incomparable_pinned_receipts_run_finishes_unverified(self):
        plan = self._comparison_plan(candidate_fixture="f" * 64)
        result = self._run(self._write_plan(plan))
        self.assertEqual(result.returncode, 3, result.stderr)
        output = self._decoded(result)
        self.assertEqual(output["state"], "completed")
        self.assertEqual(output["receipt"]["phases"]["comparison"]["status"], "not_verified")

    def _opaque_checkpoint(self):
        name = "dummy-bytes.safetensors"
        payload = b"synthetic opaque bytes only; not a model"
        (self.inputs / name).write_bytes(payload)
        return {
            "direction": "es->en", "architecture": "marian", "base_revision": "f" * 40,
            "tokenizer_sha256": "d" * 64, "checkpoint_path": name,
            "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
        }

    def test_comparison_cannot_use_an_unrelated_verified_checkpoint(self):
        plan = {**self._comparison_plan(), "checkpoint": self._opaque_checkpoint()}
        result = self._run(self._write_plan(plan))
        self.assertEqual(result.returncode, 3, result.stderr)
        phases = self._decoded(result)["receipt"]["phases"]
        self.assertEqual(phases["checkpoint"]["status"], "pass")
        self.assertEqual(phases["comparison"]["status"], "not_verified")

    def test_fully_matching_local_evidence_is_not_a_promotion_or_training(self):
        checkpoint = self._opaque_checkpoint()
        plan = self._comparison_plan(checkpoint_sha=checkpoint["checkpoint_sha256"])
        plan["data"] = self._data_plan()["data"]
        plan["checkpoint"] = checkpoint
        pin = self._write_plan(plan)
        result = self._run(pin)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = self._decoded(result)
        self.assertEqual((output["state"], output["status"]), ("completed", "pass"))
        self.assertFalse(output["promotion_eligible"])
        self.assertFalse(output["training_dispatched"])
        self.assertEqual(output["plan_sha256"], pin)
        for phase in output["receipt"]["phases"].values():
            self.assertEqual(phase["status"], "pass")
            self.assertRegex(phase["digest"], r"[0-9a-f]{64}\Z")
        self.assertEqual(self._decoded(self._audit()), output)
        for raw in (result.stdout.encode(), *self._snapshot().values()):
            self.assertNotIn(b"synthetic opaque bytes", raw)
            self.assertNotIn(b"dummy-bytes.safetensors", raw)
            self.assertNotIn(b"baseline-synthetic-metadata", raw)

    def test_changed_candidate_receipt_cannot_complete(self):
        plan = self._comparison_plan()
        pin = self._write_plan(plan)
        candidate = self.inputs / plan["comparison"]["candidate"]["path"]
        candidate.write_bytes(candidate.read_bytes() + b"\n")
        self.assertEqual(self._run(pin).returncode, 2)
        audit = self._audit()
        self.assertNotEqual(audit.returncode, 0, audit.stdout)


if __name__ == "__main__":
    unittest.main()
