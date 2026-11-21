"""Test local fixture boundaries and paired per-direction CLI behavior."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from research.unified_eval.__main__ import main


def clone_json(obj):
    return json.loads(json.dumps(obj))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


class EvalCLI(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="secret-directory-not-for-report-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.paths = {name: self.root / (name + (".json" if name == "manifest" else ".jsonl"))
                      for name in ("suite", "manifest", "baseline", "candidate")}
        self.rows = [
            {"id":"private-es-case", "source_language":"es", "target_language":"en",
             "source":"source ES demo for no raw output", "reference":"The blue door is locked at six.",
             "categories":["negation","private_category"]},
            {"id":"private-en-case", "source_language":"en", "target_language":"es",
             "source":"source EN demo for no raw output", "reference":"La puerta está cerrada a las seis.",
             "categories":["time"]},
        ]
        self.baseline = {
            "private-es-case": "The blue door is locked at six.", "private-en-case": "Partido de amigos."
        }
        self.candidate = {
            "private-es-case": "Late summer festival.", "private-en-case": "La puerta está cerrada a las seis."
        }
        self.directions = {"private-es-case":"es->en", "private-en-case":"en->es"}
        self.manifest = {"kind":"dev_suite_manifest","schema_version":1,"scope":"development",
                         "split":"dev","sealed":False,"fixture_rows":2,
                         "directions":{"es->en":1,"en->es":1},
                         "baseline_outputs_sha256_by_direction":{
                             self.directions[k]: digest(canonical({k:v})) for k,v in self.baseline.items()}}
        self.write_suite()

    def write_suite(self):
        payload = b"".join(canonical(r)+b"\n" for r in self.rows)
        self.paths["suite"].write_bytes(payload)
        self.manifest["fixture_sha256"] = digest(payload)
        self.paths["manifest"].write_bytes(canonical(self.manifest))
        self.write_receipt("baseline", self.baseline)
        self.write_receipt("candidate", self.candidate)

    def write_receipt(self, variant, outputs, **header_updates):
        header = {"kind":"dev_prediction_receipt","schema_version":1,"scope":"development",
                  "split":"dev","sealed":False,"variant":variant,
                  "fixture_sha256":self.manifest["fixture_sha256"],
                  "directions":{"es->en":{},"en->es":{}}}
        header.update(header_updates)
        out=[{"id":k, "direction":self.directions[k], "output":v} for k,v in outputs.items()]
        self.paths[variant].write_bytes(b"".join(canonical(r)+b"\n" for r in [header,*out]))

    def args(self, command="compare", lane="model_dev"):
        keys=("suite","manifest") if command=="preflight" else ("suite","manifest","baseline","candidate")
        args=[command]
        for k in keys:
            p=self.paths[k]
            args.extend([f"--{k}",str(p),f"--{k}-sha256",digest(p.read_bytes())])
        if command=="compare":
            args.extend(["--lane",lane])
        return args

    def run_cli(self, args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status=main(args)
        raw=stdout.getvalue()+stderr.getvalue()
        for secret in ("private-es-case","private-en-case","private_category","source ES demo",
                       "Late summer",str(self.root)):
            self.assertNotIn(secret,raw)
        return status,json.loads(raw)

    def test_sealed_manifest_preempts_access_to_suite(self):
        self.manifest.update(split="holdout",sealed=True)
        self.paths["manifest"].write_bytes(canonical(self.manifest))
        args=self.args("preflight")
        self.paths["suite"].unlink()
        self.assertEqual(self.run_cli(args),(
            2,{"status":"rejected","code":"manifest_not_open_development","promotion_eligible":False}))

    def test_unknown_dataset_role_and_missing_direction_rejected(self):
        self.manifest["dataset_role"]="sealed"
        self.paths["manifest"].write_bytes(canonical(self.manifest))
        self.assertEqual(self.run_cli(self.args("preflight"))[1]["code"],"manifest_invalid_metadata")
        self.manifest.pop("dataset_role")
        self.manifest.pop("directions")
        self.paths["manifest"].write_bytes(canonical(self.manifest))
        self.assertEqual(self.run_cli(self.args("preflight"))[0],2)

    def test_nul_and_lone_surrogate_are_invalid_preflight(self):
        for corrupt in ("escape\0tail","escape\ud800tail"):
            with self.subTest(corrupt=repr(corrupt)):
                self.rows[0]["reference"]=corrupt
                self.paths["suite"].write_bytes(b"".join(
                    json.dumps(r,ensure_ascii=True).encode("ascii")+b"\n" for r in self.rows))
                self.manifest["fixture_sha256"]=digest(self.paths["suite"].read_bytes())
                self.paths["manifest"].write_bytes(canonical(self.manifest))
                self.assertEqual(self.run_cli(self.args("preflight"))[1]["code"], "suite_invalid_json")

    def test_explicit_frozen_direction_layout_needed(self):
        self.manifest["directions"]={"es->en":2}
        self.paths["manifest"].write_bytes(canonical(self.manifest))
        self.assertEqual(self.run_cli(self.args("preflight"))[1]["code"],"fixture_directions_mismatch")

    def test_unpinned_or_sealed_receipt_is_not_scored(self):
        self.write_receipt("candidate",self.candidate, sealed=True)
        self.assertEqual(self.run_cli(self.args())[1]["code"],"candidate_not_open_development")
        self.write_receipt("candidate",self.candidate)
        args=self.args()
        args[args.index("--candidate-sha256")+1]="0"*64
        self.assertEqual(self.run_cli(args)[1]["code"],"candidate_checksum_mismatch")
        self.write_receipt("candidate",self.candidate,dataset_role="sealed")
        self.assertEqual(self.run_cli(self.args())[1]["code"],"candidate_invalid_metadata")

    def test_one_improved_direction_cannot_override_degradation(self):
        try:
            import sacrebleu
        except ImportError:
            self.skipTest("SacreBLEU evaluation extra not installed")
        status, report=self.run_cli(self.args())
        self.assertEqual((status,report["status"]),(3,"fail"))
        self.assertFalse(report["promotion_eligible"])
        es=report["directions"]["es->en"]
        en=report["directions"]["en->es"]
        self.assertEqual(es["gate"]["checks"]["direction_chrf"],"fail")
        self.assertEqual(en["gate"]["checks"]["direction_chrf"],"pass")
        self.assertLess(es["diagnostics"]["candidate"]["chrf"],es["diagnostics"]["baseline"]["chrf"])
        self.assertGreater(en["diagnostics"]["candidate"]["chrf"],en["diagnostics"]["baseline"]["chrf"])

    def test_exact_outputs_without_evidence_not_verified(self):
        try:
            import sacrebleu
        except ImportError:
            self.skipTest("SacreBLEU evaluation extra not installed")
        self.write_receipt("candidate",self.baseline)
        status, report=self.run_cli(self.args(lane="infrastructure"))
        self.assertEqual((status,report["status"]),(3,"not_verified"))
        self.assertEqual(report["coverage_scope"],"only_explicit_fixture_directions")
        for route in report["directions"].values():
            self.assertEqual(route["observed_exact_output_matches"],1)
            self.assertIn("independently_verified_complete_checkpoint_pins_required",
                          route["gate"]["blockers"])
            self.assertEqual(route["gate"]["checks"]["output_parity"],"pass")


if __name__ == "__main__":
    unittest.main()
