"""Offline artifact verification/privacy and aggregate comparison contracts."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from research.workflow_contract import (
    ContractError, MAX_JSON_BYTES, compare_receipts, read_pinned_json,
    validate_receipt, verify_checkpoint, verify_receipt,
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def public_eval():
    return {
        "schema_version":1, "kind":"bridgecaption_aggregate_receipt",
        "stage":"eval", "scope":"development", "direction":"es->en",
        "fixture_sha256":"1"*64, "method_sha256":"2"*64,
        "checkpoint_sha256":"3"*64, "output_sha256":"4"*64,
        "counts":{"paired_rows":16}, "metrics":{"chrf":73.5},
        "status":"completed",
    }


class TestWorkflowContract(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        (self.root/"receipts").mkdir()

    def make(self, name="receipt.json", value=None, raw=None):
        value = public_eval() if value is None else value
        if raw is None:
            raw = json.dumps(value).encode()
        path = self.root/"receipts"/name
        path.write_bytes(raw)
        return f"receipts/{name}", sha(raw)

    def verified(self, name, value):
        path,pin = self.make(name, value)
        return verify_receipt(self.root,path,expected_sha256=pin)

    def test_pin_exact_raw_bytes_before_any_json_parse(self):
        path,pin = self.make(raw=b'not even valid JSON')
        with self.assertRaisesRegex(ContractError, "receipt_raw_sha256_mismatch"):
            read_pinned_json(self.root,path,expected_sha256="0"*64)
        with self.assertRaisesRegex(ContractError, "invalid_receipt_json"):
            read_pinned_json(self.root,path,expected_sha256=pin)
        with self.assertRaisesRegex(ContractError, "independent_receipt_sha256_required"):
            verify_receipt(self.root,path,expected_sha256=None)
        path,pin = self.make()
        with (self.root/path).open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaisesRegex(ContractError, "receipt_raw_sha256_mismatch"):
            verify_receipt(self.root,path,expected_sha256=pin)

    def test_validation_drops_no_untrusted_fields_silently(self):
        for change in (
            lambda d:d.update(transcript="sensitive fake row"),
            lambda d:d.update(output_rows=["sensitive fake row"]),
            lambda d:d.update(model_payload="binary encoding"),
            lambda d:d.update(path="receipts/private.json"),
            lambda d:d.update(scope="holdout"),
            lambda d:d.update(status="approved"),
            lambda d:d.update(schema_version=True),
            lambda d:d["counts"].update(file_name=1),
            lambda d:d["metrics"].update(chrf=True),
            lambda d:d["metrics"].update(chrf=float("nan")),
            lambda d:d["metrics"].update(chrf=float("inf")),
            lambda d:d["metrics"].update(chrf=101),
            lambda d:d["metrics"].update(chrf="raw string"),
            lambda d:d["counts"].update(reviewed_rows=99),
        ):
            record = public_eval()
            change(record)
            with self.subTest(change=change), self.assertRaises(ContractError) as ex:
                validate_receipt(record)
            self.assertNotIn("sensitive",str(ex.exception))

    def test_duplicate_or_nonfinite_json_and_size_fail_closed(self):
        for raw, reason in (
            (b'{"status":"first","status":"second"}', "duplicate_receipt_json_key"),
            (b'{"score": Infinity}', "nonfinite_receipt_number"),
            (b'{"score": 1e999}', "nonfinite_receipt_number"),
            (b'[]',"receipt_json_object_required"),
            (b' '*MAX_JSON_BYTES+b'\n',"invalid_receipt_size"),
        ):
            with self.subTest(reason=reason):
                path,pin = self.make(raw=raw)
                with self.assertRaisesRegex(ContractError, reason):
                    read_pinned_json(self.root,path,expected_sha256=pin)

    def test_report_is_detached_and_contains_no_locator(self):
        receipt = self.verified("metadata.json",public_eval())
        released = receipt.to_public()
        released["counts"]["paired_rows"] = 0
        self.assertEqual(receipt.to_public()["counts"]["paired_rows"],16)
        self.assertEqual(receipt.to_public(),public_eval())
        self.assertNotIn("metadata.json",repr(receipt))
        with self.assertRaises(TypeError):
            receipt._public["counts"]["paired_rows"] = 0

    def test_noncanonical_paths_rejected(self):
        _,pin = self.make()
        for path in ("/etc/fake.json","../receipt.json","receipts/../receipt.json",
                     "receipts//receipt.json","receipts/./receipt.json","receipts\\receipt.json",
                     "C:/receipt.json","file://untrusted/receipt.json",
                     "receipts/CON.json","receipts/blob.safetensors","NUL/metadata.json"):
            with self.subTest(path=path), self.assertRaisesRegex(ContractError,"invalid_receipt_path"):
                verify_receipt(self.root,path,expected_sha256=pin)

    @unittest.skipUnless(os.name == "posix", "POSIX symlink/FIFO test")
    def test_symlinks_and_nonregular_files_rejected(self):
        path,pin = self.make()
        (self.root/"linked").symlink_to(self.root/"receipts", target_is_directory=True)
        (self.root/"receipts/link.json").symlink_to(self.root/path)
        os.mkfifo(self.root/"receipts/nonregular.json")
        for relative in ("linked/receipt.json","receipts/link.json","receipts/nonregular.json"):
            with self.subTest(relative=relative), self.assertRaises(ContractError):
                verify_receipt(self.root,relative,expected_sha256=pin)
        with self.assertRaises(ContractError):
            verify_receipt(self.root/"linked","receipt.json",expected_sha256=pin)

    def test_infrastructure_compares_checkpoint_and_output_identity_only(self):
        a = self.verified("baseline.json",public_eval())
        cfg = public_eval()
        cfg["metrics"]["chrf"] = 99.9
        b = self.verified("candidate.json",cfg)
        actual = compare_receipts(a,b)
        self.assertEqual(actual["status"],"reported_exact_match")
        self.assertEqual(actual["checks"]["reported_checkpoint_parity"],"equal")
        self.assertEqual(actual["checks"]["reported_output_parity"],"equal")
        self.assertFalse(actual["promotion_eligible"])
        self.assertEqual(actual["performance_gate"],"not_evaluated")
        failed = self.verified("failed.json",{**cfg,"status":"fail"})
        failed_report = compare_receipts(a,failed)
        self.assertEqual(failed_report["status"],"fail")
        self.assertIn("source_evaluation_reported_failure",failed_report["reason_codes"])
        cfg["output_sha256"] = "9"*64
        self.assertEqual(compare_receipts(a,self.verified("candidate.json",cfg))["status"],"fail")
        cfg = public_eval()
        cfg["checkpoint_sha256"] = None
        self.assertEqual(compare_receipts(a,self.verified("candidate.json",cfg))["status"],"not_verified")

    def test_comparison_binds_direction_fixture_method_counts_and_completed_stage(self):
        a = self.verified("baseline.json",public_eval())
        for updates in ({"direction":"fr->en"},{"fixture_sha256":"a"*64},
                        {"method_sha256":"b"*64},{"counts":{"paired_rows":12}},
                        {"stage":"train"},{"status":"not_verified"}):
            with self.subTest(updates=updates):
                b = self.verified("candidate.json",{**public_eval(),**updates})
                result = compare_receipts(a,b)
                self.assertEqual(result["status"],"not_comparable")
                self.assertFalse(result["promotion_eligible"])
        with self.assertRaises(ContractError):
            compare_receipts(public_eval(),public_eval())

    def test_model_dev_chrF_diagnostic_is_always_nonpromoting(self):
        a = self.verified("baseline.json",public_eval())
        for new,verdict in ((73,"within_half_point"),(72.99999,"regressed")):
            cfg = public_eval()
            cfg["metrics"]["chrf"] = new
            cfg["output_sha256"] = "a"*64
            result = compare_receipts(a,self.verified("candidate.json",cfg),lane="model_dev")
            self.assertEqual(result["checks"]["chrf_dev_triage"],verdict)
            self.assertEqual(result["status"],"not_verified")
            self.assertTrue(result["external_user_approval_required"])
            self.assertFalse(result["promotion_eligible"])

    def test_checkpoint_returns_only_digest_and_declared_binding(self):
        blob = b"arbitrary synthetic bytes; this is not a loadable weight format"
        path = self.root/"receipts/demo.safetensors"
        path.write_bytes(blob)
        spec = {
            "direction":"es->en","architecture":"marian","base_revision":"a"*40,
            "tokenizer_sha256":"b"*64,"checkpoint_path":"receipts/demo.safetensors",
            "checkpoint_sha256":sha(blob),
        }
        result = verify_checkpoint(self.root,spec)
        self.assertEqual(result["status"],"verified_pinned_bytes")
        self.assertEqual(result["load_compatibility"],"not_verified")
        self.assertEqual(result["checkpoint_sha256"],sha(blob))
        self.assertNotIn("receipts/demo",repr(result))
        self.assertNotIn("arbitrary synthetic bytes",repr(result))
        self.assertFalse(result["promotion_eligible"])
        with self.assertRaises(ContractError):
            verify_checkpoint(self.root,{**spec,"checkpoint_path":"../demo.safetensors"})
        with self.assertRaises(ContractError):
            verify_checkpoint(self.root,{**spec,"checkpoint_path":"receipts/not_allowed.zip"})
        path.write_bytes(blob+b"edited")
        with self.assertRaisesRegex(ContractError,"checkpoint_raw_sha256_mismatch"):
            verify_checkpoint(self.root,spec)


if __name__ == "__main__":
    unittest.main()
