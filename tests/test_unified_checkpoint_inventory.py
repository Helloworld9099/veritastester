"""M0 identity regression checks, offline and without base model dependencies."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from research.unified_models.checkpoint_inventory import read_inventory, validate_inventory, verify_checkout


class CheckpointInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = read_inventory()

    def test_preserves_all_m0_identities_and_paths_at_pinned_commits(self):
        # Frozen from the independent M0 Git object byte audit, not from the new catalog.
        c = self.catalog
        records = [
            (oid, row[0], row[1], [
                (alias[0], sorted(c["refs"][name] for name in c["refsets"][alias[1]]))
                for alias in sorted(row[3])
            ])
            for oid, row in sorted(c["artifacts"].items())
        ]
        digest = hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(digest, "d6bbf3d7672eecf31017245511b360de470b5058065789db08720858455bfd7f")
        self.assertEqual(len(records), 102)
        self.assertEqual(sum(len(a[3]) for a in records), 121)
        self.assertEqual(len(c["refs"]), 11)

    def test_historical_comparator_and_nc_records_are_not_ancestry_or_load_permissions(self):
        c = self.catalog
        es = c["runs"]["research/neural_continuation_20261116/spanish/report.json"]
        fr = c["runs"]["research/neural_continuation_20261116/french/report.json"]
        self.assertIsNone(es["parent_sha256"])
        self.assertEqual(es["comparator_sha256_not_ancestor"], "31c33afbfc7f3e5e1e58b361c099237578f214ffd6cce5dd9781c9fe0ff3dd0a")
        self.assertEqual(fr["parent_sha256"], "7445651cc93617b3c8e4cd00c614371ee7af92ccf541659a9b445f8d9edf372f")
        self.assertEqual(sum(row[2] == "research-archive-noncommercial" for row in c["artifacts"].values()), 19)
        self.assertFalse(c["bases"]["en-es"]["adapter_payloads_in_catalog"])
        bad = copy.deepcopy(c)
        row = next(row for row in bad["artifacts"].values() if row[2] == "research-archive-noncommercial")
        row[2] = "pinned-registry-reference-active-adapter"
        with self.assertRaisesRegex(ValueError, "noncommercial archive misclassified"):
            validate_inventory(bad)

    def test_ref_families_cannot_assign_the_same_path_two_identities_or_escape_directory(self):
        c = copy.deepcopy(self.catalog)
        a = next(iter(c["artifacts"].values()))
        a[3][0][0] = "research/../outside.safetensors"
        with self.assertRaisesRegex(ValueError, "invalid path"):
            validate_inventory(c)
        c = copy.deepcopy(self.catalog)
        a, b = list(c["artifacts"].values())[:2]
        b[3][0] = copy.deepcopy(a[3][0])
        b[1] = a[1]
        with self.assertRaisesRegex(ValueError, "multiple blobs"):
            validate_inventory(c)

    def test_local_check_distinguishes_missing_exact_and_same_size_tampered_bytes(self):
        payload = b"sample bytes (not a model)"
        size = len(payload)
        oid = hashlib.sha1(f"blob {size}\0".encode() + payload, usedforsecurity=False).hexdigest()
        digest = hashlib.sha256(payload).hexdigest()
        catalog = {
            "refs": {"main": "example"},
            "architecture": {"bytes_each": size},
            "refsets": {"S1": ["main"]},
            "artifacts": {oid: [digest, "es-en", "catalog-research-only", [
                ["research/sample.safetensors", "S1", "candidate-step", "research/report.json"],
            ]]},
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "research/sample.safetensors"
            p.parent.mkdir()
            self.assertEqual(verify_checkout(catalog, td, "main")["missing_paths"], ["research/sample.safetensors"])
            p.write_bytes(payload)
            self.assertEqual(verify_checkout(catalog, td, "main")["verified_paths"], 1)
            p.write_bytes(payload[:-1] + b"!")
            self.assertEqual(verify_checkout(catalog, td, "main")["mismatched_paths"], ["research/sample.safetensors"])


if __name__ == "__main__":
    unittest.main()
