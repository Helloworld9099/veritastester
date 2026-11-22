"""Verify no INT8 model can be installed on guessed license, size or revision."""
import unittest
from types import SimpleNamespace
from research.madlad400.quantized_ablation import (
    MODEL_ID,MAX_MODEL_BYTES,safe_metadata,p95
)


def sample_meta(license_name="apache-2.0",sha="a"*40,
                model_bin=3500000000):
    siblings=[
        SimpleNamespace(rfilename="model.bin",size=model_bin),
        SimpleNamespace(rfilename="config.json",size=4500),
        SimpleNamespace(rfilename="sentencepiece.model",size=5200000),
    ]
    return SimpleNamespace(sha=sha,cardData={"license":license_name},
                           siblings=siblings)


class TestEfficientModelAbalations(unittest.TestCase):
    def test_only_approved_pinned_converted_int8_model_can_be_benchmarked(self):
        approved=safe_metadata(sample_meta())
        self.assertEqual(approved["converted_repo"],MODEL_ID)
        self.assertEqual(approved["license"],"apache-2.0")
        self.assertEqual(approved["candidate_model_size_bytes"],3505204500)
        self.assertTrue(approved["conversion_not_verified_exact_against_google_pretrained"])

    def test_bad_conversion_license_hash_or_resource_never_loaded(self):
        for value in ("cc-by-nc-4.0",None,"research-only"):
            with self.subTest(license=value),self.assertRaises(RuntimeError):
                safe_metadata(sample_meta(license_name=value))
        with self.assertRaises(RuntimeError):
            safe_metadata(sample_meta(sha="main"))
        with self.assertRaises(RuntimeError):
            safe_metadata(sample_meta(model_bin=MAX_MODEL_BYTES+1))
        with self.assertRaises(RuntimeError):
            safe_metadata(sample_meta(model_bin=10))

    def test_p95_is_bounded_without_numpy(self):
        self.assertEqual(p95([]),0.0)
        self.assertEqual(p95([0.1]),0.1)
        self.assertEqual(p95([1,2,3,4,5]),5)
        self.assertEqual(p95(list(range(1,101))),95)


if __name__=="__main__":
    unittest.main()
