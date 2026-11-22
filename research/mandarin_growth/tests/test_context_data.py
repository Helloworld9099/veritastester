import unittest
from research.mandarin_growth.context_data import validate_record


class ContextDataTests(unittest.TestCase):
    def setUp(self):
        self.record={"source_language":"en","target_language":"zh",
                     "source":"That's sick!","target":"太酷了！",
                     "preceding_turns":["I just saw your new skateboard trick."],
                     "domain":"informal speech","phenomenon":"slang",
                     "license":"CC0","provenance":"human-created test fixture",
                     "reviewer_verified":True}

    def test_valid_record(self):
        self.assertTrue(validate_record(self.record))

    def test_unverified_context_cannot_train(self):
        self.record["reviewer_verified"]=False
        with self.assertRaises(ValueError):
            validate_record(self.record)

    def test_missing_context_cannot_train(self):
        self.record["preceding_turns"]=[]
        with self.assertRaises(ValueError):
            validate_record(self.record)

    def test_noncommercial_rejected(self):
        self.record["license"]="CC-BY-NC"
        with self.assertRaises(ValueError):
            validate_record(self.record)
