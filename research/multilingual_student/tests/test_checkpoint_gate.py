"""No-network tests for conservative all-six-direction student checkpoint gate."""
import unittest
from research.multilingual_student.select_checkpoint import gate

ROUTES=("es-en","fr-en","en-es","en-fr","es-fr","fr-es")


def scores(chrf=50.0,bleu=30.0):
    return {lang:{"chrF_not_accuracy":chrf,"BLEU_not_accuracy":bleu}
            for lang in ROUTES}


class TestSixDirectionCheckpointGate(unittest.TestCase):
    def test_no_english_to_french_bleu_regression_allowed(self):
        old=scores()
        new=scores(chrf=51,bleu=31)
        new["en-fr"]["BLEU_not_accuracy"]=28.5
        pass_gate,summary=gate(old,new)
        self.assertFalse(pass_gate)
        self.assertIn("en-fr:BLEU",summary["automatic_gate_blockers"])

    def test_all_six_major_subgroups_must_pass(self):
        old=scores()
        new=scores(chrf=50.8,bleu=30.2)
        self.assertTrue(gate(old,new)[0])
        new["fr-en"]["chrF_not_accuracy"]=49
        self.assertFalse(gate(old,new)[0])

    def test_no_promoting_noise_around_parent(self):
        self.assertFalse(gate(scores(),scores())[0])
        self.assertIn("mean_chrF_gain_less_than_.25",
                      gate(scores(),scores())[1]["automatic_gate_blockers"])
        with self.assertRaises(ValueError):
            gate(scores(),{"es-en":{"chrF_not_accuracy":60,"BLEU_not_accuracy":30}})


if __name__=="__main__":
    unittest.main()
