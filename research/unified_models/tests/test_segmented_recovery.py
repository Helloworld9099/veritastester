import unittest
from research.unified_models.segmented_recovery import (
    separate_current_clauses, translate_clauses
)


class TestKeepAllCurrentClauses(unittest.TestCase):
    def test_exact_reported_spanish_loop_examples_split_without_dropping_words(self):
        examples = [
            ("¡Guisantes verdes! ¡Delicioso!", ("¡Guisantes verdes!", "¡Delicioso!")),
            ("¡Tobogán mojado! ¡Quiero ir! \\n\\n", ("¡Tobogán mojado!", "¡Quiero ir!")),
            ("¡Escalar! ¡Arriba, arriba! \\n\\n", ("¡Escalar!", "¡Arriba, arriba!")),
        ]
        for source, expected in examples:
            with self.subTest(source=source):
                self.assertEqual(separate_current_clauses(source),expected)

    def test_current_utterance_only_no_context_injection_or_unmatched_splits(self):
        self.assertEqual(separate_current_clauses("Bonjour, comment allez-vous ?"),
                         ("Bonjour, comment allez-vous ?",))
        self.assertEqual(separate_current_clauses("What about this? Yes!"),
                         ("What about this?","Yes!"))
        self.assertEqual(separate_current_clauses("No eran quince, eran cincuenta."),
                         ("No eran quince, eran cincuenta.",))

    def test_rejects_overlong_or_blank_clause_inputs(self):
        with self.assertRaises(ValueError):
            separate_current_clauses("")
        with self.assertRaises(ValueError):
            separate_current_clauses("A! B! C! D! E! F! G!")
        with self.assertRaises(ValueError):
            separate_current_clauses("Hi!",max_clauses=0)

    def test_translates_all_parts_in_order_and_rejects_missing_part(self):
        calls=[]
        def translator(clause):
            calls.append(clause)
            return {"¡Guisantes verdes!":"Green peas!",
                    "¡Delicioso!":"Delicious!"}[clause]
        result=translate_clauses(("¡Guisantes verdes!","¡Delicioso!"),translator)
        self.assertEqual(result,"Green peas! Delicious!")
        self.assertEqual(calls,["¡Guisantes verdes!","¡Delicioso!"])
        with self.assertRaises(RuntimeError):
            translate_clauses(("One!", "Two!"), lambda x: "One" if x=="One!" else "")
        with self.assertRaises(ValueError):
            translate_clauses((),translator)


if __name__=="__main__":
    unittest.main()
