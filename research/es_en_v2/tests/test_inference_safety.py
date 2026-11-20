import unittest

from research.es_en_v2.inference_safety import (
    RepetitionLoopError, has_generation_loop, retry_if_loop
)
from research.es_en_v2.review_feedback_500 import load_review


class TestBoundedLoopRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_review()

    def test_catches_all_three_actual_long_repetition_failures(self):
        for excel_row in (311, 314, 416):
            with self.subTest(excel_row=excel_row):
                actual = self.rows[excel_row - 2]["Actual model translation"]
                self.assertTrue(has_generation_loop(actual))

    def test_does_not_trigger_for_short_legitimate_repetitions(self):
        for text in ("No, no, no!", "Sí, sí, claro.", "Up, up!", "I am going today.",
                     "Estoy abajo.", "Yummy! The food is yummy."):
            self.assertFalse(has_generation_loop(text))

    def test_no_redecode_on_normal_sentence(self):
        calls = []
        def retry():
            calls.append("bad")
            return "changed"
        result = retry_if_loop("How can I help you?", retry)
        self.assertEqual(result.text, "How can I help you?")
        self.assertEqual(result.attempts, 1)
        self.assertFalse(result.rescued)
        self.assertEqual(calls, [])

    def test_constrained_retry_is_returned_as_actual_new_output(self):
        calls = []
        def retry():
            calls.append(True)
            return "Green peas! Delicious!"
        result = retry_if_loop("Yummy, " * 32, retry)
        self.assertEqual(result.text, "Green peas! Delicious!")
        self.assertEqual(result.attempts, 2)
        self.assertTrue(result.was_loop)
        self.assertTrue(result.rescued)
        self.assertEqual(len(calls), 1)

    def test_fails_closed_if_second_attempt_still_loops(self):
        with self.assertRaises(RepetitionLoopError):
            retry_if_loop("Up, " * 40, lambda: "Up, " * 20)

    def test_returns_no_fabricated_translation_for_failed_recovery(self):
        with self.assertRaises(RepetitionLoopError):
            retry_if_loop("I want to go! " * 12, lambda: "")
        with self.assertRaises(ValueError):
            retry_if_loop("", lambda: "Hello")


if __name__ == "__main__":
    unittest.main()
