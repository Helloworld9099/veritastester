"""No false quality promotion; stop dead-horse training on recorded failures."""
import unittest

from research.madlad400.optimization_controller import (
    analyze_real_prior_results,compare,prioritize_next,
    REQUIRED_CURRENT_DIRECTIONS,EXPANSION_SENTINELS
)


def measurements(chrf=55,bleu=25,n=36,critical=0,latency=1.4,ram=4200):
    return {route:{
        "n":n,"chrF_not_accuracy":chrf,
        "BLEU_not_accuracy":bleu,
        "critical_errors":critical,
        "latency_p95_seconds":latency,
        "ram_mib":ram}
        for route in REQUIRED_CURRENT_DIRECTIONS+EXPANSION_SENTINELS}


class TestMeasuredTrainingDecisions(unittest.TestCase):
    def test_previous_72_real_training_steps_rejected_no_reset(self):
        receipt=analyze_real_prior_results()
        self.assertEqual(receipt["status"],"analyzed_actual_historical_training_failure")
        self.assertEqual(receipt["actual_previous_optimizer_updates"],72)
        self.assertEqual(receipt["historical_accepted_candidates"],[])
        self.assertEqual(len(receipt["historical_rejected_and_why"]),3)
        self.assertTrue(receipt["does_not_claim_new_training"])

    def test_missing_expansion_sentinels_or_one_case_blocks_promotion(self):
        parent=measurements()
        improved=measurements(chrf=57,bleu=26,n=1)
        report=compare(parent,improved)
        self.assertFalse(report["research_checkpoint_gate_passed"])
        self.assertIn("es-en:too_few_development_examples",report["blockers"])
        self.assertFalse(report["production_approval"])
        self.assertFalse(report["human_95_meaning_accuracy_verified"])
        improved=measurements(chrf=57,bleu=26)
        del improved["en-zh"]
        report=compare(parent,improved)
        self.assertFalse(report["research_checkpoint_gate_passed"])
        self.assertIn("missing_six_original_or_four_expansion_languages",report["blockers"])

    def test_critical_safety_latency_and_language_regression_blocks(self):
        parent=measurements()
        improved=measurements(chrf=57,bleu=27)
        improved["fr-es"]["critical_errors"]=1
        improved["en-ar"]["latency_p95_seconds"]=5.0
        improved["fr-en"]["BLEU_not_accuracy"]=23
        report=compare(parent,improved)
        self.assertFalse(report["research_checkpoint_gate_passed"])
        self.assertIn("fr-es:critical_error_regression",report["blockers"])
        self.assertIn("en-ar:too_slow_for_live_captioning",report["blockers"])
        self.assertIn("fr-en:BLEU_regression",report["blockers"])

    def test_safe_automatic_gate_still_not_human_approval(self):
        report=compare(measurements(),measurements(chrf=56,bleu=26))
        self.assertTrue(report["research_checkpoint_gate_passed"])
        self.assertFalse(report["production_approval"])
        self.assertFalse(report["human_95_meaning_accuracy_verified"])

    def test_next_is_baseline_then_int8_then_data_then_new_method(self):
        call=lambda **kw:prioritize_next(remaining_minutes=45,
                                         candidate_results=[],**kw)
        self.assertEqual(call(baseline_exists=False,
                              live_latency_p95_s=None,
                              dataset_hard_cases_verified=False)["next"],
                         "measure_pretrained_MADLAD_on_development")
        slow=call(baseline_exists=True,live_latency_p95_s=7,
                  dataset_hard_cases_verified=False)
        self.assertEqual(slow["next"],"int8_ctranslate2_pretrained_inference")
        self.assertFalse(slow["allow_new_training"])
        self.assertEqual(call(baseline_exists=True,live_latency_p95_s=1.1,
                              dataset_hard_cases_verified=False)["next"],
                         "mine_and_bilingually_verify_context_slang_errors")
        changed=prioritize_next(baseline_exists=True,
                     live_latency_p95_s=0.8,
                     candidate_results=[{"method":"continue_same_schedule","status":"rejected"}],
                     remaining_minutes=45,dataset_hard_cases_verified=True)
        self.assertEqual(changed["next"],"last_decoder_lora_vs_encoder_decoder_lora")
        self.assertNotEqual(changed["next"],"continue_same_schedule")
        self.assertTrue(changed["allow_new_training"])

    def test_exhausted_money_or_wall_clock_stops(self):
        report=prioritize_next(
            baseline_exists=True,live_latency_p95_s=1.4,
            candidate_results=[],remaining_minutes=0,
            dataset_hard_cases_verified=True)
        self.assertEqual(report["next"],"stop_budget_exhausted")
        self.assertFalse(report["allow_new_training"])
        self.assertEqual(report["paid_spend_usd"],0)


if __name__=="__main__":
    unittest.main()
