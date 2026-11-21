"""Unit tests for the offline paired gates, no models/corpora needed."""
from copy import deepcopy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from research.unified_eval.gates import evaluate_direction, fingerprint

DIR = "es->en"
FIX = "a" * 64
MIX = "2" * 64
PIN = {
    "base_revision": "0"*40,
    "files": {"config.json":"1"*64,"source.spm":"3"*64,"target.spm":"4"*64,
              "vocab.json":"5"*64,"model.safetensors":"6"*64},
    "adapter_sha256":"7"*64, "rollback_sha256":"8"*64,
}


class GatesTest(unittest.TestCase):
    def setUp(self):
        self.baseline = {
            "direction": DIR, "scope":"development","fixture_sha256":FIX,
            "checkpoint":deepcopy(PIN), "outputs":{"test-a":"Réponse.","test-b":"Preserved!"},
            "performance":{"input_mix_sha256":MIX},
        }
        self.candidate = deepcopy(self.baseline)
        self.trust = {"baseline_pins":deepcopy(PIN),"candidate_pins":deepcopy(PIN),
                      "expected_fixture_sha256":FIX,"expected_input_mix_sha256":MIX,"expected_warm_requests":40,
                      "expected_baseline_outputs_sha256":fingerprint(self.baseline["outputs"])}
        self.perf = Mock(return_value={"status":"pass","metrics":{}, "reasons":[]})
        self.labels = {k:{"baseline":[],"candidate":[]} for k in self.baseline["outputs"]}

    def run_gate(self, lane="infrastructure", **kwargs):
        with patch.dict(sys.modules, {"research.unified_eval.performance":
                                      SimpleNamespace(compare_performance=self.perf)}):
            return evaluate_direction(self.baseline, self.candidate, lane=lane,
                                      **{**self.trust, **kwargs})

    def scores(self, *, before=70.0, after=70.0, group_before=70.0, group_after=70.0):
        return {
            "direction": DIR,"n":2,
            "hashes":{"case_spec": "0"*64,
                      "baseline_outputs":fingerprint(self.baseline["outputs"]),
                      "candidate_outputs":fingerprint(self.candidate["outputs"])},
            "baseline":{"chrf":before},"candidate":{"chrf":after},
            "rows":[{"id":"test-a","groups":["slang"]},
                    {"id":"test-b","groups":["numbers"]}],
            "subgroups":{
                "slang":{"n":1,"baseline":{"chrf":group_before},"candidate":{"chrf":group_after}},
                "numbers":{"n":1,"baseline":{"chrf":70},"candidate":{"chrf":70}},
            },
        }

    def test_infrastructure_exact_even_when_chrF_matches(self):
        result = self.run_gate()
        self.assertEqual(result["status"],"pass")
        self.assertEqual(result["checks"]["checkpoint_parity"],"pass")
        self.assertEqual(result["summary"]["changed_output_rows"],0)
        self.assertFalse(result["promotion_eligible"])
        self.assertTrue(result["external_user_approval_required"])
        self.assertEqual(result["independent_human_evaluation"],"not_evaluated")
        self.candidate["outputs"]["test-a"] = "Re\u0301ponse."
        after = self.run_gate(score=self.scores())
        self.assertIn("exact_output_bytes_changed",after["failures"])
        self.assertEqual(after["summary"]["changed_output_rows"],1)
        self.assertNotIn("Réponse.", str(after))

    def test_checkpoint_pin_and_rollback_are_exact_for_infrastructure(self):
        loose = deepcopy(PIN)
        loose["rollback_sha256"] = "f"*64
        self.candidate["checkpoint"] = loose
        self.assertIn("receipt_checkpoint_disagrees_with_trusted_pin", self.run_gate()["failures"])
        self.assertIn("checkpoint_identity_changed",
                      self.run_gate(candidate_pins=loose)["failures"])
        self.assertEqual(self.run_gate(baseline_pins=None)["status"],"blocked")

    def test_unrelated_scope_rows_directions_and_untrusted_baseline_block(self):
        mismatches = [
            lambda: self.candidate.update(direction="fr->en"),
            lambda: self.candidate.update(scope="final_holdout"),
            lambda: self.candidate.update(outputs={"missing":"changed"}),
            lambda: self.candidate.update(outputs={"test-a":" ","test-b":"text"}),
            lambda: self.baseline["outputs"].update({"test-a":"rewritten baseline"}),
            lambda: self.candidate.update(fixture_sha256="f"*64),
        ]
        before,after = deepcopy(self.baseline),deepcopy(self.candidate)
        for edit in mismatches:
            self.baseline,self.candidate = deepcopy(before),deepcopy(after)
            edit()
            with self.subTest(edit=edit):
                self.assertEqual(self.run_gate()["status"],"blocked")
        self.assertFalse(self.perf.called)

    def test_model_dev_direction_half_point_is_inclusive_but_not_a_promotion(self):
        changed_pin = deepcopy(PIN)
        changed_pin["adapter_sha256"] = "9"*64
        self.candidate["checkpoint"] = changed_pin
        self.candidate["outputs"]["test-a"] = "Réponse modifiée."
        result = self.run_gate("model_dev", candidate_pins=changed_pin, critical=self.labels,
                               score=self.scores(after=69.5))
        self.assertEqual(result["status"],"pass")
        self.assertFalse(result["promotion_eligible"])
        self.assertEqual(result["summary"]["chrf_delta"],-0.5)
        self.assertEqual(self.perf.call_args.kwargs["change_kind"],"model")
        bad = self.run_gate("model_dev", candidate_pins=changed_pin, critical=self.labels,
                            score=self.scores(after=69.499))
        self.assertIn("dev_chrF_drop_exceeds_half_point",bad["failures"])

    def test_missing_subgroup_evidence_does_not_mask_known_direction_failure(self):
        score = self.scores(after=65.0)
        score["subgroups"] = {}
        bad = self.run_gate("model_dev", critical=self.labels, score=score)
        self.assertEqual(bad["checks"]["direction_chrf"], "fail")
        self.assertEqual(bad["checks"]["subgroup_chrf"], "blocked")
        self.assertEqual(bad["status"], "fail")

    def test_subgroup_regression_cannot_be_concealed_by_aggregate(self):
        score = self.scores(after=72, group_after=69.99)
        bad = self.run_gate("model_dev", critical=self.labels, score=score)
        self.assertEqual(bad["summary"]["regressing_subgroups"],1)
        self.assertIn("development_subgroup_regressed",bad["failures"])
        score["subgroups"].pop("slang")
        self.assertIn("complete_paired_subgroups_required",
                      self.run_gate("model_dev", critical=self.labels, score=score)["blockers"])

    def test_critical_labels_are_paired_and_never_guessed(self):
        self.labels["test-a"]["baseline"] = ["negation:q01"]
        self.labels["test-a"]["candidate"] = ["negation:q01"]
        stable = self.run_gate("model_dev", critical=self.labels, score=self.scores())
        self.assertEqual(stable["checks"]["critical_errors"],"pass")
        self.labels["test-b"]["candidate"] = ["quantity:q02"]
        new = self.run_gate("model_dev", critical=self.labels, score=self.scores())
        self.assertEqual(new["summary"]["new_confirmed_critical_errors"],1)
        self.assertIn("new_confirmed_critical_meaning_error", new["failures"])
        self.labels["test-b"]["baseline"] = ["quantity:q03"]
        # Same category is not enough when it is a different critical defect.
        changed_issue = self.run_gate("model_dev", critical=self.labels, score=self.scores())
        self.assertEqual(changed_issue["summary"]["new_confirmed_critical_errors"],1)
        self.labels["test-b"]["candidate"] = ["quantity"]
        self.assertIn("paired_verified_human_dev_labels_required",
                      self.run_gate("model_dev", critical=self.labels, score=self.scores())["blockers"])
        no_evidence = self.run_gate("model_dev", score=self.scores())
        self.assertEqual(no_evidence["status"],"blocked")
        partial = {"test-a":{"baseline":[],"candidate":[]}}
        self.assertEqual(self.run_gate("model_dev", critical=partial,
                                      score=self.scores())["status"],"blocked")

    def test_stale_or_untrusted_scores_fail_closed(self):
        invalidators = [
            lambda score: score["hashes"].update(candidate_outputs="b"*64),
            lambda score: score["baseline"].update(chrf=float("nan")),
            lambda score: score.update(n=True),
            lambda score: score["subgroups"]["numbers"].update(n=7),
            lambda score: score["rows"].append(deepcopy(score["rows"][0])),
        ]
        for index, mutate in enumerate(invalidators):
            score = self.scores()
            mutate(score)
            result = self.run_gate("model_dev", critical=self.labels, score=score)
            self.assertEqual(result["status"],"blocked")
            self.assertIn("paired_dev_chrF_missing_or_unverified" if index < 3
                          else "complete_paired_subgroups_required", result["blockers"])

    def test_performance_pass_requires_trusted_input_mix_and_numeric_gate(self):
        self.assertEqual(self.run_gate(expected_input_mix_sha256=None)["status"],"blocked")
        self.candidate["performance"]["input_mix_sha256"] = "3"*64
        self.assertEqual(self.run_gate()["status"],"blocked")
        self.assertFalse(self.perf.called)
        self.candidate["performance"]["input_mix_sha256"] = MIX
        self.perf.return_value = {"status":"incomparable","reasons":["entry"],"metrics":{}}
        self.assertEqual(self.run_gate()["status"],"blocked")
        self.assertEqual(self.perf.call_args.kwargs["expected_direction"],DIR)
        self.assertEqual(self.perf.call_args.kwargs["expected_fixture_sha256"],FIX)
        self.perf.return_value = {"status":"fail","reasons":["peak_rss"],"metrics":{}}
        self.assertIn("latency_or_rss_regression",self.run_gate()["failures"])


if __name__ == "__main__":
    unittest.main()
