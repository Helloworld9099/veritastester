"""Synthetic M6 metadata-only validation; no model files, data, or backend."""
from copy import deepcopy
import unittest

from research.unified_training.shared_plan import (
    SharedPlanError, metadata_sha256, prepare_shared_plan,
)

ES, FR = "es->en", "fr->en"
CLEAR = "reviewed_commercial_training_permitted"


def digest(number):
    return f"{number:064x}"


def corpus(direction, num, objective="joint_speech"):
    paired = objective == "joint_speech"
    return {
        "schema_version": 1, "kind": "shared_corpus_rights_review",
        "direction":direction, "split":"train", "sealed":False,
        "visibility":"public_train",
        "material_type":"paired_speech" if paired else "parallel_text",
        "covered_modalities":["audio","transcript","translation"] if paired else [
            "source_text","translation"],
        "corpus_manifest_sha256":digest(num), "membership_sha256":digest(num + 1000),
        "license":"CC-BY-4.0", "lineage_sha256":digest(90), "lineage_status":CLEAR,
        "rights_evidence_sha256":digest(91),"training_rights_confirmed":True,
    }


def teacher(original, *, seed=3):
    return {
        "schema_version":1, "kind":"shared_teacher_output_rights_review",
        "direction":original["direction"], "split":"train", "sealed":False,
        "visibility":"approved_local_derivative",
        "output_kind":"text_translation_targets",
        "source_corpus_review_sha256":metadata_sha256(original),
        "source_membership_sha256":original["membership_sha256"],
        "teacher_checkpoint_sha256":digest(450 + seed),"teacher_outputs_sha256":digest(470 + seed),
        "model_license":"Apache-2.0","output_license":"CC-BY-4.0",
        "lineage_status":CLEAR,"lineage_sha256":digest(92),"rights_evidence_sha256":digest(93),
        "inference_rights_confirmed":True,"output_reuse_for_training_confirmed":True,
    }


class SharedSpeechPlanTest(unittest.TestCase):
    def setUp(self):
        self.student = {
            "architecture":"single_streaming_speech_model", "base_revision":"a"*40,
            "checkpoint_sha256":digest(1), "tokenizer_sha256":digest(2),
            "audio_processor_sha256":digest(3), "directions":[ES,FR],
            "license":"Apache-2.0","lineage_sha256":digest(94),"lineage_status":CLEAR,
            "rights_evidence_sha256":digest(95),"training_rights_confirmed":True,
        }
        self.base_pin = metadata_sha256(self.student)
        self.plan = {"schema_version":1,"kind":"shared_speech_training_preparation",
                     "scope":"research_only","strategy":"joint_training",
                     "student":self.student,"directions":[ES,FR],"materials":[]}
        self.reviews = {}
        self.add(corpus(ES,10))
        self.add(corpus(FR,11))

    def add(self, review, objective="joint_speech", teacher_review=None):
        pin = metadata_sha256(review)
        self.reviews[pin] = review
        teacher_pin = None
        if teacher_review is not None:
            teacher_pin = metadata_sha256(teacher_review)
            self.reviews[teacher_pin] = teacher_review
        self.plan["materials"].append({
            "student_base_sha256":self.base_pin, "direction":review["direction"],
            "objective":objective,"corpus_review_sha256":pin,"teacher_review_sha256":teacher_pin,
        })

    def run_plan(self, **kwargs):
        supplied = {"expected_plan_sha256":metadata_sha256(self.plan),
                    "expected_base_sha256":self.base_pin, "receipts":self.reviews,
                    "trusted_receipt_sha256":set(self.reviews), **kwargs}
        return prepare_shared_plan(self.plan, **supplied)

    def replace_first_review(self, *, teacher_review=None, **updates):
        row = self.plan["materials"].pop(0)
        old = self.reviews.pop(row["corpus_review_sha256"])
        review = {**old, **updates}
        self.add(review, teacher_review=teacher_review)
        return review

    def test_joint_requires_one_common_student_and_is_diagnostic_only(self):
        result = self.run_plan()
        self.assertEqual(result["status"],"prepared_metadata_only")
        self.assertEqual(result["student_base_sha256"],self.base_pin)
        self.assertEqual(result["directions"][ES]["paired_speech_materials"],1)
        for key in ("training_authorized","production_activated","paired_record_overlap_verified",
                    "independent_evaluation_performed"):
            self.assertIs(result[key],False)
        for key in ("corpus_rows_read","model_files_read","teacher_calls","optimizer_updates"):
            self.assertEqual(result[key],0)
        self.assertNotIn("corpus_manifest_sha256",repr(result))
        self.assertNotIn(digest(10),repr(result))

    def test_exact_pins_separate_from_manifests(self):
        stale = metadata_sha256(self.plan)
        self.plan["strategy"] = "distillation"
        with self.assertRaisesRegex(SharedPlanError,"plan_identity_mismatch"):
            self.run_plan(expected_plan_sha256=stale)
        self.plan["strategy"] = "joint_training"
        with self.assertRaisesRegex(SharedPlanError,"student_base_identity_mismatch"):
            self.run_plan(expected_base_sha256=digest(100))
        pin = self.plan["materials"][0]["corpus_review_sha256"]
        altered = {**self.reviews, pin:{**self.reviews[pin],"license":"MIT"}}
        with self.assertRaisesRegex(SharedPlanError,"review_identity_mismatch"):
            self.run_plan(receipts=altered)
        with self.assertRaisesRegex(SharedPlanError,"review_requires_independent_pin"):
            self.run_plan(trusted_receipt_sha256=[])

    def test_each_arm_matches_same_base_and_explicit_routes(self):
        self.plan["materials"][0]["student_base_sha256"] = digest(876)
        with self.assertRaisesRegex(SharedPlanError,"student_mismatch"):
            self.run_plan()
        self.plan["materials"][0]["student_base_sha256"] = self.base_pin
        self.plan["materials"][0]["direction"] = FR
        with self.assertRaisesRegex(SharedPlanError,"matching_direction"):
            self.run_plan()
        self.plan["materials"][0]["direction"] = ES
        self.plan["directions"].remove(FR)
        with self.assertRaisesRegex(SharedPlanError,"identical_direction_set"):
            self.run_plan()

    def test_input_only_train_open_public_rights(self):
        for changes, code in [
            ({"split":"test"},"reviewed_train"), ({"split":"holdout"},"reviewed_train"),
            ({"split":"dev"},"reviewed_train"), ({"sealed":True},"reviewed_train"),
            ({"visibility":"private"},"reviewed_train"),
            ({"license":"CC-BY-NC-4.0"},"rights_not_cleared"),
            ({"license":"UNKNOWN"},"rights_not_cleared"),
            ({"lineage_status":"unknown"},"rights_not_cleared"),
            ({"training_rights_confirmed":True,"rights_evidence_sha256":None},"rights_not_cleared"),
            ({"covered_modalities":["audio","translation"]},"modality"),
        ]:
            with self.subTest(changes=changes):
                self.setUp()
                self.replace_first_review(**changes)
                with self.assertRaisesRegex(SharedPlanError,code):
                    self.run_plan()

    def test_student_restricted_lineage_blocked_even_when_pin_is_valid(self):
        self.student["lineage_status"] = "unknown"
        self.base_pin = metadata_sha256(self.student)
        for row in self.plan["materials"]:
            row["student_base_sha256"] = self.base_pin
        with self.assertRaisesRegex(SharedPlanError,"student_rights_not_cleared"):
            self.run_plan()

    def test_text_cannot_satisfy_audio_objective(self):
        self.plan["materials"].pop(1)
        self.add(corpus(FR,101,"text_auxiliary"),objective="text_auxiliary")
        with self.assertRaisesRegex(SharedPlanError,"paired_audio_"):
            self.run_plan()

    def test_rights_reviewed_teacher_signals_are_text_only_and_optional(self):
        self.plan["strategy"] = "distillation"
        with self.assertRaisesRegex(SharedPlanError,"no_reviewed_teacher"):
            self.run_plan()
        text = corpus(ES,22,"text_auxiliary")
        target = teacher(text)
        self.add(text,objective="text_auxiliary",teacher_review=target)
        result = self.run_plan()
        self.assertEqual(result["directions"][ES]["reviewed_teacher_outputs"],1)
        self.assertEqual(result["directions"][ES]["text_auxiliary_materials"],1)
        self.assertEqual(result["directions"][FR]["reviewed_teacher_outputs"],0)
        self.assertEqual(result["teacher_signal_scope"],"text_translation_targets_only")
        self.assertNotIn(target["teacher_outputs_sha256"],repr(result))
        self.plan["strategy"] = "joint_training"
        with self.assertRaisesRegex(SharedPlanError,"explicit_distillation"):
            self.run_plan()

    def test_teacher_model_terms_dont_grant_output_reuse_and_lineage_must_be_clean(self):
        for changes, code in [
            ({"output_reuse_for_training_confirmed":False}, "rights_not_cleared"),
            ({"inference_rights_confirmed":False}, "rights_not_cleared"),
            ({"model_license":"CC-BY-NC-4.0"}, "rights_not_cleared"),
            ({"output_license":"UNKNOWN"}, "rights_not_cleared"),
            ({"lineage_status":"unknown"}, "rights_not_cleared"),
            ({"output_kind":"neural_weight_merge"}, "translation_targets"),
            ({"source_membership_sha256":digest(876)}, "translation_targets"),
            ({"visibility":"private_holdout"}, "translation_targets"),
            ({"split":"test"}, "translation_targets"),
        ]:
            with self.subTest(changes=changes):
                self.setUp()
                self.plan["strategy"] = "distillation"
                extra = corpus(FR,310,"text_auxiliary")
                self.add(extra,objective="text_auxiliary",teacher_review={**teacher(extra),**changes})
                with self.assertRaisesRegex(SharedPlanError,code):
                    self.run_plan()

    def test_policy_fails_unknown_payloads_and_reused_manifest(self):
        self.plan["tokenizer_merge"] = True
        with self.assertRaisesRegex(SharedPlanError,"invalid_shared_plan_fields"):
            self.run_plan()
        self.plan.pop("tokenizer_merge")
        row = deepcopy(self.plan["materials"][0])
        self.plan["materials"].append(row)
        with self.assertRaisesRegex(SharedPlanError,"membership_or_manifest_reused"):
            self.run_plan()


if __name__ == "__main__":
    unittest.main()
