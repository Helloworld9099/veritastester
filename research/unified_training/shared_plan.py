"""Prepare rights-gated metadata for one future shared streaming speech model.

No files, checkpoints, corpora, teacher outputs or network are accessed. A
caller must obtain trusted plan/base/review pins independently of the supplied
metadata. This checks exact identities and review declarations; it neither
licenses material nor attests to the underlying records or a model.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping
import hashlib
import json
import re


class SharedPlanError(ValueError):
    """An independently pinned, rights-reviewed plan is incomplete or invalid."""


_SHA = re.compile(r"[0-9a-f]{64}\Z")
_REV = re.compile(r"[0-9a-f]{40}\Z")
_DIR = re.compile(r"([a-z]{2,3})->([a-z]{2,3})\Z")
_LICENSES = frozenset({"CC0-1.0", "CC-BY-4.0", "Apache-2.0", "MIT"})
_PLAN = {"schema_version", "kind", "scope", "strategy", "student", "directions", "materials"}
_STUDENT = {"architecture", "base_revision", "checkpoint_sha256", "tokenizer_sha256",
            "audio_processor_sha256", "directions", "license", "lineage_sha256",
            "lineage_status", "rights_evidence_sha256", "training_rights_confirmed"}
_MATERIAL = {"direction", "objective", "student_base_sha256",
             "corpus_review_sha256", "teacher_review_sha256"}
_CORPUS = {"schema_version", "kind", "direction", "split", "sealed", "material_type",
           "corpus_manifest_sha256", "membership_sha256", "license", "lineage_sha256",
           "lineage_status", "rights_evidence_sha256", "training_rights_confirmed",
           "covered_modalities", "visibility"}
_TEACHER = {"schema_version", "kind", "direction", "split", "sealed", "source_corpus_review_sha256",
            "source_membership_sha256", "teacher_checkpoint_sha256", "teacher_outputs_sha256",
            "model_license", "output_license", "lineage_status", "lineage_sha256",
            "rights_evidence_sha256", "inference_rights_confirmed",
            "output_reuse_for_training_confirmed", "output_kind", "visibility"}
_MODES = {"joint_speech": {"audio", "transcript", "translation"},
          "text_auxiliary": {"source_text", "translation"}}
_MODE_TYPE = {"joint_speech": "paired_speech", "text_auxiliary": "parallel_text"}


def _require(ok, reason):
    if not ok:
        # Codes contain no raw metadata values, model locators, or corpus IDs.
        raise SharedPlanError(reason)


def _sha(value):
    return type(value) is str and _SHA.fullmatch(value) is not None


def _direction(value):
    parsed = _DIR.fullmatch(value) if type(value) is str else None
    return bool(parsed and parsed[1] != parsed[2])


def _keys(obj, fields, reason):
    _require(type(obj) is dict and obj.keys() == fields, reason)


def _snapshot(value):
    _require(isinstance(value, Mapping), "metadata_object_required")
    try:
        raw = json.dumps(dict(value), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
        _require(0 < len(raw) <= (1 << 20), "metadata_size_limit")
        return raw, json.loads(raw)
    except SharedPlanError:
        raise
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise SharedPlanError("invalid_metadata") from None


def metadata_sha256(value: Mapping) -> str:
    """Canonical metadata digest. Independently acquire trusted expected pins."""
    return hashlib.sha256(_snapshot(value)[0]).hexdigest()


def _schema(obj, kind):
    return type(obj["schema_version"]) is int and obj["schema_version"] == 1 and obj["kind"] == kind


def _dirs(value):
    _require(type(value) is list and 1 <= len(value) <= 64 and all(_direction(v) for v in value)
             and len(set(value)) == len(value), "invalid_or_duplicate_directions")
    return frozenset(value)


def _license(value):
    return type(value) is str and value in _LICENSES


def _clear_lineage(obj):
    return (obj["lineage_status"] == "reviewed_commercial_training_permitted"
            and _sha(obj["lineage_sha256"]) and _sha(obj["rights_evidence_sha256"]))


def _trusted_review(receipts, trusted, pin):
    _require(_sha(pin) and pin in trusted and pin in receipts, "review_requires_independent_pin")
    raw, review = _snapshot(receipts[pin])
    _require(hashlib.sha256(raw).hexdigest() == pin, "review_identity_mismatch")
    return review


def _review_corpus(review, direction, objective):
    _keys(review, _CORPUS, "invalid_corpus_review_fields")
    _require(_schema(review, "shared_corpus_rights_review")
             and review["direction"] == direction
             and review["split"] == "train" and review["sealed"] is False
             and review["visibility"] == "public_train",
             "corpus_must_be_reviewed_train_for_matching_direction")
    _require(_clear_lineage(review) and _license(review["license"])
             and review["training_rights_confirmed"] is True, "corpus_rights_not_cleared")
    _require(_sha(review["corpus_manifest_sha256"]) and _sha(review["membership_sha256"]),
             "unbound_corpus_manifest_or_membership")
    modalities = review["covered_modalities"]
    _require(review["material_type"] == _MODE_TYPE[objective] and type(modalities) is list
             and all(type(v) is str for v in modalities)
             and len(modalities) == len(_MODES[objective]) and set(modalities) == _MODES[objective],
             "material_modality_does_not_match_objective")


def _review_teacher(review, *, direction, corpus_pin, membership):
    _keys(review, _TEACHER, "invalid_teacher_review_fields")
    _require(_schema(review, "shared_teacher_output_rights_review")
             and review["direction"] == direction and review["split"] == "train"
             and review["sealed"] is False and review["source_corpus_review_sha256"] == corpus_pin
             and review["source_membership_sha256"] == membership
             and review["visibility"] == "approved_local_derivative"
             and review["output_kind"] == "text_translation_targets",
             "teacher_output_requires_matching_reviewed_train_translation_targets")
    _require(_clear_lineage(review) and _license(review["model_license"])
             and _license(review["output_license"]) and review["inference_rights_confirmed"] is True
             and review["output_reuse_for_training_confirmed"] is True,
             "teacher_model_or_output_rights_not_cleared")
    _require(_sha(review["teacher_checkpoint_sha256"]) and _sha(review["teacher_outputs_sha256"]),
             "unbound_teacher_checkpoint_or_outputs")


def prepare_shared_plan(plan: Mapping, *, expected_plan_sha256: str, expected_base_sha256: str,
                        receipts: Mapping[str, Mapping],
                        trusted_receipt_sha256: Collection[str]) -> dict:
    """Return redacted preparation metadata; never train, run a teacher or load data.

    The base pin covers the full supplied student JSON entry, including model,
    audio processor, tokenizer, declared routes and rights/lineage evidence.
    Receipt pins are approved separately; inclusion in the plan is not trust.
    This intentionally does not support private, restricted, or unclear
    lineages, dev/test/holdout, tokenizer merging, or audio training from text.
    """
    _require(_sha(expected_plan_sha256) and _sha(expected_base_sha256),
             "independent_plan_and_base_pins_required")
    _require(isinstance(receipts, Mapping) and len(receipts) <= 512
             and isinstance(trusted_receipt_sha256, Collection)
             and not isinstance(trusted_receipt_sha256, (str, bytes, Mapping))
             and len(trusted_receipt_sha256) <= 512
             and all(_sha(v) for v in trusted_receipt_sha256), "invalid_review_trust_pins")
    trusted = frozenset(trusted_receipt_sha256)
    raw, config = _snapshot(plan)
    _require(hashlib.sha256(raw).hexdigest() == expected_plan_sha256, "plan_identity_mismatch")
    _keys(config, _PLAN, "invalid_shared_plan_fields")
    strategy = config["strategy"]
    _require(_schema(config, "shared_speech_training_preparation")
             and config["scope"] == "research_only"
             and type(strategy) is str and strategy in {"joint_training", "distillation"},
             "unsupported_plan_strategy_or_scope")
    routes = _dirs(config["directions"])
    student = config["student"]
    _keys(student, _STUDENT, "invalid_shared_student_fields")
    _require(metadata_sha256(student) == expected_base_sha256, "student_base_identity_mismatch")
    _require(student["architecture"] == "single_streaming_speech_model"
             and type(student["base_revision"]) is str and _REV.fullmatch(student["base_revision"])
             and _dirs(student["directions"]) == routes
             and all(_sha(student[k]) for k in (
                 "checkpoint_sha256", "tokenizer_sha256", "audio_processor_sha256")),
             "one_pinned_student_with_identical_direction_set_required")
    _require(_license(student["license"]) and _clear_lineage(student)
             and student["training_rights_confirmed"] is True, "student_rights_not_cleared")

    materials = config["materials"]
    _require(type(materials) is list and 1 <= len(materials) <= 256, "invalid_material_count")
    extras, paired, distilled = Counter(), Counter(), Counter()
    known_corpus, known_membership, known_outputs = set(), set(), set()
    for row in materials:
        _keys(row, _MATERIAL, "invalid_material_fields")
        direction, objective = row["direction"], row["objective"]
        _require(_direction(direction) and direction in routes and type(objective) is str
                 and objective in _MODES and row["student_base_sha256"] == expected_base_sha256,
                 "material_direction_objective_or_student_mismatch")
        corpus_pin = row["corpus_review_sha256"]
        corpus = _trusted_review(receipts, trusted, corpus_pin)
        _review_corpus(corpus, direction, objective)
        # An exact duplicate must not be counted as extra independent coverage.
        manifest, membership = corpus["corpus_manifest_sha256"], corpus["membership_sha256"]
        _require(manifest not in known_corpus and membership not in known_membership,
                 "material_membership_or_manifest_reused")
        known_corpus.add(manifest)
        known_membership.add(membership)
        (paired if objective == "joint_speech" else extras)[direction] += 1
        teacher_pin = row["teacher_review_sha256"]
        if teacher_pin is not None:
            _require(strategy == "distillation", "teacher_output_requires_explicit_distillation_plan")
            teacher = _trusted_review(receipts, trusted, teacher_pin)
            _review_teacher(teacher, direction=direction, corpus_pin=corpus_pin, membership=membership)
            outputs = teacher["teacher_outputs_sha256"]
            _require(outputs not in known_outputs, "teacher_output_reused")
            known_outputs.add(outputs)
            distilled[direction] += 1

    _require(paired.keys() == routes,
             "paired_audio_transcript_translation_required_for_every_direction")
    _require(strategy != "distillation" or bool(distilled), "distillation_has_no_reviewed_teacher_outputs")
    return {
        "schema_version": 1,
        "kind": "shared_speech_plan_receipt",
        "scope": "research_only",
        "status": "prepared_metadata_only",
        "strategy": strategy,
        "plan_sha256": expected_plan_sha256,
        "student_base_sha256": expected_base_sha256,
        "directions": {
            route: {"paired_speech_materials": paired[route],
                    "text_auxiliary_materials": extras[route],
                    "reviewed_teacher_outputs": distilled[route]}
            for route in sorted(routes)
        },
        "corpus_rows_read": 0,
        "model_files_read": 0,
        "teacher_signal_scope": "text_translation_targets_only",
        "teacher_calls": 0,
        "optimizer_updates": 0,
        "training_authorized": False,
        "production_activated": False,
        "paired_record_overlap_verified": False,
        "independent_evaluation_performed": False,
    }
