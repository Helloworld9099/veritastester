"""Private text is deliberately absent: use synthetic fixtures only."""

import hashlib
import json

import pytest

from research.unified_eval.metrics import MetricError, score_direction


def annotation(check_id, category, kind, *expected):
    return {"id": check_id, "category": category, "kind": kind, "expected": list(expected)}


def run(cases, baseline, candidate, languages=("zh", "en")):
    return score_direction(
        cases, baseline, candidate, source_language=languages[0], target_language=languages[1],
    )


def digest(outputs):
    return hashlib.sha256(
        json.dumps(outputs, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def test_direction_scores_use_same_ids_and_output_sha_no_raw_strings():
    references = [
        "The Zeta display was not calibrated.",
        "Close the old window before leaving.",
    ]
    source = "模拟私有文本不应出现在报告"
    prior = "模拟的更早上下文"
    check = "Zeta display"
    data = [
        {"id": "opaque-2", "source": source, "context": [prior], "reference": references[0],
         "categories": ["clean", "technical"], "checks": [annotation("tech", "technical_terms", "contains", check)]},
        {"id": "opaque-1", "reference": references[1], "categories": ["clean"]},
    ]
    wrong = "Unique disallowed assertion, completely irrelevant."
    before = {"opaque-1": wrong, "opaque-2": wrong}
    after = {"opaque-1": references[1], "opaque-2": references[0]}
    report = run(data, before, after)
    assert report["direction"] == "zh->en"
    assert report["n"] == 2 and [r["id"] for r in report["rows"]] == ["opaque-2", "opaque-1"]
    assert report["baseline"]["chrf"] < report["candidate"]["chrf"] == pytest.approx(100)
    assert report["baseline"]["bleu"] < report["candidate"]["bleu"] == pytest.approx(100)
    assert report["candidate_outputs_sha256"] == report["hashes"]["candidate_outputs"] == digest(after)
    assert report["baseline_outputs_sha256"] == report["hashes"]["baseline_outputs"] == digest(before)
    assert report["metrics"]["bleu"]["tokenizer"] == "13a"
    assert "eff:yes" in report["metrics"]["bleu"]["signature"]
    assert "version:" in report["metrics"]["chrf"]["signature"]
    assert report["rows"][0]["groups"] == ["clean", "technical"]
    assert report["subgroups"]["clean"]["n"] == 2
    assert report["subgroups"]["technical"]["n"] == 1
    assert report["subgroups"]["technical"]["candidate"]["chrf"] == pytest.approx(100)
    assert report["candidate"]["checks"]["by_category"] == {"technical_terms": {"passed": 1, "total": 1}}
    assert report["rows"][1]["candidate"]["checks"] == []
    serialized = json.dumps(report, ensure_ascii=False)
    assert all(secret not in serialized for secret in (source, prior, wrong, check, *references))
    reordered = run(data, dict(reversed(list(before.items()))), dict(reversed(list(after.items()))))
    assert reordered == report


def test_annotations_use_explicit_alternatives_and_word_boundaries():
    data = [{
        "id": "root-1", "reference": "I can't go. Colson Labs ordered two chassis and a gimbal.",
        "checks": [
            annotation("neg-a", "negation", "required_literal", "can't", "cannot"),
            annotation("neg-b", "negation", "forbidden_literal", "can go"),
            annotation("num-a", "numerals", "number", "2"),
            annotation("ent-a", "entities", "contains", "Colson Labs"),
            annotation("tech-a", "technical_terms", "contains", "gimbal"),
            annotation("context-a", "context", "contains", "chassis"),
            annotation("slang-a", "slang", "required_literal", "no way", "forget it"),
            annotation("asr-a", "noisy_input", "required_literal", "windscreen", "windshield"),
        ],
        "categories": ["context", "noisy"],
    }]
    base = {"root-1": "I can go. Colson Lab ordered 2.5 unit chassis and a gimballed mount."}
    cand = {"root-1": "I can’t go. Colson Labs ordered ２ chassis and a gimbal. Forget it. Windscreen."}
    r = run(data, base, cand)
    bad_checks = {x["check_id"]: x["passed"] for x in r["rows"][0]["baseline"]["checks"]}
    assert bad_checks == {
        "neg-a": False, "neg-b": False, "num-a": False, "ent-a": False, "tech-a": False,
        "context-a": True, "slang-a": False, "asr-a": False,
    }
    assert all(x["passed"] for x in r["rows"][0]["candidate"]["checks"])
    assert r["candidate"]["checks"]["total"] == 8
    assert r["candidate"]["checks"]["passed"] == 8
    assert r["subgroups"]["noisy"]["baseline"]["checks"]["total"] == 8


@pytest.mark.parametrize("output", ["12.5", "12%", "312", "A12", "12th", "-12", "1 012", "12/04"])
def test_different_numeric_tokens_do_not_pass(output):
    cases = [{"id": "x", "reference": "Quantity: twelve.", "checks": [annotation("quantity", "numerals", "number", "12")]}]
    r = run(cases, {"x": output}, {"x": "It's ۱۲ units."})
    assert r["baseline"]["checks"]["passed"] == 0
    assert r["candidate"]["checks"]["passed"] == 1


def test_chinese_entity_and_digits_can_touch_source_script():
    data = [{"id": "a", "reference": "北京后来收到了12件。",
             "checks": [annotation("ticker", "entities", "contains", "北京"), annotation("qty", "numerals", "number", "12")]}]
    r = run(data, {"a": "我到北边送了5件。"}, {"a": "北京收到12件。"}, ("EN", "ZH-Hant"))
    assert r["direction"] == "en->zh-hant"
    assert r["metrics"]["bleu"]["tokenizer"] == "zh"
    assert r["candidate"]["checks"]["passed"] == 2
    assert r["baseline"]["checks"]["passed"] == 0


def test_empty_present_output_is_not_missing_and_cannot_pass_forbidden_check():
    data = [{"id": "x", "reference": "Stay right here.",
             "checks": [annotation("bad-prefix", "context", "not_contains", "go elsewhere")]}]
    r = run(data, {"x": "  "}, {"x": "Stay right here."})
    assert r["baseline"]["empty_outputs"] == 1
    assert r["baseline"]["chrf"] == r["baseline"]["bleu"] == 0
    assert r["baseline"]["checks"]["passed"] == 0
    assert r["candidate"]["checks"]["passed"] == 1
    with pytest.raises(MetricError, match="exactly match"):
        run(data, {}, {"x": "Stay right here."})


@pytest.mark.parametrize(
    "bad_case, message",
    [
        ({"id": "real name not allowed as a label", "reference": "Reference."}, "opaque ASCII"),
        ({"id": "a", "reference": "   "}, "nonempty Unicode"),
        ({"id": "a", "reference": "Reference.", "categories": ["x", "x"]}, "duplicate subgroup"),
        ({"id": "a", "reference": "Reference.", "checks": [annotation("r", "numerals", "number", "unknown amount")]}, "exact numeric"),
        ({"id": "a", "reference": "Reference.", "checks": [annotation("r", "unverified", "contains", "test")]}, "unknown annotation category"),
        ({"id": "a", "reference": "Reference.", "checks": [annotation("r", "slang", "contains", "NAME", "name")]}, "duplicate normalized"),
    ],
)
def test_invalid_annotation_receipts_reject_without_raw_payload(bad_case, message):
    with pytest.raises(MetricError, match=message) as err:
        run([bad_case], {"a": ""}, {"a": ""})
    assert "Reference." not in str(err.value)
    assert str(bad_case.get("id")) not in str(err.value) or bad_case.get("id") == "a"


def test_duplicate_ids_and_extra_arm_fail_closed():
    case = {"id": "a", "reference": "Thirty-two."}
    with pytest.raises(MetricError, match="duplicate case ID"):
        run([case, case], {"a": ""}, {"a": ""})
    with pytest.raises(MetricError, match="candidate output IDs"):
        run([case], {"a": ""}, {"a": "", "extra": "other"})
