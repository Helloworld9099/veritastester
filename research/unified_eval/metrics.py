"""Deterministic, paired offline translation diagnostics.

Lexical annotations are small, explicit checks: they do not establish semantic
correctness, dimensions not covered by the supplied checklist are unassessed,
and overlap scores are not percentages of true meaning accuracy.

Call `score_direction(cases, baseline_outputs, candidate_outputs,
source_language="zh", target_language="en")`. Case IDs and subgroup names must
be opaque, non-sensitive labels. One reference per case. Cases carry an `id`,
`reference`, optional `categories` (subgroup labels), and optional `checks`.
A check is `{id, category, kind, expected: [alternative, ...]}`. Text outside
the scoring contract is ignored. The default and only receipt is redacted:
no source, prior, reference, output, or expected literals are rendered.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import hashlib
import json
import math
import re
import unicodedata


SCHEMA_VERSION = 1
CHECK_CATEGORIES = frozenset({
    "negation", "numerals", "entities", "technical_terms", "context", "slang", "noisy_input",
})
CHECK_KINDS = {
    "required_literal": "required_literal",
    "forbidden_literal": "forbidden_literal",
    "required_numeric": "required_numeric",
    "contains": "required_literal",
    "not_contains": "forbidden_literal",
    "number": "required_numeric",
}
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}", flags=re.ASCII)
_LANG = re.compile(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", flags=re.ASCII)
# Separators are retained as written and variant surfaces must be explicitly
# listed by the annotator. Decimal/thousands formats are not guessed.
_NUMERIC = re.compile(
    r"[-+]?(?:\d+(?:(?:[.,:/\-']\d+)|(?: \d{3}(?!\d)))*|[.,]\d+)(?: *[%‰])?",
    flags=re.UNICODE,
)


class MetricError(ValueError):
    """Invalid paired metric input; diagnostics do not include input content."""


def _text(value: Any, label: str, *, nonempty: bool = True) -> str:
    if (
        not isinstance(value, str)
        or (nonempty and not value.strip())
        or "\0" in value
        or any(0xD800 <= ord(char) <= 0xDFFF for char in value)
    ):
        raise MetricError(f"{label} must be valid {'nonempty ' if nonempty else ''}Unicode text")
    return value


def _id(value: Any, label: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise MetricError(f"{label} must be an opaque ASCII identifier")
    return value


def _language(value: Any) -> str:
    if not isinstance(value, str) or _LANG.fullmatch(value) is None:
        raise MetricError("language must be a BCP 47-style code")
    return value.lower()


def _array(value: Any, label: str) -> Sequence:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise MetricError(f"{label} must be an array")
    return value


def _sha_object(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _norm(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("’", "'").replace("−", "-")
    return " ".join(normalized.split())


def _norm_num(text: str) -> str:
    return "".join(str(unicodedata.decimal(ch)) if ch.isdecimal() else ch for ch in _norm(text))


def _matches_num_surface(text: str) -> set[str]:
    text = _norm_num(text)
    tokens = set()
    for match in _NUMERIC.finditer(text):
        left = text[match.start() - 1] if match.start() else ""
        right = text[match.end()] if match.end() < len(text) else ""
        if any(_wordish(ch) and not _uses_script_without_spaces(ch) for ch in (left, right)):
            continue
        tokens.add(match.group(0))
    return tokens


def _uses_script_without_spaces(char: str) -> bool:
    if not char:
        return False
    code = ord(char)
    return (
        0x3040 <= code <= 0x30FF    # kana
        or 0x3400 <= code <= 0x9FFF  # common Han and Extension A
        or 0xF900 <= code <= 0xFAFF  # compatibility Han
        or 0x20000 <= code <= 0x323AF  # supplementary Han
        or 0x0E00 <= code <= 0x0EFF  # Thai/Lao
        or 0x1780 <= code <= 0x17FF  # Khmer
        or 0x1000 <= code <= 0x109F  # Myanmar
    )


def _wordish(char: str) -> bool:
    return bool(char and (char.isalnum() or char == "_" or unicodedata.category(char).startswith("M")))


def _literal_match(text: str, fragment: str) -> bool:
    """Phrases: case/Unicode normalized and word bounded unless script lacks spaces."""
    start = text.find(fragment)
    while start != -1:
        end = start + len(fragment)
        left = text[start - 1] if start else ""
        right = text[end] if end < len(text) else ""
        left_touching = (
            _wordish(left) and _wordish(fragment[0])
            and not (_uses_script_without_spaces(left) or _uses_script_without_spaces(fragment[0]))
        )
        right_touching = (
            _wordish(right) and _wordish(fragment[-1])
            and not (_uses_script_without_spaces(right) or _uses_script_without_spaces(fragment[-1]))
        )
        if not (left_touching or right_touching):
            return True
        start = text.find(fragment, start + 1)
    return False


def _checks(value: Any) -> list[dict[str, Any]]:
    checks = []
    seen = set()
    for raw in _array(value, "checks"):
        if not isinstance(raw, Mapping) or set(raw) != {"id", "category", "kind", "expected"}:
            raise MetricError("annotation check requires id, category, kind, and expected alternatives")
        check_id = _id(raw["id"], "check ID")
        if check_id in seen:
            raise MetricError("duplicate annotation check ID in one case")
        seen.add(check_id)
        category = raw["category"]
        kind = raw["kind"]
        if not isinstance(category, str) or category not in CHECK_CATEGORIES:
            raise MetricError("unknown annotation category")
        if not isinstance(kind, str) or kind not in CHECK_KINDS:
            raise MetricError("unknown annotation check kind")
        kind = CHECK_KINDS[kind]
        values = [_text(part, "expected alternative") for part in _array(raw["expected"], "expected alternatives")]
        if not values:
            raise MetricError("annotation needs at least one expected alternative")
        if kind == "required_numeric":
            if category != "numerals":
                raise MetricError("numeric checks must use the numerals category")
            if any(_NUMERIC.fullmatch(_norm_num(part)) is None for part in values):
                raise MetricError("numeric alternatives must each be one exact numeric token")
        normalized = [_norm_num(part) if kind == "required_numeric" else _norm(part) for part in values]
        if not all(normalized):
            raise MetricError("empty normalized annotation literal")
        if len(set(normalized)) != len(normalized):
            raise MetricError("duplicate normalized expected alternatives")
        checks.append({"id": check_id, "category": category, "kind": kind, "expected": values})
    return checks


def _case_data(cases: Any) -> list[dict[str, Any]]:
    items = []
    seen = set()
    for item in _array(cases, "cases"):
        if not isinstance(item, Mapping) or not {"id", "reference"} <= set(item):
            raise MetricError("each case must supply id and reference")
        case_id = _id(item["id"], "case ID")
        if case_id in seen:
            raise MetricError("duplicate case ID")
        seen.add(case_id)
        groups = [_id(v, "subgroup") for v in _array(item.get("categories", []), "categories")]
        if len(groups) != len(set(groups)):
            raise MetricError("duplicate subgroup for one case")
        case = {
            "id": case_id,
            "reference": _text(item["reference"], "reference"),
            "categories": sorted(groups),
            "checks": _checks(item.get("checks", [])),
        }
        if "source" in item:
            case["source"] = _text(item["source"], "source")
        if "context" in item:
            context = item["context"]
            if isinstance(context, str):
                case["context"] = _text(context, "context", nonempty=False)
            else:
                case["context"] = [_text(turn, "prior turn") for turn in _array(context, "context")]
        items.append(case)
    if not items:
        raise MetricError("at least one paired case is required")
    return items


def _outputs(value: Any, ids: Sequence[str], label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(ids):
        raise MetricError(f"{label} output IDs must exactly match the paired case IDs")
    return {case_id: _text(value[case_id], f"{label} output", nonempty=False) for case_id in ids}


def _checked(output: str, checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normal = _norm(output)
    counts = _matches_num_surface(output) if any(c["kind"] == "required_numeric" for c in checks) else set()
    report = []
    for check in checks:
        kind = check["kind"]
        if not normal:
            passed = False  # an empty model response cannot satisfy absence checks
        elif kind == "required_numeric":
            passed = any(_norm_num(part) in counts for part in check["expected"])
        else:
            hits = any(_literal_match(normal, _norm(part)) for part in check["expected"])
            passed = hits if kind == "required_literal" else not hits
        report.append({"check_id": check["id"], "category": check["category"], "passed": passed})
    return report


def _summarize_checks(all_checks: Sequence[Sequence[dict[str, Any]]]) -> dict[str, Any]:
    categories: dict[str, dict[str, int]] = {}
    for case_checks in all_checks:
        for verdict in case_checks:
            stats = categories.setdefault(verdict["category"], {"passed": 0, "total": 0})
            stats["total"] += 1
            stats["passed"] += int(verdict["passed"])
    return {
        "passed": sum(value["passed"] for value in categories.values()),
        "total": sum(value["total"] for value in categories.values()),
        "by_category": dict(sorted(categories.items())),
    }


def _finite(score: Any) -> float:
    value = float(score)
    if not math.isfinite(value) or value < -1e-9 or value > 100.0 + 1e-9:
        raise MetricError("SacreBLEU returned a nonfinite or out-of-range score")
    return min(max(value, 0.0), 100.0)


def _tokenizer(target_language: str) -> str:
    primary = target_language.split("-", 1)[0]
    if primary == "zh":
        return "zh"
    # CJK variants beyond Chinese are usable via chrF; BLEU uses Intl unless
    # a separate, pinned segmentation stack is defined in a later protocol.
    return "13a" if primary in {"en", "es", "fr"} else "intl"


def score_direction(
    cases: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, str],
    candidate: Mapping[str, str],
    *,
    source_language: str,
    target_language: str,
) -> dict[str, Any]:
    """Score the same IDs and reference against both arms, returning a redacted receipt.

    BLEU is effective-order corpus BLEU (caption fragments may be short); each
    subgroup is independently computed as a corpus, never as a mean of sentence
    scores. Row chrF is diagnostic, not a separate BLEU estimate. All checks
    must be explicitly annotated when used; an unannotated case is not passed.
    An empty present output counts as text failure; a missing ID is rejected.
    """
    source_language, target_language = _language(source_language), _language(target_language)
    selected = _case_data(cases)
    ids = [case["id"] for case in selected]
    baseline = _outputs(baseline, ids, "baseline")
    candidate = _outputs(candidate, ids, "candidate")
    try:
        from sacrebleu.metrics import BLEU, CHRF
    except ImportError as exc:
        raise MetricError("offline metrics require the repository's sacrebleu evaluation dependency") from exc

    tokenizer = _tokenizer(target_language)
    chrf = CHRF(char_order=6, word_order=0, beta=2)
    bleu = BLEU(tokenize=tokenizer, smooth_method="exp", effective_order=True, force=True)
    rows = []
    indices_by_group: dict[str, list[int]] = {}
    references = [case["reference"] for case in selected]
    for index, case in enumerate(selected):
        case_id = case["id"]
        for group in case["categories"]:
            indices_by_group.setdefault(group, []).append(index)
        row = {"id": case_id, "groups": case["categories"]}
        for arm, outputs in (("baseline", baseline), ("candidate", candidate)):
            output = outputs[case_id]
            row[arm] = {
                "chrf": _finite(chrf.sentence_score(output, [case["reference"]]).score),
                "checks": _checked(output, case["checks"]),
            }
        rows.append(row)

    def receipt(indices: Sequence[int]) -> dict[str, Any]:
        subset_references = [[references[i] for i in indices]]
        summary = {"n": len(indices)}
        for arm, outputs in (("baseline", baseline), ("candidate", candidate)):
            hypotheses = [outputs[ids[i]] for i in indices]
            summary[arm] = {
                "chrf": _finite(chrf.corpus_score(hypotheses, subset_references).score),
                "bleu": _finite(bleu.corpus_score(hypotheses, subset_references).score),
                "empty_outputs": sum(not output.strip() for output in hypotheses),
                "checks": _summarize_checks([rows[i][arm]["checks"] for i in indices]),
            }
        return summary

    overall = receipt(tuple(range(len(ids))))
    signatures = {"chrf": str(chrf.get_signature()), "bleu": str(bleu.get_signature())}
    subgroups = {k: receipt(indices) for k, indices in sorted(indices_by_group.items())}
    case_hash, base_hash, new_hash = _sha_object(selected), _sha_object(baseline), _sha_object(candidate)
    return {
        "metric_schema_version": SCHEMA_VERSION,
        "direction": f"{source_language}->{target_language}",
        **overall,
        "baseline_outputs_sha256": base_hash,
        "candidate_outputs_sha256": new_hash,
        "hashes": {"case_spec": case_hash, "baseline_outputs": base_hash, "candidate_outputs": new_hash},
        "metrics": {
            "chrf": {"name": "SacreBLEU chrF2", "signature": signatures["chrf"]},
            "bleu": {"name": "SacreBLEU BLEU", "signature": signatures["bleu"], "tokenizer": tokenizer},
            "annotation_policy": "deterministic lexical checks, not semantic adjudication",
        },
        "rows": rows,
        "subgroups": subgroups,
    }
