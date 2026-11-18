"""Conservative numeric-semantic guardrail for bilingual translation evaluation.

This is a *partial automatic error check*, not an accuracy score. No trained
judges and no hidden model calls. Source/test rows are never used for training.
"""
from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
import re

# Attach percentage markers even when separated by a narrow/nonbreaking space.
TOKEN = re.compile(r"(?<!\w)[+-]?\d+(?:[,.]\d+)*(?:[ \u00a0\u202f]*%)?(?!\w)")
GROUP_SPACES = re.compile(r"\d{1,3}(?:[\u00a0\u202f ]\d{3})+(?:[.,]\d+)?")

def parse_number(value: str, lang: str) -> tuple[str, bool] | None:
    raw = value.replace("\u00a0", " ").replace("\u202f", " ").strip()
    is_percent = raw.endswith("%")
    raw = raw.rstrip("%").strip().replace(" ", "")
    if not raw:
        return None
    # Ambiguous "1,234" in a French/Spanish original could mean 1.234 or 1234.
    # Better report "uncertain" than claim a false error.
    if lang in ("fr", "es"):
        if re.fullmatch(r"\d{1,3},\d{3}", raw) and not raw.startswith("0,"):
            return None
        if "," in raw and "." in raw:
            if raw.rfind(",") > raw.rfind("."):
                raw = raw.replace(".", "").replace(",", ".")
            else:
                raw = raw.replace(",", "")
        elif "," in raw:
            raw = raw.replace(",", ".")
        elif re.fullmatch(r"\d{1,3}(?:\.\d{3})+", raw):
            raw = raw.replace(".", "")
    elif lang == "en":
        if "," in raw and "." in raw:
            raw = raw.replace(",", "")
        elif re.fullmatch(r"\d{1,3}(?:,\d{3})+", raw):
            raw = raw.replace(",", "")
        elif "," in raw:
            return None
    else:
        raise ValueError("Language must be fr, es or en")
    try:
        digit = Decimal(raw)
    except InvalidOperation:
        return None
    if not digit.is_finite():
        return None
    return (str(digit.normalize()), is_percent)


def tokens(text: str, language: str):
    if not isinstance(text, str):
        raise ValueError("Expected translation text")
    matches = list(GROUP_SPACES.finditer(text))
    excluded = set()
    output = []
    for match in matches:
        # Grouped thousands are explicit and can be compared exactly.
        normalized = match.group().replace(" ", "").replace("\u00a0", "").replace("\u202f", "")
        parsed = parse_number(normalized, "en" if "," in normalized else "fr")
        if parsed is not None:
            output.append(parsed)
        excluded.update(range(match.start(), match.end()))
    uncertain = 0
    for match in TOKEN.finditer(text):
        if any(i in excluded for i in range(match.start(), match.end())):
            continue
        parsed = parse_number(match.group(), language)
        if parsed is None:
            uncertain += 1
        else:
            output.append(parsed)
    return Counter(output), uncertain


def numeric_fidelity(source: str, translation: str, language: str):
    """Flag obvious altered/missing numbers; do not treat 'pass' as adequacy."""
    if language not in ("french", "spanish"):
        raise ValueError("Only French and Spanish are supported")
    src_lang = {"french": "fr", "spanish": "es"}[language]
    left, source_uncertain = tokens(source, src_lang)
    right, target_uncertain = tokens(translation, "en")
    if source_uncertain or target_uncertain:
        status = "uncertain"
    elif not left and not right:
        status = "no_numeric_facts"
    elif left == right:
        status = "numbers_preserved"
    else:
        status = "numeric_error"
    return {
        "status": status,
        "source_values": [str(x) for x in sorted(left.elements())],
        "target_values": [str(x) for x in sorted(right.elements())],
        "limitations": "Only literal numeric facts; no semantic-equivalence or 90-percent accuracy claim.",
    }
