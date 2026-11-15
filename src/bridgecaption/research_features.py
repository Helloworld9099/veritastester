"""Shared eight-feature vector for training and context-memory inference.

No reference translation or utility label enters these features.
This is a lexical pilot, not a multilingual semantic encoder or JEPA.
"""
import re


def words(text: str) -> set[str]:
    return set(re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE))


def grams(text: str) -> set[str]:
    text = " ".join(text.casefold().split())
    return {text[i:i + 3] for i in range(max(0, len(text) - 2))}


def overlap(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def context_features(utterance: str, previous: str, candidate: str,
                     situation: str = "", multiview: bool = True) -> tuple[float, ...]:
    """One stable feature signature shared by training and production."""
    return (1.0,
            overlap(words(utterance), words(candidate)),
            overlap(grams(utterance), grams(candidate)),
            overlap(words(previous), words(candidate)),
            overlap(grams(previous), grams(candidate)),
            overlap(words(situation), words(candidate)) if multiview else 0.0,
            overlap(grams(situation), grams(candidate)) if multiview else 0.0,
            min(len(candidate) / 500.0, 1.0))
