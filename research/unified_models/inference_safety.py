"""Conservative, opt-in anti-degeneration guard for local translation output.

Does NOT repair semantic slang, invented entities, pronoun errors or context.
A review-flagged output triggers ONE local re-decode with a no-repeat
constraint; if the retry also degenerates, reject it instead of returning a
hallucinated/truncated string. Production provider remains unchanged.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable

WORDS = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class RetryResult:
    text: str
    attempts: int
    was_loop: bool
    rescued: bool


class RepetitionLoopError(ValueError):
    """Neither base decode nor bounded retry produced a reliable sentence."""


def _tokens(text: str) -> list[str]:
    return WORDS.findall(unicodedata.normalize("NFKC", text).casefold())


def has_generation_loop(text: str) -> bool:
    """Flags excessive contiguous word/phrase cycles, not normal repetitions.

    Four repetitions of an n-gram can be natural when short (e.g., "no no no
    no"). Require >=8 total tokens and that the repeated segment dominates
    the output. Detect 1..5 word cycles independent of punctuation and case.
    """
    tokens = _tokens(text)
    if len(tokens) < 8:
        return False
    for width in range(1, 6):
        if 4 * width > len(tokens):
            continue
        for start in range(len(tokens) - 4 * width + 1):
            pattern = tokens[start:start + width]
            repetitions = 1
            index = start + width
            while tokens[index:index + width] == pattern:
                repetitions += 1
                index += width
                if index >= len(tokens):
                    break
            coverage = repetitions * width / len(tokens)
            if repetitions >= 4 and repetitions * width >= 8 and coverage >= .65:
                return True
    return False


def retry_if_loop(initial: str, retry: Callable[[], str]) -> RetryResult:
    """Retry once *only* after obvious repetition. No answer editing."""
    if not isinstance(initial, str) or not initial.strip():
        raise ValueError("Translation output is blank")
    if not has_generation_loop(initial):
        return RetryResult(text=initial, attempts=1, was_loop=False, rescued=False)
    alternate = retry()
    if not isinstance(alternate, str) or not alternate.strip():
        raise RepetitionLoopError("Retry returned empty translation")
    if has_generation_loop(alternate):
        raise RepetitionLoopError("Both original and constrained retry looped")
    return RetryResult(text=alternate, attempts=2, was_loop=True, rescued=True)


# Controlled experimental decoding ONLY, requiring validation on real data.
RETRY_GENERATION_OPTIONS = {
    "num_beams": 4,
    "max_new_tokens": 96,
    "no_repeat_ngram_size": 3,
    "repetition_penalty": 1.12,
}
