"""Grounded conversational memory selection. Never invent unstated facts.

The trained logistic predictor is explicitly opt-in and requires independently
annotated, conversation/speaker-disjoint data. Otherwise, use a conservative
heuristic, not a pretend pretrained semantic model.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .research_features import context_features, overlap, words, grams


@dataclass(frozen=True)
class ContextSelection:
    memories: tuple[str, ...]
    mode: str
    abstained: bool


class ContextSelector:
    def __init__(self, model_path: str | Path | None = None,
                 variant: str = "two_view"):
        if variant not in {"single_view", "two_view"}:
            raise ValueError("Unsupported context-view variant")
        self.variant = variant
        self.weights: tuple[float, ...] | None = None
        self.threshold: float | None = None
        if model_path is None:
            model_path = os.environ.get("BRIDGE_CONTEXT_MODEL", "")
        if model_path:
            obj = json.loads(Path(model_path).read_text(encoding="utf-8"))
            if obj.get("method") != "supervised_utility_predictor_not_RL_or_JEPA":
                raise ValueError("Unrecognized context predictor artifact")
            model = obj.get("models", {}).get(variant, {})
            weights = model.get("weights")
            threshold = model.get("threshold")
            if (not isinstance(weights, list) or len(weights) != 8 or
                    any(type(x) not in (int, float) or not math.isfinite(x) for x in weights) or
                    type(threshold) not in (int, float) or not .5 <= threshold <= 1):
                raise ValueError("Malformed context predictor weights or threshold")
            self.weights = tuple(weights)
            self.threshold = float(threshold)

    def _trained_probability(self, current: str, earlier: str,
                             candidate: str, situation: str) -> float:
        assert self.weights is not None
        values = context_features(current, earlier, candidate, situation,
                                  self.variant == "two_view")
        value = sum(w * x for w, x in zip(self.weights, values))
        if value >= 0:
            return 1 / (1 + math.exp(-value))
        ex = math.exp(value)
        return ex / (1 + ex)

    def select(self, current: str, previous: list[str], language: str = "auto",
               topic: str = "general", *, enabled: bool = True,
               background: str = "") -> ContextSelection:
        if not enabled or not previous:
            return ContextSelection((), "disabled" if not enabled else "empty", True)
        candidates = [" ".join(x.split())[:500] for x in previous[-4:] if x.strip()]
        if not candidates:
            return ContextSelection((), "empty", True)
        current = " ".join(current.split())[:1500]
        if self.weights is not None:
            selected = []
            # Do not use withheld reference translations or usefulness labels.
            for i, candidate in enumerate(candidates):
                score = self._trained_probability(
                    current, " ".join(filter(None, [background, *candidates[:i]])),
                    candidate, topic)
                if score >= self.threshold:
                    selected.append(candidate)
            result = tuple(selected[-2:])
            return ContextSelection(result, "trained_predictor", not bool(result))

        from .context import slang_hints
        ambiguous = bool(slang_hints(current, language)) or bool(re.search(
            r"(?<!\w)(?:that|this|it|they|them|he|she|those|these|there|same|one|este|esa|eso|esto|elle|celui)(?!\w)",
            current.casefold()))
        if ambiguous:
            result = tuple(candidates[-2:])
        else:
            result = tuple(m for m in candidates if max(
                overlap(words(current), words(m)),
                overlap(grams(current), grams(m))) >= .1)[-2:]
        return ContextSelection(result, "heuristic_untrained", not bool(result))
