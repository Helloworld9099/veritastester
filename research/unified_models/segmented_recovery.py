"""Bounded source-clause segmentation as RESEARCH-ONLY translation recovery.

Preserves each current-speaker clause without using untrained past dialogue.
Not automatically enabled for app users: splitting sentences may damage
pronouns, pragmatic tone and domain terminology, so require human adequacy.
"""
from __future__ import annotations

import re

# Only split at completed sentence punctuation, with following whitespace
# and a plausible next utterance. No hidden future conversation is used.
SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+(?=[¡¿A-Za-zÀ-ÖØ-öø-ÿ])"
)
ENCODED_BREAKS = re.compile(r"(?:\\[nr])+")


def separate_current_clauses(text: str, *, max_clauses: int = 6) -> tuple[str, ...]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Current utterance required")
    if not 1 <= max_clauses <= 8:
        raise ValueError("Unsafe clause budget")
    # The public 500-case development export contains escaped linebreaks;
    # live ASR transcription normally contains real whitespace instead.
    clean = " ".join(ENCODED_BREAKS.sub(" ", text).split())
    clauses = tuple(x.strip() for x in SENTENCE_BOUNDARY.split(clean) if x.strip())
    if not clauses:
        raise ValueError("No current clauses")
    if len(clauses) > max_clauses:
        raise ValueError("Too many clauses for this limited research decoder")
    if " ".join(clauses) != clean:
        raise RuntimeError("Source-clause segmentation lost words or characters")
    return clauses


def translate_clauses(clauses: tuple[str,...], inference) -> str:
    """inference is callable for each current-source clause; never a translator key."""
    if not isinstance(clauses, tuple) or not clauses:
        raise ValueError("Expected nonempty ordered current-source clauses")
    outputs=[]
    for clause in clauses:
        translated=inference(clause)
        if not isinstance(translated,str) or not translated.strip():
            raise RuntimeError("Segment decoder omitted an entire source clause")
        outputs.append(translated.strip())
    if len(outputs)!=len(clauses):
        raise RuntimeError("A source clause was not translated")
    return " ".join(outputs)
