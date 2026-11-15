"""Compose two-view and selective-memory *review candidates* from local corpora.

A previous turn is NOT automatically helpful and a distractor is NOT automatically
harmful. No training label is synthesized. Reviewers must add a verified
situation and reference translation before the existing annotation CLI accepts
a row as supervised training data.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .corpus_prep import audit, digest, private_write
from .sense_model import contains_expression


def load_normalized(paths: list[Path], limit: int = 200_000):
    items, seen = [], set()
    for path in paths:
        with path.open(encoding="utf-8") as file:
            for number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                if len(items) >= limit:
                    raise ValueError("Record limit reached")
                row = json.loads(line)
                if row.get("schema") != "bridgecaption_corpus_v1":
                    raise ValueError(f"Unexpected normalized schema in {path.name}:{number}")
                if not isinstance(row.get("utterance"), str) or not row["utterance"].strip():
                    raise ValueError("Missing utterance")
                key = row.get("case_id")
                if not isinstance(key, str) or not key:
                    raise ValueError("Missing case ID")
                if key in seen:
                    raise ValueError(f"Duplicate case ID across datasets: {key}")
                seen.add(key)
                items.append(row)
    if not items:
        raise ValueError("No normalized records")
    return items


def leakage_report(records):
    report = audit(records)
    source_labels = Counter((r["dataset"], r["label_kind"]) for r in records)
    report["source_label_families"] = {
        f"{source}/{label}": count for (source,label),count in sorted(source_labels.items())
    }
    report["official_test_rows"] = sum(r["dataset"] == "diabla" for r in records)
    report["directly_ready_for_memory_training"] = 0
    report["directly_ready_for_slang_meaning_training"] = 0
    return report


def human_memory_candidates(records, *, seed=2026, max_items=5000):
    """Only non-test multi-turn dialogue; explicit same/different-group candidates."""
    eligible = [r for r in records if r["role"] == "conversation_review" and
                r["split"] in ("train", "dev") and r["utterance"]]
    groups = defaultdict(list)
    for row in eligible:
        groups[(row["dataset"], row["language"], row["split"], row["group_id"])].append(row)
    pools = defaultdict(list)
    for key, items in groups.items():
        dataset, language, split, _group_id = key
        # Nonempty dialogue turns across different conversations in the same split.
        pools[(dataset, language, split)].append((key, items))
    rng = random.Random(seed)
    outputs = []
    for key, turns in sorted(groups.items()):
        if len(turns) < 3:
            continue
        dataset, language, split, group_id = key
        unrelated = [(other_key, other_turns) for other_key, other_turns
                     in pools[(dataset, language, split)] if other_key != key]
        for i in range(2, len(turns)):
            current, prior, older = turns[i], turns[i-1], turns[i-2]
            candidates = [("prior_turn_unjudged", older)]
            if unrelated:
                other_key, other_turns = rng.choice(unrelated)
                candidates.append(("other_dialogue_unjudged", rng.choice(other_turns)))
            for origin, candidate in candidates:
                outputs.append({
                    "case_id": f"review:{digest(current['case_id']+'|'+candidate['case_id']+'|'+origin)}",
                    "group_id": group_id,
                    "language": language,
                    "utterance": current["utterance"],
                    "previous": prior["utterance"],
                    "candidate_memory": candidate["utterance"],
                    "situation": "",  # Must be grounded by a human, never guessed.
                    "reference_translation": "",  # Needs independent fluent annotation.
                    "provenance": f"{dataset}:{current['case_id']}:{candidate['case_id']}",
                    "split": split,
                    "candidate_origin": origin,
                    "candidate_group_id": candidate["group_id"],
                    "rights_status": current.get("rights_status", ""),
                    "source_sha256": current.get("source_sha256", ""),
                    "source_filename": current.get("source_filename", ""),
                    "human_judgement": None,
                    "warning": "Unlabelled comparison. Older != helpful; other dialogue != harmful.",
                })
                if len(outputs) >= max_items:
                    return outputs
    return outputs


def temporal_alignment_pairs(records, *, seed=2026, max_items=5000):
    """Self-supervised adjacency proxy: NOT usefulness or translation truth.

    Positive means "adjacent turns in the SAME dialogue." Negative means
    "from DIFFERENT dialogues in the same language and split." Neither label
    claims semantic equivalence, helpful memory, or a verified second scene.
    """
    eligible = [r for r in records if r["role"] == "conversation_review" and
                r["split"] in {"train", "dev"}]
    groups = defaultdict(list)
    for row in eligible:
        groups[(row["dataset"], row["language"], row["split"], row["group_id"])].append(row)
    buckets = defaultdict(list)
    for key, turns in groups.items():
        buckets[key[:3]].append((key, turns))
    rng = random.Random(seed)
    pairs = []
    for key, turns in sorted(groups.items()):
        if len(turns) < 2:
            continue
        dataset, language, split, group_id = key
        alternatives = [(other_key, values) for other_key, values in buckets[key[:3]]
                        if other_key != key]
        for i in range(1, len(turns)):
            current, prior = turns[i], turns[i-1]
            pair = {
                "case_id": f"adj:{digest(current['case_id']+':'+prior['case_id'])}",
                "group_id": group_id,
                "language": language, "split": split,
                "view_utterance": current["utterance"],
                "view_previous_turn": prior["utterance"],
                "proxy_label": "same_dialogue_adjacent",
                "proxy_task": "temporal_coherence_not_memory_usefulness",
                "human_judgement": None, "situation_view": "",
                "provenance": f"{dataset}:{current['case_id']}:{prior['case_id']}",
            }
            pairs.append(pair)
            if alternatives:
                other_key, other = rng.choice(alternatives)
                candidate = rng.choice(other)
                pairs.append({**pair,
                    "case_id": f"adj:{digest(current['case_id']+':'+candidate['case_id'])}",
                    "view_previous_turn": candidate["utterance"],
                    "proxy_label": "different_dialogue",
                    "provenance": f"{dataset}:{current['case_id']}:{candidate['case_id']}"})
            if len(pairs) >= max_items:
                return pairs[:max_items]
    return pairs


def read_terms(path: Path):
    """Optional explicitly reviewed term list, never a fabricated sense dictionary."""
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"language","expression"}.issubset(reader.fieldnames):
            raise ValueError("Term CSV requires language,expression")
        terms = defaultdict(set)
        for row in reader:
            code = (row.get("language") or "").strip().casefold()
            expression = (row.get("expression") or "").strip()
            if not code or not expression or len(expression) > 60:
                raise ValueError("Invalid language/expression in term list")
            terms[code].add(expression)
    return terms


def human_sense_candidates(records, *, terms=None, max_items=5000):
    result, seen = [], set()
    terms = terms or {}
    for r in records:
        if r["split"] == "test" or r["role"] in ("evaluation_only", "parallel_text"):
            continue
        # Source-provided idiom labels may use a dictionary form, e.g.
        # "spill the beans" when the actual utterance says "spilled the beans".
        # Preserve those as HUMAN-REVIEW candidates, not verified senses.
        # In contrast, externally curated terms must match the actual text.
        source_expression = r.get("expression") or ""
        candidate_terms = [source_expression] if source_expression else []
        candidate_terms += [term for term in sorted(terms.get(r["language"], []))
                            if contains_expression(r["utterance"], term)]
        for expression in dict.fromkeys(filter(None, candidate_terms)):
            verbatim_match = contains_expression(r["utterance"], expression)
            if not verbatim_match and expression != source_expression:
                continue
            key = (r["case_id"], expression.casefold())
            if key in seen:
                continue
            seen.add(key)
            # Idiom-type holdouts are correct for idiomaticity, but exact sense
            # training needs the same expression in independent documents.
            document_id = r.get("extra", {}).get("document_id", "")
            review_group = ("magpie-document:" + digest(document_id)
                            if r["dataset"] == "magpie" and document_id else r["group_id"])
            result.append({
                "case_id": f"review:{digest(r['case_id']+'|'+expression)}",
                "group_id": review_group, "language": r["language"],
                "source_group_id": r["group_id"], "source_document_id": document_id,
                "expression": expression, "utterance": r["utterance"],
                "previous": r["previous"], "situation": "",  # Independent view needed.
                "meaning": None, "source_suggested_definition": r.get("candidate_definition", ""),
                "source_label": r.get("source_label", ""),
                "label_kind": r.get("label_kind", ""),
                "source_expression_matches_verbatim": verbatim_match,
                "rights_status": r.get("rights_status", ""),
                "source_sha256": r.get("source_sha256", ""),
                "source_filename": r.get("source_filename", ""),
                "provenance": r["provenance"],
                "split": r["split"],
                "warning": ("Source phrase is not verbatim in the utterance (it may be a "
                            "dictionary form or incorrect metadata); verify manually. "
                            if not verbatim_match else "") +
                           "Source label/definition is not an independently reviewed sense.",
            })
            if len(result) >= max_items:
                return result
    return result


def main():
    parser = argparse.ArgumentParser(description="Create human curation queues, not model labels")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--task", choices=("audit","memory","sense","alignment"), default="audit")
    parser.add_argument("--terms", type=Path, help="Optional human-curated language/expression CSV")
    parser.add_argument("--max-items", type=int, default=2000)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_items <= 200_000:
        parser.error("--max-items must be 1..200000")
    try:
        records = load_normalized(args.inputs)
        summary = leakage_report(records)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if summary["cross_split_group_conflicts"]:
            parser.error("Refusing review preparation because group leakage was detected")
        if summary["cross_split_exact_text_warnings"]:
            print("WARNING: Repeated long utterances across splits; manually inspect duplicates")
        if args.task == "memory":
            output = human_memory_candidates(records, max_items=args.max_items)
        elif args.task == "sense":
            output = human_sense_candidates(
                records, terms=read_terms(args.terms) if args.terms else None,
                max_items=args.max_items)
        elif args.task == "alignment":
            output = temporal_alignment_pairs(records, max_items=args.max_items)
        else:
            output = []
        print(f"Prepared {len(output)} {args.task} records; "
              "alignment labels mean dialogue adjacency only, not helpful memories")
        if args.write:
            if args.task == "audit":
                parser.error("Audit mode never writes source records")
            if not args.out:
                parser.error("--out required with --write")
            private_write(args.out, output)
            print(f"Saved {len(output)} unlabelled review candidates to {args.out}")
        else:
            print("DRY RUN; no data written and no training started")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
