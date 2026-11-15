"""Parse the official DiaBLa JSON as held-out-only data. Does not execute HF code."""
import json
from pathlib import Path
from bridgecaption.corpus_prep import prepare, private_write, audit


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "data/private/raw/diabla-DiaBLa.json"
    data = json.loads(source.read_text())
    exports = []
    empty_reference = 0
    for dialogue_id, dialogue in sorted(data["dialogues"].items()):
        history = []
        # Preserve source insertion order, as the published loader does.
        for uid, utterance in dialogue["utterances"].items():
            text = utterance["original_text"]
            reference = utterance["reference_translation"]
            if reference and str(reference).strip():
                exports.append({"id": dialogue_id+":"+uid, "dialogue_id": dialogue_id,
                                "orig": text, "ref": reference,
                                "utterance_meta": {"lang": utterance["language"]},
                                "previous": " ".join(history[-3:])})
            else:
                empty_reference += 1
            history.append(text)
    verified = root / "data/private/raw/diabla-grouped-test.jsonl"
    private_write(verified, exports)
    rows = prepare("diabla", verified, rights="cc-by-sa-4.0-evaluation-only-reviewed")
    private_write(root / "data/private/diabla-heldout.jsonl", rows)
    report = {"audit": audit(rows), "source_dialogues": len(data["dialogues"]),
              "empty_references_excluded": empty_reference,
              "reference_target": "opposite language; filter French source for English-target evaluation",
              "used_for_training_or_threshold_tuning": False,
              "translation_inference_executed": False}
    out = root / "results/diabla_import.json"
    with out.open("x") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
