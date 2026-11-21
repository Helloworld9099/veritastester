import hashlib
import json

import pytest

from research.unified_eval.review import ReviewError, load_development_review


PINS = dict(fixture_sha256="a" * 64, baseline_outputs_sha256="b" * 64,
            candidate_outputs_sha256="c" * 64)
CASES = {"example-01": "es->en", "example-02": "fr->en"}


@pytest.fixture
def review(tmp_path):
    data = {"version": 1, "kind": "paired_blinded_development_review", **PINS,
            "judgments": {"example-01": {"baseline": [], "candidate": ["quantity:time_1"]},
                          "example-02": {"baseline": ["negation:request_1"], "candidate": []}}}
    def read(*, modify=None, sha=None):
        if modify is not None:
            modify(data)
        raw = json.dumps(data).encode()
        path = tmp_path / "development.json"
        path.write_bytes(raw)
        return load_development_review(path, expected_sha256=sha or hashlib.sha256(raw).hexdigest(),
                                       case_directions=CASES, **PINS)
    return read


def test_independently_pinned_development_review_groups_paired_labels(review):
    assert review()["es->en"]["example-01"] == {"baseline": [], "candidate": ["quantity:time_1"]}


@pytest.mark.parametrize("change, message", [
    (lambda x: x.update(kind="sealed_final"), "development-review schema"),
    (lambda x: x.update(baseline_outputs_sha256="d" * 64), "pins do not match"),
    (lambda x: x["judgments"].pop("example-02"), "every selected case"),
    (lambda x: x["judgments"]["example-01"].pop("baseline"), "Paired development"),
    (lambda x: x["judgments"]["example-01"].update(candidate=["unknown-meaning"]),
     "Unknown or duplicate"),
])
def test_development_labels_reject_wrong_scope_pins_or_coverage(review, change, message):
    with pytest.raises(ReviewError, match=message):
        review(modify=change)


def test_locally_changed_receipt_must_match_independently_supplied_digest(review):
    with pytest.raises(ReviewError, match="checksum"):
        review(sha="e" * 64)
