import csv
import pytest
from bridgecaption.review_export import export_candidates, HEADERS


def candidate(**changes):
    return {"case_id": "case-1", "group_id": "group-1", "language": "en",
            "expression": "bear fruit", "utterance": "Efforts bear fruit",
            "previous": "", "situation": "", "split": "train",
            "provenance": "licensed-source:1", "rights_status": "cc-by-4.0-reviewed",
            "meaning": None, **changes}


def test_export_keeps_ungrounded_fields_blank_and_schema_exact(tmp_path):
    out = tmp_path / "private"
    result = export_candidates([candidate()], "sense", out)
    assert result["ready_rows"] == 0
    assert result["needs_review_rows"] == 1
    with (out / "sense_needs_review.csv").open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == HEADERS["sense"]
        row, = reader
    assert row["situation"] == row["previous"] == ""
    assert (out / "sense_queue.csv").read_text().count("\n") == 1
    assert "situation" in (out / "sense_review_metadata.csv").read_text()


@pytest.mark.parametrize("change", [{"split": "test"}, {"split": ""}, {"rights_status": "review-required"}])
def test_export_rejects_holdout_and_unknown_rights(tmp_path, change):
    with pytest.raises(ValueError):
        export_candidates([candidate(**change)], "sense", tmp_path / "private")


def test_export_never_overwrites_or_duplicates_review_work(tmp_path):
    out = tmp_path / "private"
    with pytest.raises(ValueError, match="duplicate"):
        export_candidates([candidate(), candidate()], "sense", out)
    export_candidates([candidate()], "sense", out)
    with pytest.raises(ValueError, match="overwrite"):
        export_candidates([candidate()], "sense", out)
