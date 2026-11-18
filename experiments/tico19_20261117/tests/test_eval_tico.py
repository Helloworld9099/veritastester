"""No-network TICO benchmark tests: frozen selection and language direction."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_tico import select_rows


def tsv(language="fr", count=300):
    lines = ["\t".join(("sourceLang", "targetLang", "sourceString",
                        "targetString", "stringID", "url", "license", "translator_id"))]
    for idx in range(count):
        lines.append("\t".join([
            "en", language, f"English medical text case number {idx}",
            f"Phrase etrangere du secteur medical numero {idx}",
            f"testcase_{idx}", f"https://example.org/document/{idx // 20}",
            "CC0", "translator-placeholder",
        ]))
    return ("\n".join(lines) + "\n").encode()


class TestExternalBenchmark(unittest.TestCase):
    def test_french_direction_and_frozen_selection(self):
        selected, details = select_rows(tsv(), "french")
        self.assertEqual(len(selected), 256)
        self.assertTrue(all(source.startswith("Phrase etrangere") for source, _ in selected))
        self.assertTrue(all(reference.startswith("English medical") for _, reference in selected))
        self.assertEqual(details, select_rows(tsv(), "french")[1])
        self.assertEqual(len(details["selected_ids_sha256"]), 64)
        self.assertGreater(details["source_group_count"], 10)

    def test_spanish_language_and_alignment(self):
        selected, details = select_rows(tsv(language="es-LA"), "spanish")
        self.assertEqual((len(selected), details["n"]), (256, 256))
        with self.assertRaisesRegex(ValueError, "target language"):
            select_rows(tsv(language="fr"), "spanish")

    def test_modified_parallel_schema_fails(self):
        with self.assertRaisesRegex(ValueError, "parallel TSV schema"):
            select_rows(tsv(count=2) + b"en\tfr\tonly_three_fields\n", "french")
        rows = tsv(count=270).decode().splitlines()
        rows[5] = rows[4]
        with self.assertRaisesRegex(ValueError, "duplicate TICO"):
            select_rows(("\n".join(rows)).encode(), "french")

    def test_missing_or_malformed_header_fails(self):
        with self.assertRaisesRegex(ValueError, "TSV header"):
            select_rows(b"en\tfr\tno header\n", "french")
        with self.assertRaisesRegex(ValueError, "TSV header"):
            select_rows(b"", "french")

    def test_quoted_tsv_fields_are_not_split_mid_translation(self):
        source = "English medical example with embedded tab \t marker"
        foreign = "Phrase etrangere avec tabulation \t interne"
        import csv
        import io
        stream = io.StringIO()
        writer = csv.writer(stream, delimiter="\t", quotechar='"')
        writer.writerow(("sourceLang", "targetLang", "sourceString", "targetString",
                         "stringID", "url", "license", "translator_id"))
        writer.writerow(("en", "fr", source, foreign, "special-1",
                         "document-a", "CC0", "test"))
        rows, info = select_rows(stream.getvalue().encode(), "french", count=1)
        self.assertEqual(rows[0], (foreign, source))
        self.assertEqual(info["n"], 1)

    def test_insufficient_unique_bilingual_data_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Insufficient unique"):
            select_rows(tsv(count=10), "french", count=256)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            select_rows(tsv(), "german")


if __name__ == "__main__":
    unittest.main()
