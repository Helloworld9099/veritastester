"""Lightweight OPUS Books research safeguards; runs without Torch or network."""
import io
import sys
from pathlib import Path
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import books


def make_archive(pair="en-fr", source="fr", num_books=6, rows_each=12):
    zip_bytes = io.BytesIO()
    source_rows, target_rows, metadata = [], [], []
    for book in range(num_books):
        for line in range(rows_each):
            source_rows.append(f"Texte dans le livre numero {book} section {line}")
            target_rows.append(f"Sentence in translated book number {book} section {line}")
            metadata.append(f"en/book{book}.xml.gz {source}/book{book}.xml.gz s{line}.0 s{line}.0")
    with zipfile.ZipFile(zip_bytes, mode="w") as z:
        prefix = "Books." + pair
        z.writestr(prefix + "." + source, "\n".join(source_rows))
        z.writestr(prefix + ".en", "\n".join(target_rows))
        z.writestr(prefix + ".ids", "\n".join(metadata))
    return zip_bytes.getvalue()


class TestBooks(unittest.TestCase):
    def test_document_split_disjoint_and_repeatable(self):
        grouped = books._parse_archived(make_archive(), books.BOOKS["french"])
        x = books.split_books(grouped, seed=33, train_count=18, dev_count=8, test_count=8)
        y = books.split_books(grouped, seed=33, train_count=18, dev_count=8, test_count=8)
        self.assertEqual(x, y)
        self.assertEqual(tuple(map(len, (x["train"], x["dev"], x["test"]))), (18, 8, 8))
        self.assertEqual(x["book_groups_total"], 6)
        self.assertEqual(len(x["doc_group_sha256"]["dev"]), 1)
        self.assertEqual(len(x["doc_group_sha256"]["test"]), 1)
        for col in (0, 1):
            self.assertFalse({row[col] for row in x["train"]} & {row[col] for row in x["dev"]})
            self.assertFalse({row[col] for row in x["train"]} & {row[col] for row in x["test"]})
            self.assertFalse({row[col] for row in x["dev"]} & {row[col] for row in x["test"]})

    def test_strict_source_and_target_direction(self):
        grouped = books._parse_archived(
            make_archive(pair="en-es", source="es"), books.BOOKS["spanish"])
        result = books.split_books(grouped, train_count=18, dev_count=8, test_count=8)
        self.assertTrue(all(row[0].startswith("Texte dans") for row in result["train"]))
        self.assertTrue(all(row[1].startswith("Sentence in") for row in result["test"]))

    def test_reject_bad_checksum_before_parsing(self):
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            books.read_parallel(make_archive(), "french")

    def test_reject_missing_book_metadata(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as z:
            z.writestr("Books.en-fr.fr", "Phrase etrange et longue")
            z.writestr("Books.en-fr.en", "Strange and long sentence")
        with self.assertRaisesRegex(ValueError, "document identifiers"):
            books._parse_archived(data.getvalue(), books.BOOKS["french"])

    def test_reject_too_few_groups(self):
        with self.assertRaisesRegex(ValueError, "3 nonempty"):
            books.split_books({"doc1": [("Bonjour longtemps", "Long hello")]},
                              train_count=1, dev_count=1, test_count=1)


if __name__ == "__main__":
    unittest.main()
