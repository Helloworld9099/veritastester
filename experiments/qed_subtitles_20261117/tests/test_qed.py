"""No-network tests of QED source direction, document separation, and deduplication."""
import io
from pathlib import Path
import sys
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_qed import QED, aligned_subtitles, split_books


def archive(language, docs=12, each=12):
    spec = QED[language]
    prefix = "QED." + spec["pair"]
    src, en, alignment_xml = [], [], ['<cesAlign version="1.0">']
    for doc in range(docs):
        alignment_xml.append(
            f'<linkGrp fromDoc="en/lecture{doc}.xml.gz" '
            f'toDoc="{spec["source"]}/lecture{doc}.xml.gz">')
        for i in range(each):
            src.append(f"Traduction internationale discours numero {doc} segment {i}")
            en.append(f"International lecture sentence document {doc} number {i}")
            alignment_xml.append(f'<link xtargets="s{i};s{i}" />')
        alignment_xml.append('</linkGrp>')
    alignment_xml.append('</cesAlign>')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr(prefix + "." + spec["source"], "\n".join(src))
        z.writestr(prefix + ".en", "\n".join(en))
        z.writestr(prefix + ".xml", "\n".join(alignment_xml))
    return buffer.getvalue()


class TestQED(unittest.TestCase):
    def test_two_languages_document_separation(self):
        for language in QED:
            grouped, sha = aligned_subtitles(archive(language), language)
            self.assertEqual(len(grouped), 12)
            self.assertEqual(len(sha), 64)
            split = split_books(grouped, seed=84, train_count=64, dev_count=16, test_count=16)
            self.assertEqual((len(split["train"]), len(split["dev"]), len(split["test"])),
                             (64, 16, 16))
            self.assertTrue(all(t.startswith("International lecture") for s, t in split["test"]))

    def test_missing_identifiers_fail(self):
        s = io.BytesIO()
        with zipfile.ZipFile(s, "w") as z:
            z.writestr("QED.en-fr.fr", "Bonjour longue phrase")
            z.writestr("QED.en-fr.en", "Hello long phrase")
        with self.assertRaisesRegex(RuntimeError, "text/XML alignment"):
            aligned_subtitles(s.getvalue(), "french")

    def test_changed_archive_digest_is_rejected(self):
        previous = QED["french"]["sha256"]
        try:
            QED["french"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "changed unexpectedly"):
                aligned_subtitles(archive("french"), "french")
        finally:
            QED["french"]["sha256"] = previous


if __name__ == "__main__":
    unittest.main()
