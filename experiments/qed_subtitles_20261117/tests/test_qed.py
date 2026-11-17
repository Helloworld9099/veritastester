"""No-network tests for verified bilingual QED TMX pairs and split integrity."""
import gzip
import hashlib
from contextlib import contextmanager
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_qed import QED, aligned_subtitles, select_cohorts


def archive(language, docs=16, each=20):
    spec = QED[language]
    pieces = ['<?xml version="1.0" encoding="UTF-8" ?>',
              '<tmx version="1.4"><header srclang="en" datatype="PlainText"/>',
              '<body>']
    for doc in range(docs):
        for i in range(each):
            src = f'Traduction internationale discours numero {doc} segment {i}'
            en = f'International lecture sentence document {doc} number {i}'
            pieces.append(
                f'<tu><tuv xml:lang="en"><seg>{en}</seg></tuv>'
                f'<tuv xml:lang="{spec["source"]}"><seg>{src}</seg></tuv></tu>')
    pieces.extend(['</body>', '</tmx>'])
    return gzip.compress('\n'.join(pieces).encode('utf-8'))


@contextmanager
def checked_fixture(language, payload):
    """Supply fixture's own verified digest without weakening production pins."""
    original = QED[language]["sha256"]
    try:
        QED[language]["sha256"] = hashlib.sha256(payload).hexdigest()
        yield
    finally:
        QED[language]["sha256"] = original


class TestQED(unittest.TestCase):
    def test_historical_archive_digests_are_pinned(self):
        self.assertEqual(QED["french"]["sha256"], "35b4ca45f4d105976c47745a7a6b39d6f3c16403bdcebc8124a89cea262848df")
        self.assertEqual(QED["spanish"]["sha256"], "6ecd8a6c1e12d767c3451b52d09860891213dbc382b00b8bfd5ba06962460ea0")

    def test_pair_alignment_and_splits_for_both_languages(self):
        for language in QED:
            payload = archive(language)
            with checked_fixture(language, payload):
                rows, sha, eligible = aligned_subtitles(payload, language, reservoir_cap=500)
            self.assertEqual(eligible, 320)
            self.assertEqual(len(sha), 64)
            self.assertEqual(len(rows), 320)
            tr, dev, test = select_cohorts(rows, seed=84,
                                           train_count=128, dev_count=32, test_count=32)
            self.assertEqual((len(tr), len(dev), len(test)), (128, 32, 32))
            self.assertTrue(all(s.startswith('Traduction internationale') for s, _ in tr))
            self.assertTrue(all(t.startswith('International lecture') for _, t in test))

    def test_mismatched_moses_is_never_used(self):
        payload = b"not a gzip archive"
        with checked_fixture("french", payload):
            with self.assertRaisesRegex(RuntimeError, "compressed QED TMX"):
                aligned_subtitles(payload, "french")

    def test_source_digest_mismatch_fails(self):
        previous = QED["french"]["sha256"]
        try:
            QED["french"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "changed since recorded checksum"):
                aligned_subtitles(archive("french"), "french")
        finally:
            QED["french"]["sha256"] = previous

    def test_deduplicate_source_and_english_exact_text(self):
        data = archive("spanish", docs=4, each=20)
        with checked_fixture("spanish", data):
            rows, _, _ = aligned_subtitles(data, "spanish", reservoir_cap=100)
        self.assertEqual(len(rows), 80)
        row = rows[0]
        with self.assertRaisesRegex(RuntimeError, "duplicated across"):
            select_cohorts([row, row, ('Esto texto diferente ahora', 'This is a different sentence')],
                           train_count=1, dev_count=1, test_count=1)

    def test_tagged_tmx_with_missing_language_fails_conservatively(self):
        payload = gzip.compress(
            b'<?xml version="1.0"?><tmx><body><tu><tuv xml:lang="en"><seg>Only English here</seg></tuv></tu></body></tmx>')
        with checked_fixture("french", payload):
            rows, digest, eligible = aligned_subtitles(payload, "french")
        self.assertEqual((rows, eligible), ([], 0))
        self.assertEqual(len(digest), 64)
        with self.assertRaisesRegex(ValueError, "Too few distinct"):
            select_cohorts(rows, train_count=1, dev_count=1, test_count=1)


if __name__ == "__main__":
    unittest.main()
