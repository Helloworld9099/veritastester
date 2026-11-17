"""No-network tests for verified bilingual QED TMX pairs and split integrity."""
import gzip
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


class TestQED(unittest.TestCase):
    def test_pair_alignment_and_splits_for_both_languages(self):
        for language in QED:
            rows, sha, eligible = aligned_subtitles(archive(language), language, reservoir_cap=500)
            self.assertEqual(eligible, 320)
            self.assertEqual(len(sha), 64)
            self.assertEqual(len(rows), 320)
            tr, dev, test = select_cohorts(rows, seed=84,
                                           train_count=128, dev_count=32, test_count=32)
            self.assertEqual((len(tr), len(dev), len(test)), (128, 32, 32))
            self.assertTrue(all(s.startswith('Traduction internationale') for s, _ in tr))
            self.assertTrue(all(t.startswith('International lecture') for _, t in test))

    def test_mismatched_moses_is_never_used(self):
        with self.assertRaisesRegex(RuntimeError, "compressed QED TMX"):
            aligned_subtitles(b"not a gzip archive", "french")

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
        rows, _, _ = aligned_subtitles(data, "spanish", reservoir_cap=100)
        self.assertEqual(len(rows), 80)
        row = rows[0]
        with self.assertRaisesRegex(RuntimeError, "duplicated across"):
            select_cohorts([row, row, ('Esto texto diferente ahora', 'This is a different sentence')],
                           train_count=1, dev_count=1, test_count=1)

    def test_tagged_tmx_with_missing_language_fails_conservatively(self):
        payload = gzip.compress(
            b'<?xml version="1.0"?><tmx><body><tu><tuv xml:lang="en"><seg>Only English here</seg></tuv></tu></body></tmx>')
        with self.assertRaisesRegex(RuntimeError, "Insufficient distinct"):
            aligned_subtitles(payload, "french")


if __name__ == "__main__":
    unittest.main()
