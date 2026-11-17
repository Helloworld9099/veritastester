import io
from pathlib import Path
import sys
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train
import local_translate


def archive(rows, file_name='fra.txt'):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as z:
        z.writestr(file_name, '\n'.join('\t'.join(x) for x in rows))
    return buffer.getvalue()


class TestCpuContinuation(unittest.TestCase):
    def test_french_to_english_order_and_disjointness(self):
        rows = [(f'English text number {i}', f'Phrase francaise numero {i}', 'CC BY 2.0') for i in range(30)]
        result = train.parse_pairs(archive(rows), seed=11, training_count=12, dev_count=6)
        fitted, dev, total = result
        self.assertEqual((len(fitted), len(dev), total), (12, 6, 30))
        self.assertTrue(all(source.startswith('Phrase francaise') and target.startswith('English text') for source, target in fitted))
        self.assertFalse({s for s, _ in fitted} & {s for s, _ in dev})
        self.assertEqual(train.parse_pairs(archive(rows), seed=11, training_count=12, dev_count=6), result)

    def test_spanish_archive_support(self):
        rows = [(f'English text number {i}', f'Palabras espanolas numero {i}', 'CC-BY') for i in range(30)]
        fitted, dev, total = train.parse_pairs(archive(rows, 'spa.txt'), seed=11,
                                                training_count=12, dev_count=6,
                                                archive_name='spa.txt')
        self.assertEqual((len(fitted), len(dev), total), (12, 6, 30))
        self.assertTrue(all(src.startswith('Palabras espanolas') for src, _ in fitted))

    def test_strict_duplicate_exclusion(self):
        rows = [('The exact English sentence number one', 'Ceci est une phrase francaise.'),
                ('The exact English sentence number one', 'Cette ligne francaise est differente.'),
                ('Different English words here now', 'Ceci est une phrase francaise.')]
        with self.assertRaisesRegex(ValueError, 'Not enough unique'):
            train.parse_pairs(archive(rows), training_count=2, dev_count=1)

    def test_reject_invalid_archive(self):
        with self.assertRaisesRegex(ValueError, 'fra.txt'):
            train.parse_pairs(archive([('a', 'b')], file_name='other.txt'), training_count=1, dev_count=1)

    def test_local_inference_command(self):
        for language in ('french', 'spanish'):
            cmd = local_translate.build_command(Path('/repo'), language, Path('/tmp/base'), 'bonjour', baseline=True)
            self.assertIn('--baseline', cmd)
            self.assertIn('--text', cmd)
            self.assertIn('bonjour', cmd)
            self.assertIn(f'neural_continuation_20261116/{language}', ' '.join(cmd))
            self.assertNotIn('GEMINI_API_KEY', ' '.join(cmd))
        with self.assertRaises(ValueError):
            local_translate.build_command(Path('/repo'), 'unknown', Path('/tmp/base'), 'bonjour')
        with self.assertRaises(ValueError):
            local_translate.build_command(Path('/repo'), 'french', Path('/tmp/base'), '  ')

    def test_pre_registered_gate(self):
        old = {'chrf': 70., 'bleu': 42.}
        self.assertFalse(train.accepted_candidate({'chrf': 70.24, 'bleu': 44.}, old))
        self.assertFalse(train.accepted_candidate({'chrf': 70.4, 'bleu': 41.99}, old))
        self.assertTrue(train.accepted_candidate({'chrf': 70.25, 'bleu': 42.}, old))


if __name__ == '__main__':
    unittest.main()
