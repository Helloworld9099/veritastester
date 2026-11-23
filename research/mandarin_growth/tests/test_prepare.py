import unittest
from research.mandarin_growth.prepare import choose, parse_moses_zip


class MandarinPreparationTests(unittest.TestCase):
    def test_disjoint_train_dev(self):
        raw=[(f"English sentence number {i} today",
              f"这是第{i}个中文句子，今天很不错。") for i in range(30)]
        train,dev=choose(raw,"en-zh",ban=set(),seed=42,train_count=12,dev_count=8)
        self.assertEqual(len(train),12)
        self.assertEqual(len(dev),8)
        self.assertFalse({x["id"] for x in train}&{x["id"] for x in dev})
        self.assertTrue(all(x["license"]=="CC-BY-2.0-FR" for x in train+dev))

    def test_bad_pair_rejected(self):
        with self.assertRaises(ValueError):
            choose([],"en-de",ban=set(),seed=1)
        with self.assertRaises(ValueError):
            parse_moses_zip(b"garbage","en-de")
