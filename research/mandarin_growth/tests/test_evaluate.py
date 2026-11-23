"""Research-only Mandarin direction safeguards."""
import unittest
from research.mandarin_growth.train import MANDARIN_ROUTES,evaluate_mandarin


class TestMandarinEvaluation(unittest.TestCase):
    def test_only_mandarin_routes(self):
        self.assertEqual(MANDARIN_ROUTES,{"en-zh","zh-en","fr-zh","zh-fr","es-zh","zh-es"})
        with self.assertRaises(ValueError):
            evaluate_mandarin(None,None,{"yue-en":[]},n_per_route=1)
        with self.assertRaises(ValueError):
            evaluate_mandarin(None,None,{},n_per_route=12)
        with self.assertRaises(ValueError):
            evaluate_mandarin(None,None,{x:[] for x in MANDARIN_ROUTES},n_per_route=0)


if __name__=="__main__":
    unittest.main()
