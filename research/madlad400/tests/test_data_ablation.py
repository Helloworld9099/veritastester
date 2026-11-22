"""Data-only experiments cover meaning-critical cases without training on reviews."""
import unittest

from research.madlad400.data_ablation import (
    CATEGORIES,PATTERNS,categories,deterministic_rank,
    quality_candidates,stats,
)


class TestQualityDiversityAblation(unittest.TestCase):
    def test_detect_development_weakness_categories_without_making_gold(self):
        self.assertIn("idiom_or_regional_phrase",
                      categories("I'm down, no big deal.","Está bien, no pasa nada."))
        self.assertIn("explicit_negation",
                      categories("She didn't call me back.","Ella no me llamó."))
        self.assertIn("time_number",
                      categories("The meeting is at 10 PM tomorrow.",
                                 "La reunión es mañana a las 10 PM."))
        self.assertIn("pronoun_reference",
                      categories("She told him about the train.",
                                 "Ella le habló del tren."))
        self.assertEqual(len(CATEGORIES),6)

    def test_real_parallel_candidate_sample_but_not_independent_accuracy(self):
        raw=[
            (f"She did not come back by 5 PM on day {n}.",
             f"Ella no volvió antes de las 5 PM el día {n}.")
            for n in range(190)
        ]
        selected,pool=quality_candidates(raw,"en-es",set(),seed=123)
        self.assertGreaterEqual(len(selected),116)
        self.assertTrue(all(x["license"]=="CC-BY-2.0-FR" for x in selected))
        self.assertTrue(all(x["independent_bilingual_adequacy_verified"] is False
                            and x["eligible_for_unreviewed_context_or_slang_training"] is False
                            for x in selected))
        result=stats(selected)
        self.assertEqual(result["count"],len(selected))
        self.assertGreater(result["category_counts"]["time_number"],0)
        self.assertGreater(result["category_counts"]["explicit_negation"],0)

    def test_development_utterances_and_private_details_cannot_be_selected(self):
        raw=[
            (f"She did not come back by 5 PM on day {n}.",
             f"Ella no volvió antes de las 5 PM el día {n}.")
            for n in range(200)
        ]
        excluded={"she did not come back by 5 pm on day 0."}
        rows,_=quality_candidates(raw,"en-es",excluded,seed=17)
        self.assertTrue(all(x["row_id"]!="0" for x in rows))
        self.assertNotEqual(
            deterministic_rank(17,"en-es",3),
            deterministic_rank(17,"en-es",4))


if __name__=="__main__":
    unittest.main()
