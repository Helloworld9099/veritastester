"""Network-free regression checks for round3 training and data isolation."""
import math
import unittest

from research.conversation_growth.round3_data import (
    PAIRS, identities, split_samples, directions, settings
)
from research.conversation_growth.round3 import (
    ROUTES, acceptance,compare_scores,learning_rate,make_schedule
)


def synthetic_sources():
    return {p:[
        {"source":f"Source {p} number {i} in a complete sentence.",
         "target":f"Destination {p} number {i} with different wording.",
         "row_id":str(i),"provenance":"offline synthetic only"}
        for i in range(1500)] for p in PAIRS}


class Round3OfflineTests(unittest.TestCase):
    def test_source_split_no_leakage_and_six_routes(self):
        candidates=synthetic_sources()
        candidates["en-fr"][1200]["source"]=candidates["en-es"][3]["source"]
        train,dev=split_samples(candidates)
        self.assertEqual({p:len(v) for p,v in train.items()},{p:800 for p in PAIRS})
        self.assertEqual({p:len(v) for p,v in dev.items()},{p:20 for p in PAIRS})
        all_train={v for rows in train.values() for row in rows for v in identities(row)}
        all_dev={v for rows in dev.values() for row in rows for v in identities(row)}
        previous={v for rows in candidates.values() for row in rows[:350] for v in identities(row)}
        self.assertFalse(all_train & all_dev)
        self.assertFalse(previous & all_dev)
        self.assertEqual(set(directions(dev)),set(ROUTES))
        self.assertNotIn("1200",{r["row_id"] for r in dev["en-fr"]})

    def test_global_forbidden_string_rejected(self):
        candidates=synthetic_sources()
        forbidden={candidates["en-fr"][0]["source"].casefold()}
        train,dev=split_samples(candidates,forbidden=forbidden)
        all_ids={v for rows in train.values() for row in rows for v in identities(row)}
        all_ids.update(v for rows in dev.values() for row in rows for v in identities(row))
        self.assertFalse(all_ids & forbidden)

    def test_fixed_balanced_real_pair_schedule(self):
        cfg,_=settings()
        candidates=synthetic_sources()
        training,_=split_samples(candidates)
        a=make_schedule(training,cfg)
        b=make_schedule(training,cfg)
        self.assertEqual(a,b)
        self.assertEqual(len(a),240)
        self.assertEqual({p:sum(item["route"]==p for item in a) for p in ROUTES},
                         {p:40 for p in ROUTES})
        ids=[row["group_id"] for item in a for row in item["examples"]]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(len(ids),480)

    def test_warmup_and_cosine_decay(self):
        cfg,_=settings()
        values=[learning_rate(s,cfg) for s in range(1,241)]
        self.assertGreater(values[1],values[0])
        self.assertAlmostEqual(values[23],cfg["train"]["max_learning_rate"])
        self.assertGreater(values[23],values[120])
        self.assertAlmostEqual(values[239],cfg["train"]["min_learning_rate"])
        self.assertTrue(all(math.isfinite(x) and x>0 for x in values))

    def test_both_development_gates_mandatory(self):
        cfg,_=settings()
        original={x:{"n":36,"chrF_not_accuracy":50.0,
                       "BLEU_not_accuracy":25.0} for x in ROUTES}
        aligned={x:{"n":20,"chrF_not_accuracy":50.0,
                      "BLEU_not_accuracy":25.0} for x in ROUTES}
        strong_original={x:{"n":36,"chrF_not_accuracy":50.7,
                              "BLEU_not_accuracy":25.5} for x in ROUTES}
        strong_aligned={x:{"n":20,"chrF_not_accuracy":51.0,
                             "BLEU_not_accuracy":26.0} for x in ROUTES}
        good,gates=acceptance(original,strong_original,aligned,strong_aligned,cfg)
        self.assertTrue(good)
        bad,gates=acceptance(original,original,aligned,strong_aligned,cfg)
        self.assertFalse(bad)
        self.assertFalse(gates["massive"]["eligible"])
        bad,gates=acceptance(original,strong_original,aligned,aligned,cfg)
        self.assertFalse(bad)
        self.assertFalse(gates["tatoeba"]["eligible"])
        regression={k:dict(v) for k,v in strong_aligned.items()}
        regression["fr-es"]["chrF_not_accuracy"]=49.4
        self.assertFalse(compare_scores(aligned,regression,label="tatoeba",cfg=cfg)["eligible"])

    def test_no_silent_production_or_sealed_test_changes(self):
        cfg,_=settings()
        self.assertTrue(cfg["restrictions"]["no_paid_compute"])
        self.assertTrue(cfg["restrictions"]["no_production_changes"])
        self.assertTrue(cfg["restrictions"]["no_final_holdout"])
        self.assertTrue(cfg["restrictions"]["no_reviewed_500_training_or_validation"])
        self.assertEqual(cfg["parent"]["sha256"],
                         "a00c7965211853df07ce3fa032a76283dfc984dcdaeac85e03263445029ffb8e")


if __name__=="__main__":
    unittest.main()
