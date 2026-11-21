"""Fast no-network contract tests for the first genuinely shared M2M100 model."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from research.multilingual_student.model import CONFIG, ROUTES, read_config, route
from research.multilingual_student.data import (
    PAIR_DIRECTIONS, group_valid, hash_groups, load_groups, make_step_schedule,
    normalize, prepare, text_quality
)


class TestOneNeuralModelContract(unittest.TestCase):
    def test_no_previous_legacy_base_mixed_and_mit_pin(self):
        cfg=read_config()
        self.assertEqual(cfg["model"]["id"],"facebook/m2m100_418M")
        self.assertEqual(len(cfg["model"]["revision"]),40)
        self.assertEqual(cfg["model"]["license"],"mit")
        self.assertEqual(len(ROUTES),6)
        self.assertTrue(cfg["model"]["one_shared_tokenizer"])
        self.assertTrue(cfg["model"]["one_shared_pretrained_base"])
        self.assertFalse(cfg["production_enabled"])
        self.assertEqual(cfg["compute"]["max_paid_spend_usd"],0)

    def test_six_routes_share_one_model_but_no_unsupported_language(self):
        self.assertEqual({f"{a}-{b}" for a,b in PAIR_DIRECTIONS},ROUTES)
        self.assertEqual(route("fr-FR","es-MX"),("fr","es"))
        self.assertEqual(route("EN-us","FR"),("en","fr"))
        for src,tgt in (("auto","en"),("zh","en"),("es","es"),("../en","fr"),("","en")):
            with self.subTest(src=src,tgt=tgt),self.assertRaises(ValueError):
                route(src,tgt)

    def test_licensing_and_test_split_cannot_change_without_explicit_check(self):
        original=json.loads(CONFIG.read_text())
        self.assertFalse(original["data"]["final_test_may_be_accessed"])
        self.assertFalse(original["data"]["previous_500_review_examples_used_as_training"])
        self.assertIn("French WMT24 Chat CC-BY-NC trained adapters",
                      original["excluded_model_lineage"])

    def test_noisy_or_misaligned_tri_lang_examples_are_rejected(self):
        base={"en":"Please open the window now","es":"Por favor abre la ventana ahora",
              "fr":"Veuillez ouvrir la fenêtre maintenant"}
        self.assertTrue(group_valid(base))
        self.assertFalse(group_valid({**base,"en":"a"}))
        self.assertFalse(group_valid({**base,"fr":"Please open the window now"}))
        self.assertFalse(group_valid({**base,"es":"write to private@example.com now"}))
        self.assertFalse(group_valid({**base,"en":"Move to 15 after class",
                                     "es":"Muévete al 50 despues de clase"}))
        self.assertEqual(normalize("  HELLO   now "), "hello now")

    def test_data_loader_refuses_any_final_holdout(self):
        with self.assertRaises(ValueError):
            load_groups("test",dataset_loader=lambda *a,**k:[])


class TestDeterministicGroupDisjointness(unittest.TestCase):
    @staticmethod
    def dataset(repo,locale,*,revision,split,trust_remote_code):
        assert split in ("train","validation")
        assert revision=="ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740"
        prefix={"en-US":"Please check the blue schedule at",
                "es-ES":"Por favor revisa el calendario azul en",
                "fr-FR":"Veuillez vérifier le calendrier bleu à"}[locale]
        return [{"id":i,"utt":f"{prefix} {i} today",
                 "intent":"calendar_check", "scenario":"calendar"}
                for i in range(600 if split=="train" else 250)]

    def test_true_six_direction_pairs_and_deterministic_steps(self):
        a,b,manifest=prepare(dataset_loader=self.dataset,exclude_reviewed=False)
        self.assertEqual(set(a),ROUTES)
        self.assertEqual(set(b),ROUTES)
        self.assertEqual(manifest["train_groups"],120)
        self.assertEqual(manifest["development_groups"],12)
        self.assertTrue(manifest["final_test_accessed"] is False)
        self.assertFalse(manifest["semantic_correctness_independently_reviewed"])
        train_ids={r["group_id"] for value in a.values() for r in value}
        dev_ids={r["group_id"] for value in b.values() for r in value}
        self.assertFalse(train_ids & dev_ids)
        first=make_step_schedule(a,seed=2026112101,steps=18,batch_size=2)
        second=make_step_schedule(a,seed=2026112101,steps=18,batch_size=2)
        self.assertEqual(first,second)
        self.assertEqual(len(first),18)
        for route in ROUTES:
            self.assertEqual(sum(row["route"]==route for row in first),3)
        self.assertEqual(first[0]["route"],"es-en")
        for item in first:
            self.assertEqual(len(item["examples"]),2)
        _,_,manifest2=prepare(dataset_loader=self.dataset,exclude_reviewed=False)
        self.assertEqual(manifest["train_sha256"],manifest2["train_sha256"])
        self.assertEqual(manifest["dev_sha256"],manifest2["dev_sha256"])

    def test_original_user_review_stays_out_of_train_and_dev(self):
        # Real pinned 500-row CSV is present in the branch.
        from research.multilingual_student.data import get_reviewed_sources
        review=get_reviewed_sources()
        self.assertGreater(len(review),200)
        self.assertLessEqual(len(review),500)
        a,b,meta=prepare(dataset_loader=self.dataset,exclude_reviewed=True)
        self.assertTrue(meta["reviewed_500_never_train_or_dev"])
        for all_routes in (a,b):
            for rows in all_routes.values():
                self.assertTrue(all(normalize(x["src"]) not in review and
                                    normalize(x["tgt"]) not in review for x in rows))


if __name__=="__main__":
    unittest.main()
