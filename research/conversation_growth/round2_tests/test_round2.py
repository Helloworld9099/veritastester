"""Small offline tests for safe shared adapter expansion and disjoint route samples."""
import unittest
from research.conversation_growth.round2 import (
    config, schedule_batches, lock_previously_learned_ranks,
)


class TestRound2(unittest.TestCase):
    def test_six_routes_no_reused_parallel_group_ids(self):
        cfg=config()
        corpora={}
        for pair in ("en-es","en-fr","es-fr"):
            corpora[pair]=[{
                "row_id":str(i),"source":f"Sentence {i} in {pair}",
                "target":f"Equivalent {i} of {pair}",
                "license":"CC-BY-2.0-FR","provenance":"fixed offline fixture"
            } for i in range(40)]
        a=schedule_batches(corpora,cfg)
        b=schedule_batches(corpora,cfg)
        self.assertEqual(a,b)
        self.assertEqual(len(a),48)
        self.assertEqual(len({x["optimizer_step"] for x in a}),48)
        self.assertEqual(set(x["route"] for x in a),
                         {"en-es","es-en","en-fr","fr-en","es-fr","fr-es"})
        all_group_ids=[x["group_id"] for step in a for x in step["items"]]
        self.assertEqual(len(all_group_ids),len(set(all_group_ids)))
        self.assertTrue(all(step["origin"]=="Tatoeba_CC-BY-2.0-FR_parallel" for step in a))

    def test_rank_growth_is_function_preserving(self):
        import torch
        torch.manual_seed(15)
        x=torch.randn(5,7)
        a=torch.randn(4,7)
        b=torch.randn(6,4)
        new_a=torch.cat([a,torch.randn(4,7)],dim=0)
        new_b=torch.cat([b,torch.zeros(6,4)],dim=1)
        old=(x @ a.T) @ b.T * (8/4)
        expanded=(x @ new_a.T) @ new_b.T * (16/8)
        torch.testing.assert_close(old,expanded,rtol=1e-6,atol=1e-6)

    def test_existing_rank_masked(self):
        import torch
        from torch import nn
        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A=nn.Parameter(torch.ones(8,3))
                self.lora_B=nn.Parameter(torch.ones(5,8))
        # 12 wrappers are required by the real model.
        fake=nn.Module()
        fake.layers=nn.ModuleList([Layer() for _ in range(12)])
        lock_previously_learned_ranks(fake)
        loss=sum(layer.lora_A.sum()+layer.lora_B.sum() for layer in fake.layers)
        loss.backward()
        for layer in fake.layers:
            self.assertTrue(torch.count_nonzero(layer.lora_A.grad[:4])==0)
            self.assertTrue(torch.count_nonzero(layer.lora_B.grad[:,:4])==0)
            self.assertTrue(torch.count_nonzero(layer.lora_A.grad[4:])>0)
            self.assertTrue(torch.count_nonzero(layer.lora_B.grad[:,4:])>0)

    def test_no_access_to_test_or_reviewed_pseudogold(self):
        cfg=config()
        self.assertFalse(cfg["release_approved"])
        self.assertFalse(cfg["final_holdout_access"])
        self.assertEqual(cfg["paid_spend_cap_usd"],0)
        self.assertTrue(cfg["restrictions"]["forbid_retraining_on_reviewer_500"])
        self.assertTrue(cfg["restrictions"]["do_not_train_massive_intent_aligned_utterances_as_literal_gold"])


if __name__=="__main__":
    unittest.main()
