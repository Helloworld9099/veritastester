"""Offline contracts: exact MADLAD base, license, 400+ tokenizer tags, one LoRA."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from research.madlad400.model import (
    adapter_state,attach_shared_lora,config,file_sha256,
    language_input,supported_target_markers,verify_model_card,
    merge_shared_adapter_into_base
)


class FakeTokenizer:
    unk_token_id=0
    def __init__(self):
        self.tags={"<2en>":11,"<2es>":12,"<2fr>":13}
        self.tags.update({f"<2x{i:04d}>":100+i for i in range(410)})
    def get_vocab(self):
        return self.tags
    def convert_tokens_to_ids(self,token):
        return self.tags.get(token,0)


class TestPinnedMADLADBackbone(unittest.TestCase):
    def test_policy_and_no_false_human_claim(self):
        cfg=config()
        self.assertEqual(cfg["model"]["repo_id"],"google/madlad400-3b-mt")
        self.assertEqual(cfg["model"]["revision"],
                         "fa184c675da0b5c9e1c8694fccd4e12e2d422094")
        self.assertEqual(cfg["model"]["license"],"apache-2.0")
        self.assertFalse(cfg["prior_model"]["selected_lora_compatible"])
        self.assertTrue(cfg["prior_model"]["preserve_past_models"])
        self.assertEqual(cfg["budget"]["max_paid_spend_usd"],0)
        self.assertFalse(cfg["acceptance"]["only_promote_after_independent_human_review"] is False)
        self.assertFalse(cfg["dataset"]["final_test_access"])

    def test_upstream_model_revision_and_license_are_pinned(self):
        cfg=config()["model"]
        info=SimpleNamespace(sha=cfg["revision"],cardData={"license":"apache-2.0"})
        mock=SimpleNamespace(model_info=lambda repo_id,revision:info)
        self.assertEqual(verify_model_card(model_api=mock)["license"],"apache-2.0")
        with self.assertRaises(RuntimeError):
            verify_model_card(model_api=SimpleNamespace(
                model_info=lambda *a,**k:SimpleNamespace(
                    sha=cfg["revision"],cardData={"license":"cc-by-nc-4.0"})))
        with self.assertRaises(RuntimeError):
            verify_model_card(model_api=SimpleNamespace(
                model_info=lambda *a,**k:SimpleNamespace(
                    sha="f"*40,cardData={"license":"apache-2.0"})))

    def test_real_marker_contract_cannot_equate_catalog_with_supported_model(self):
        markers=supported_target_markers(FakeTokenizer())
        self.assertGreaterEqual(len(markers),400)
        self.assertEqual(language_input("Hola, amigo","fr",markers),
                         "<2fr> Hola, amigo")
        with self.assertRaises(ValueError):
            language_input("Hi","zzz",markers)
        with self.assertRaises(ValueError):
            language_input("","en",markers)
        class Small:
            unk_token_id=0
            def get_vocab(self):return {"<2en>":1,"<2es>":2,"<2fr>":3}
            def convert_tokens_to_ids(self,x):return self.get_vocab().get(x,0)
        with self.assertRaisesRegex(RuntimeError,"count"):
            supported_target_markers(Small())

    def test_lora_learns_without_mutating_frozen_base(self):
        import torch
        from torch import nn
        torch.manual_seed(2026)
        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.q=nn.Linear(16,16,bias=False)
                self.v=nn.Linear(16,16,bias=False)
        class Sub(nn.Module):
            def __init__(self,name):
                super().__init__()
                setattr(self,name,Attention())
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer=nn.ModuleList((Sub("SelfAttention"),Sub("EncDecAttention")))
        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.block=nn.ModuleList((Block(),Block()))
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder=Decoder()
                self.emb=nn.Embedding(20,16)
        net=Net()
        original=net.decoder.block[-1].layer[0].SelfAttention.q.weight.detach().clone()
        details=attach_shared_lora(net,rank=4,alpha=8)
        self.assertEqual(len(details["targets"]),4)
        self.assertEqual(details["trainable_tensors"],8)
        self.assertTrue(details["all_pretrained_parameters_frozen"])
        layer=net.decoder.block[-1].layer[0].SelfAttention.q
        x=torch.randn(2,16)
        out=layer(x).sum()
        out.backward()
        self.assertIsNone(layer.base.weight.grad)
        self.assertIsNone(net.emb.weight.grad)
        self.assertIsNotNone(layer.lora_B.grad)
        opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=0.1)
        opt.step()
        self.assertTrue(torch.equal(layer.base.weight,original))
        self.assertNotEqual(float(layer.lora_B.abs().sum()),0.0)
        self.assertEqual(len(adapter_state(net)),8)
        # One actual trained adapter can be folded into one standalone
        # neural weight set without silently discarding learned progress.
        probe=torch.randn(3,16)
        net.eval()
        with torch.no_grad():
            before=net.decoder.block[-1].layer[0].SelfAttention.q(probe)
        merged=merge_shared_adapter_into_base(net,rank=4,alpha=8)
        with torch.no_grad():
            after=net.decoder.block[-1].layer[0].SelfAttention.q(probe)
        self.assertEqual(merged["merged_projection_count"],4)
        torch.testing.assert_close(before,after,rtol=1e-5,atol=1e-5)
        self.assertEqual(len(adapter_state(net)),0)

    def test_sha_checksum_is_streaming(self):
        with tempfile.TemporaryDirectory() as p:
            f=Path(p)/"proof.txt"
            f.write_bytes(b"hello")
            self.assertEqual(file_sha256(f),
                             "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824")


if __name__=="__main__":
    unittest.main()
