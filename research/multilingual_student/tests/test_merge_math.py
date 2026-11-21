"""Prove a shared LoRA can be folded into one ordinary weight set.

This numerical unit test runs with CPU PyTorch only and does not download the
1.94GB M2M100 model. The real-model parity test is in export_standalone.py.
"""
import unittest

from research.multilingual_student.model import (
    attach_shared_lora,merge_into_one_model,only_adapter_state
)


class TestStandaloneSharedWeightMerge(unittest.TestCase):
    def test_folding_lora_preserves_outputs_and_removes_all_adapters(self):
        import torch
        from torch import nn
        torch.manual_seed(42)

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj=nn.Linear(1024,1024)
                self.v_proj=nn.Linear(1024,1024)

        class EncLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn=Attention()

        class DecLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn=Attention()
                self.encoder_attn=Attention()

        class FakeTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder=nn.Module()
                self.decoder=nn.Module()
                self.encoder.layers=nn.ModuleList([EncLayer(),EncLayer()])
                self.decoder.layers=nn.ModuleList([DecLayer(),DecLayer()])

        class FakeM2M100(nn.Module):
            def __init__(self):
                super().__init__()
                self.model=FakeTransformer()

        net=FakeM2M100()
        meta=attach_shared_lora(net,rank=4,alpha=8,encoder_last=2,decoder_last=2)
        self.assertEqual(len(meta["targets"]),12)
        self.assertEqual(meta["trainable_parameters"],98304)
        self.assertEqual(len(meta["tensor_keys"]),24)
        with torch.no_grad():
            for name,module in net.named_modules():
                if hasattr(module,"lora_B"):
                    module.lora_B.normal_(mean=0,std=.01)
        inp=torch.randn(2,3,1024)
        expected={}
        for name in meta["targets"]:
            expected[name]=net.get_submodule(name)(inp).detach().clone()
        save=only_adapter_state(net)
        self.assertEqual(len(save),24)
        merged=merge_into_one_model(net)
        self.assertIs(merged,net)
        self.assertEqual(len(only_adapter_state(net)),0)
        for name in meta["targets"]:
            layer=net.get_submodule(name)
            self.assertIsInstance(layer,nn.Linear)
            actual=layer(inp)
            self.assertTrue(torch.allclose(actual,expected[name],atol=2e-5,rtol=1e-5),
                            "LoRA merge changed "+name)

    def test_training_new_student_keeps_legacy_models_out_of_merge(self):
        from research.multilingual_student.model import read_config
        cfg=read_config()
        self.assertFalse(cfg["production_enabled"])
        self.assertTrue(any("WMT24" in label for label in cfg["excluded_model_lineage"]))
        self.assertEqual(cfg["model"]["id"],"facebook/m2m100_418M")


if __name__=="__main__":
    unittest.main()
