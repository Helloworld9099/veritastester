"""Pinned 400+ language pretrained MADLAD-400 translation research backend.

One T5 encoder/decoder with one shared trainable adapter across directions.
The older M2M100 and language specialists remain untouched. The 3B weights
are downloaded only by the explicit free-compute training workflow.

Requires torch, transformers, safetensors, sentencepiece, huggingface_hub for
real inference/training. Policy and source checks work without these imports.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/"research/madlad400/config.json"
EXPERIMENT=ROOT/"research/madlad400/experiment_001"
TAG=re.compile(r"<2([a-z][a-z0-9_-]{1,15})>\Z")


def config()->dict:
    cfg=json.loads(CONFIG.read_text(encoding="utf-8"))
    model=cfg["model"]
    if (cfg["schema_version"]!=1 or cfg["status"]!="research_only_do_not_activate"
        or model["repo_id"]!="google/madlad400-3b-mt"
        or not re.fullmatch(r"[0-9a-f]{40}",model["revision"])
        or model["license"]!="apache-2.0"
        or cfg["budget"]["max_paid_spend_usd"]!=0
        or cfg["dataset"]["final_test_access"] is not False
        or cfg["dataset"]["reviewed_500_excluded"] is not True
        or cfg["prior_model"]["preserve_past_models"] is not True
        or cfg["prior_model"]["selected_lora_compatible"] is not False):
        raise ValueError("MADLAD research policy/provenance not satisfied")
    return cfg


def file_sha256(path:Path)->str:
    sha=hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda:source.read(4*1024*1024),b""):
            sha.update(block)
    return sha.hexdigest()


def verify_model_card(model_api=None)->dict:
    """Fail closed on exact pinned commit and commercially cleared license."""
    from huggingface_hub import HfApi
    cfg=config()["model"]
    api=model_api or HfApi()
    info=api.model_info(cfg["repo_id"],revision=cfg["revision"])
    card=info.cardData or {}
    license_value=str(card.get("license","")).lower()
    if info.sha!=cfg["revision"] or license_value!="apache-2.0":
        raise RuntimeError("Upstream model revision or license not verified")
    return {"repo_id":cfg["repo_id"],"revision":info.sha,
            "license":license_value,
            "model_card":"https://huggingface.co/"+cfg["repo_id"],
            "weights_from_pretrained_translation_model":True}


def supported_target_markers(tokenizer)->frozenset[str]:
    """Real tokenizer native tags, not a catalog list or language guess."""
    vocab=tokenizer.get_vocab()
    if not isinstance(vocab,dict):
        raise RuntimeError("Unexpected MADLAD tokenizer vocabulary")
    markers={m.group(1) for token in vocab if (m:=TAG.fullmatch(token))}
    if not {"en","es","fr"}.issubset(markers):
        raise RuntimeError("MADLAD tokenizer lacks required EN/ES/FR target markers")
    if len(markers)<400 or len(markers)>800:
        raise RuntimeError("Unexpected MADLAD native target marker count")
    for lang in ("en","es","fr"):
        marker=f"<2{lang}>"
        if tokenizer.convert_tokens_to_ids(marker)==tokenizer.unk_token_id:
            raise RuntimeError("Language marker is an unknown token: "+marker)
    return frozenset(markers)


def language_input(text:str,target:str,allowed:frozenset[str])->str:
    if not isinstance(text,str) or not text.strip() or len(text)>500:
        raise ValueError("Short current utterance text is required")
    if target not in allowed or not re.fullmatch(r"[a-z][a-z0-9_-]{1,15}",target):
        raise ValueError("Native target language is not in pinned model vocabulary")
    return "<2"+target+"> "+text.strip()


def load_backbone(*,check_rights:bool=True,dtype=None):
    import torch
    from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
    cfg=config()
    if check_rights:
        verify_model_card()
    mc=cfg["model"]
    tokenizer=AutoTokenizer.from_pretrained(
        mc["repo_id"],revision=mc["revision"],use_fast=True,
        trust_remote_code=False)
    markers=supported_target_markers(tokenizer)
    train_dtype=dtype or torch.bfloat16
    if train_dtype not in (torch.bfloat16,torch.float32):
        raise ValueError("Pinned research code accepts only CPU bfloat16/float32")
    model=AutoModelForSeq2SeqLM.from_pretrained(
        mc["repo_id"],revision=mc["revision"],dtype=train_dtype,
        low_cpu_mem_usage=True,device_map="cpu",
        use_safetensors=True,trust_remote_code=False)
    if getattr(model.config,"model_type",None)!="t5":
        raise RuntimeError("Unexpected MADLAD neural architecture")
    model.eval()
    return model,tokenizer,markers


def attach_shared_lora(model,*,rank:int=4,alpha:int=8):
    """Freeze pretrained weights; add one small adapter to final decoder block.

    Only 4 decoder attention q/v projections carry learned parameters;
    still a single shared learned set across all native target languages.
    """
    import torch
    from torch import nn
    if not 1<=rank<=16 or not 1<=alpha<=64:
        raise ValueError("LoRA shape/scale out of bounded research configuration")
    if not hasattr(model,"decoder") or not hasattr(model.decoder,"block"):
        raise TypeError("Expected actual T5 model decoder blocks")
    for p in model.parameters():
        p.requires_grad_(False)
    last=model.decoder.block[-1]
    targets=[
        ("decoder.block."+str(len(model.decoder.block)-1)+".layer.0.SelfAttention.q",
         last.layer[0].SelfAttention,"q"),
        ("decoder.block."+str(len(model.decoder.block)-1)+".layer.0.SelfAttention.v",
         last.layer[0].SelfAttention,"v"),
        ("decoder.block."+str(len(model.decoder.block)-1)+".layer.1.EncDecAttention.q",
         last.layer[1].EncDecAttention,"q"),
        ("decoder.block."+str(len(model.decoder.block)-1)+".layer.1.EncDecAttention.v",
         last.layer[1].EncDecAttention,"v"),
    ]
    class LoRALinear(nn.Module):
        def __init__(self,base:nn.Linear):
            super().__init__()
            self.base=base
            self.lora_A=nn.Parameter(torch.empty(rank,base.in_features,dtype=torch.float32))
            self.lora_B=nn.Parameter(torch.zeros(base.out_features,rank,dtype=torch.float32))
            nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))
        def forward(self,x):
            base=self.base(x)
            delta=((x.float() @ self.lora_A.t()) @ self.lora_B.t())*(alpha/rank)
            return base+delta.to(base.dtype)
    for name,parent,attr in targets:
        current=getattr(parent,attr)
        if not isinstance(current,nn.Linear):
            raise RuntimeError("Expected frozen T5 attention linear at "+name)
        setattr(parent,attr,LoRALinear(current))
    tensors={k:v for k,v in model.named_parameters() if v.requires_grad}
    expected=sum(p.numel() for p in tensors.values())
    if len(tensors)!=8 or not 5000<expected<150000:
        raise RuntimeError("Model adapter is not the intended small shared LoRA")
    return {"targets":[name for name,_,_ in targets],
            "rank":rank,"alpha":alpha,"trainable_tensors":len(tensors),
            "trainable_parameters":expected,
            "all_pretrained_parameters_frozen":True,
            "all_native_languages_share_adapter":True}


def adapter_state(model):
    return {name:tensor.detach().float().cpu().contiguous()
            for name,tensor in model.state_dict().items()
            if name.endswith(".lora_A") or name.endswith(".lora_B")}


def translate_short(model,tokenizer,text,source,target,markers,*,max_source_tokens=48):
    """Research-only inference: target prefix controls MADLAD generation."""
    import torch
    if source not in markers or target not in markers or source==target:
        raise ValueError("MADLAD requires two distinct native language identifiers")
    prefix=language_input(text,target,markers)
    encoded=tokenizer(prefix,return_tensors="pt",truncation=False)
    if encoded["input_ids"].shape[1]>max_source_tokens:
        raise ValueError("Do not silently truncate source/ASR utterances")
    start=__import__("time").monotonic()
    model.eval()
    with torch.inference_mode():
        output=model.generate(**encoded,num_beams=1,do_sample=False,
                              max_new_tokens=36,use_cache=True)
    result=tokenizer.decode(output[0],skip_special_tokens=True).strip()
    if not result:
        raise RuntimeError("MADLAD translation returned empty result")
    return {"text":result,"seconds":__import__("time").monotonic()-start,
            "target_marker":"<2"+target+">",
            "independent_human_accuracy_verified":False}
