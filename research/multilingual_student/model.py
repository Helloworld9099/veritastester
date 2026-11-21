"""ONE neural translation model (M2M100) with ONE shared six-direction LoRA.

Distinct from the old Spanish/French Marian *router*. This initializes a
separately pretrained multilingual model and learns one common adapter shared
by en, es, fr in both directions. It never mixes incompatible Marian tensors
or trains on NC-licensed WMT24/Spanish subtitle examples.

All language paths share exactly the same base embeddings, decoder and adapter.
Production remains on the unchanged, previously working provider.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "research/multilingual_student/config.json"
LANGUAGES = frozenset(("en", "es", "fr"))
ROUTES = frozenset(("es-en", "fr-en", "en-es", "en-fr", "es-fr", "fr-es"))


def read_config() -> dict:
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    if data["mode"] != "research_training_only" or data["production_enabled"] is not False:
        raise ValueError("Multilingual training may never change production defaults")
    if (data["model"]["id"] != "facebook/m2m100_418M"
        or data["model"]["revision"] != "55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636"
        or data["model"]["license"].lower() != "mit"):
        raise ValueError("Unapproved or unpinned model base")
    if (data["data"]["license"] != "cc-by-4.0"
        or data["data"]["final_test_may_be_accessed"] is not False
        or data["data"]["previous_500_review_examples_used_as_training"] is not False):
        raise ValueError("Dataset restrictions changed")
    if set(data["directions"]) != ROUTES or len(data["directions"]) != 6:
        raise ValueError("Expected six paths through exactly one shared model")
    if data["compute"]["max_paid_spend_usd"] != 0 or data["compute"]["gpu_hours"] != 0:
        raise ValueError("Paid or GPU compute was not authorized")
    return data


def route(source: str, target: str) -> tuple[str, str]:
    def normalize(value: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("Unknown language cannot be guessed")
        pieces = value.strip().lower().split("-")
        if len(pieces) not in (1,2) or len(pieces[0]) != 2:
            raise ValueError("Language code must be explicit")
        language = pieces[0]
        if language not in LANGUAGES:
            raise ValueError("Unsupported source/target language")
        return language
    src, tgt = normalize(source), normalize(target)
    if src == tgt or src+"-"+tgt not in ROUTES:
        raise ValueError("No multilingual route for source-to-target")
    return src, tgt


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_model_rights():
    from huggingface_hub import HfApi
    spec=read_config()["model"]
    details=HfApi().model_info(repo_id=spec["id"], revision=spec["revision"])
    card_license=(details.cardData or {}).get("license","")
    if details.sha != spec["revision"] or card_license.lower()!="mit":
        raise RuntimeError("Exact upstream model revision or MIT license cannot be verified")
    return {"model_id":spec["id"], "revision":details.sha, "model_card_license":card_license}


def load_shared_model(*, verify_license: bool = True):
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
    spec=read_config()["model"]
    if verify_license:
        verify_model_rights()
    tokenizer=M2M100Tokenizer.from_pretrained(
        spec["id"], revision=spec["revision"], trust_remote_code=False)
    model=M2M100ForConditionalGeneration.from_pretrained(
        spec["id"],revision=spec["revision"], use_safetensors=False,
        weights_only=True, trust_remote_code=False)
    if model.config.model_type!="m2m_100" or model.config.d_model!=1024:
        raise RuntimeError("Unexpected M2M100 pretrained architecture")
    for language in LANGUAGES:
        if not isinstance(tokenizer.get_lang_id(language), int):
            raise RuntimeError("Base tokenizer does not support language "+language)
    return model,tokenizer


def attach_shared_lora(model, *, rank: int = 4, alpha: int = 8,
                       encoder_last: int = 2, decoder_last: int = 2):
    """Install ONE set of LoRA tensors, NOT one adapter per language."""
    import torch
    from torch import nn

    if not 1 <= rank <= 16 or not 1 <= encoder_last <= 3 or not 1 <= decoder_last <= 3:
        raise ValueError("Unapproved adapter budget")

    class LowRankLinear(nn.Module):
        def __init__(self, base: nn.Linear):
            super().__init__()
            self.base=base
            self.rank=rank
            self.alpha=alpha
            self.lora_A=nn.Parameter(torch.empty(rank,base.in_features))
            self.lora_B=nn.Parameter(torch.zeros(base.out_features,rank))
            nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))

        def forward(self,x):
            return self.base(x) + ((x @ self.lora_A.T) @ self.lora_B.T) * (self.alpha/self.rank)

    for p in model.parameters():
        p.requires_grad_(False)
    layers=[]
    enc=model.model.encoder.layers
    dec=model.model.decoder.layers
    for i in range(len(enc)-encoder_last,len(enc)):
        for name in ("q_proj","v_proj"):
            layers.append((f"model.encoder.layers.{i}.self_attn.{name}",
                           getattr(enc[i].self_attn,name)))
    for i in range(len(dec)-decoder_last,len(dec)):
        for att in ("self_attn","encoder_attn"):
            for name in ("q_proj","v_proj"):
                layers.append((f"model.decoder.layers.{i}.{att}.{name}",
                               getattr(getattr(dec[i],att),name)))
    if len(layers) != 2*encoder_last+4*decoder_last:
        raise RuntimeError("Unexpected multilingual model layer shape")
    for name, module in layers:
        if not isinstance(module,nn.Linear):
            raise RuntimeError("LoRA expects a plain Linear module: "+name)
        parent,key=name.rsplit(".",1)
        setattr(model.get_submodule(parent),key,LowRankLinear(module))

    keys=tuple(sorted(k for k in model.state_dict() if ".lora_" in k))
    if len(keys)!=2*len(layers):
        raise RuntimeError("Expected one low-rank tensor pair per target")
    trainable=[p for p in model.parameters() if p.requires_grad]
    n=sum(p.numel() for p in trainable)
    if not 10000 < n < 350000:
        raise RuntimeError("Shared trainable adapter unexpectedly large")
    return {"targets":[name for name,_ in layers],
            "tensor_keys":keys,"trainable_parameters":n,
            "shared_across_all_six_directions":True,
            "rank":rank,"alpha":alpha}


def only_adapter_state(model) -> dict:
    import torch
    return {key:value.detach().cpu().contiguous()
            for key,value in model.state_dict().items() if ".lora_" in key}


def load_adapter(model,adapter_path: Path, *,sha: str):
    import torch
    from safetensors.torch import load_file
    if not adapter_path.is_file() or sha256(adapter_path)!=sha:
        raise RuntimeError("Adapter file not found or hash mismatch")
    weights=load_file(str(adapter_path))
    state=model.state_dict()
    required={key for key in state if ".lora_" in key}
    if set(weights)!=required:
        raise RuntimeError("Shared adapter does not fit the single M2M100 network")
    with torch.no_grad():
        for key,value in weights.items():
            if state[key].shape!=value.shape or not torch.isfinite(value).all():
                raise RuntimeError("Invalid LoRA tensor: "+key)
            state[key].copy_(value)


def generate(model,tokenizer,text:str,source:str,target:str,*,max_length:int=72):
    import torch
    source,target=route(source,target)
    if not isinstance(text,str) or not text.strip():
        raise ValueError("Speech text is required")
    tokenizer.src_lang=source
    encoded=tokenizer(text.strip(),return_tensors="pt",truncation=False)
    if encoded["input_ids"].shape[1]>max_length:
        raise ValueError("Current utterance exceeds checked token budget; do not truncate")
    with torch.no_grad():
        pred=model.generate(**encoded,forced_bos_token_id=tokenizer.get_lang_id(target),
                            num_beams=2,max_new_tokens=72,use_cache=True)
    decoded=tokenizer.decode(pred[0],skip_special_tokens=True).strip()
    if not decoded:
        raise RuntimeError("Unified model returned empty translation")
    return decoded


def merge_into_one_model(model):
    """Remove all LoRA wrappers after mathematically merging them into base.

    The returned M2M100 is one standalone set of weights; when saved using
    save_pretrained(...,safe_serialization=True), all six language directions
    use the same checkpoint WITHOUT a separate runtime adapter.
    """
    import torch
    from torch import nn
    changed=0
    for module_name,module in list(model.named_modules()):
        if (hasattr(module,"lora_A") and hasattr(module,"lora_B")
            and isinstance(getattr(module,"base",None),nn.Linear)):
            parent,key=module_name.rsplit(".",1)
            with torch.no_grad():
                merged = module.lora_B.float() @ module.lora_A.float()
                module.base.weight.add_(merged.to(module.base.weight.dtype) *
                                        (module.alpha/module.rank))
            setattr(model.get_submodule(parent),key,module.base)
            changed+=1
    if changed != 12:
        raise RuntimeError("Expected exactly 12 merged shared LoRA modules")
    if any(".lora_" in key for key in model.state_dict()):
        raise RuntimeError("Standalone model still depends on LoRA wrappers")
    return model
