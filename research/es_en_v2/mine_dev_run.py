#!/usr/bin/env python3
"""ACTUAL dev-only error mining for licensed Spanish LoRA continuation.

Score the pinned publisher-validation cohort only. Compare parent with
already-trained stage-3 continuation, emit aggregated error distributions and
hashed case IDs (not private/user text). No test/final holdout is opened and
no human semantics are claimed from automatic flags.
"""
from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"experiments/cpu_continuation_20261117"),
              str(ROOT/"research/es_en_v2")]
import train as core
from train_three import aligned,vetted_rows
from dev_mining import mine_dev,issue_tags

OLD_PATH=ROOT/"experiments/massive_stage3_20261117/saved/spanish/adapter_selected.safetensors"
NEW_PATH=ROOT/"experiments/es_en_v2_runs/candidate_after_three.safetensors"
OLD_SHA="14dddb5040afe49f8f05a1bfba3458d3321524b53b0fc05b359ad41a1ab75a4b"
NEW_SHA="3bf849bb9a4377dcb9f937cdc7750a7f26616fc1c58cf703cef0ba8e46430a71"


def run(out: Path):
    if out.exists():
        raise FileExistsError("Development-error receipt exists already")
    if core.sha256(OLD_PATH)!=OLD_SHA or core.sha256(NEW_PATH)!=NEW_SHA:
        raise RuntimeError("Pinned parent/new Spanish adaptation changed")
    summary=json.loads((ROOT/"experiments/es_en_v2_runs/summary.json").read_text())
    if summary["final_dev_candidate_sha256"]!=NEW_SHA or summary["successful_rounds"]!=3:
        raise RuntimeError("Candidate chain is not finalized")
    import torch
    from torch import nn
    from transformers import MarianMTModel,MarianTokenizer
    from safetensors.torch import load_file
    from sacrebleu.metrics import BLEU,CHRF

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    source=core.LANGUAGES["spanish"]
    base=ROOT/"research/es_en_v2/.temporary_dev_mining_base"
    core.download_base(base,ROOT/"research"/source["manifest_rel"],source)
    tok=MarianTokenizer.from_pretrained(str(base),local_files_only=True)
    model=MarianMTModel.from_pretrained(str(base),local_files_only=True,
                                       weights_only=True,use_safetensors=source["use_safetensors"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    class LoRA(nn.Module):
        def __init__(self,layer):
            super().__init__()
            self.base=layer
            self.lora_A=nn.Parameter(torch.zeros(8,layer.in_features))
            self.lora_B=nn.Parameter(torch.zeros(layer.out_features,8))
        def forward(self,x):
            return self.base(x)+((x@self.lora_A.T)@self.lora_B.T)*2.0

    protocol=json.loads((ROOT/"research"/source["protocol_rel"]).read_text())
    for name in protocol["target_modules"]:
        module=model.get_submodule(name)
        if not isinstance(module,nn.Linear):
            raise RuntimeError("Unsupported LoRA layer")
        parent,attr=name.rsplit(".",1)
        setattr(model.get_submodule(parent),attr,LoRA(module))
    state=model.state_dict()
    keys={k for k in state if ".lora_" in k}

    def attach(path):
        weights=load_file(str(path))
        if set(weights)!=keys:
            raise RuntimeError("Adapter key mismatch")
        with torch.no_grad():
            for k,value in weights.items():
                if value.shape!=state[k].shape or not torch.isfinite(value).all():
                    raise RuntimeError("Invalid adapter value")
                state[k].copy_(value)
        model.eval()

    english=aligned("en-US","validation")
    spanish=aligned("es-ES","validation")
    rows,_=vetted_rows(spanish,english,2.5)
    rows=sorted(rows,key=lambda x:hashlib.sha256(("v2dev"+x[0]).encode()).hexdigest())[:128]
    if len(rows)!=128:
        raise RuntimeError("Frozen development selection changed")
    dev_id_sha=hashlib.sha256("\n".join(row[0] for row in rows).encode()).hexdigest()
    srcs=[r[1] for r in rows]
    refs=[r[2] for r in rows]

    def translations():
        preds=[]
        with torch.no_grad():
            for i in range(0,len(srcs),8):
                enc=tok(srcs[i:i+8],return_tensors="pt",padding=True,
                        truncation=True,max_length=128)
                sequences=model.generate(**enc,num_beams=4,max_new_tokens=96)
                preds.extend(tok.batch_decode(sequences,skip_special_tokens=True))
        return preds

    score={}
    outputs={}
    start=time.monotonic()
    for title,path in (("parent",OLD_PATH),("after_three",NEW_PATH)):
        attach(path)
        predictions=translations()
        outputs[title]=predictions
        mined=mine_dev([
            {"case_id":case_id,"split":"dev","source":spanish,
             "reference":english,"hypothesis":p,
             "source_language":"es","target_language":"en",
             "license_id":"CC-BY-4.0","provenance_url":"https://huggingface.co/datasets/AmazonScience/massive"}
            for (case_id,spanish,english),p in zip(rows,predictions)
        ],max_cases=128)
        score[title]={
            "chrf_not_accuracy":CHRF().corpus_score(predictions,[refs]).score,
            "bleu_not_accuracy":BLEU().corpus_score(predictions,[refs]).score,
            "suspected_error_tag_counts":mined["tag_counts"],
            "suspected_flagged_rows":mined["flagged_n"],
            "human_confirmed_error_count":0,
            "worst_case_id_hashes":[hashlib.sha256(p["case_id"].encode()).hexdigest()
                                    for p in mined["ranked_cases"][:12]],
        }
    modified=0
    better=0
    worse=0
    chrf=CHRF()
    for a,b,ref in zip(outputs["parent"],outputs["after_three"],refs):
        old=chrf.sentence_score(a,[ref]).score
        new=chrf.sentence_score(b,[ref]).score
        modified+=a!=b
        better+=new>old
        worse+=new<old
    report={
        "status":"actual_dev_only_translation_mining",
        "parent_sha256":OLD_SHA,"candidate_sha256":NEW_SHA,
        "development_origin":"MASSIVE official validation CC BY 4.0",
        "development_group_id_sha256":dev_id_sha,
        "development_cases":len(srcs),
        "model_scores":score,
        "changed_translations_n":modified,
        "changed_reference_similarity_improved_n":better,
        "changed_reference_similarity_worsened_n":worse,
        "elapsed_seconds":time.monotonic()-start,
        "rss_peak_kib_linux":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "heldout_or_test_read":False,
        "human_semantic_error_labels":0,
        "research_only":True,
        "warning":"No true contextual group/dialogue or slang human evaluation "
                  "exists in this author validation set. Error flags are heuristics.",
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"dev_error_mining.json").write_text(json.dumps(report,indent=2))
    print("REAL_SPANISH_DEV_MINING",json.dumps(report),flush=True)


if __name__=="__main__":
    run(ROOT/"research/es_en_v2/dev_mining_receipt")
