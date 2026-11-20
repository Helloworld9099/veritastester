#!/usr/bin/env python3
"""REAL inference on 40 explicitly synthetic bilingual development stress tests.

Tests context/slang/idiom sensitivity for research diagnosis. Agent-written
references are NOT human truth; chrF here is not meaning accuracy. No final
holdout and no production model changes.
"""
from __future__ import annotations

import gc
import hashlib
import json
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"experiments/cpu_continuation_20261117"),
              str(ROOT/"research/es_en_v2")]
import train as core
from dev_mining import issue_tags

ES_ADAPTER=ROOT/"experiments/es_en_v2_runs/candidate_after_three.safetensors"
ES_ADAPTER_SHA="3bf849bb9a4377dcb9f937cdc7750a7f26616fc1c58cf703cef0ba8e46430a71"
EN_BASE_SHA="5bc4493d463cf000c1f0b50f8d56886a392ed4ab"
EN_BASE="Helsinki-NLP/opus-mt-en-es"


def cases():
    path=ROOT/"research/es_en_v2/dev_stress_slang_context.jsonl"
    rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if len(rows)!=40 or {r["direction"] for r in rows}!={"es-en","en-es"}:
        raise RuntimeError("Synthetic stress corpus changed")
    if any(r.get("split")!="dev" or r.get("human_verified") is not False or
           r.get("final_holdout_eligible") is not False for r in rows):
        raise RuntimeError("Final holdout or human-verified status falsely asserted")
    if len({r["case_id"] for r in rows})!=len(rows):
        raise RuntimeError("Duplicate stress case")
    return rows,hashlib.sha256(path.read_bytes()).hexdigest()


def load(spanish_to_english):
    import torch
    from torch import nn
    from safetensors.torch import load_file
    from transformers import MarianMTModel,MarianTokenizer
    from huggingface_hub import snapshot_download

    if spanish_to_english:
        spec=core.LANGUAGES["spanish"]
        path=ROOT/"research/es_en_v2/.stress_base_es"
        core.download_base(path,ROOT/"research"/spec["manifest_rel"],spec)
        use_safetensors=spec["use_safetensors"]
    else:
        path=Path(snapshot_download(
            repo_id=EN_BASE,revision=EN_BASE_SHA,
            allow_patterns=["config.json","generation_config.json","pytorch_model.bin",
                            "source.spm","target.spm","vocab.json",
                            "tokenizer_config.json","special_tokens_map.json"]))
        # Every file is checked against previous successful original-weight
        # baseline report rather than trusting a moved model tag.
        baseline=json.loads((ROOT/"research/es_en_v2/benchmarks/en_es/"
                             "en_es_original_baseline.json").read_text())
        if baseline["model_sha"]!=EN_BASE_SHA:
            raise RuntimeError("Original EN->ES version changed")
        for filename,sha in baseline["model_file_sha256"].items():
            p=path/filename
            if not p.is_file() or core.sha256(p)!=sha:
                raise RuntimeError("EN->ES model file checksum mismatch")
        use_safetensors=False
    tok=MarianTokenizer.from_pretrained(str(path),local_files_only=True)
    model=MarianMTModel.from_pretrained(str(path),local_files_only=True,
                                       use_safetensors=use_safetensors,weights_only=True)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    def attach_es():
        if not spanish_to_english:
            raise RuntimeError("No EN->ES adapter has been trained yet")
        if core.sha256(ES_ADAPTER)!=ES_ADAPTER_SHA:
            raise RuntimeError("New ES->EN candidate SHA mismatch")
        spec=core.LANGUAGES["spanish"]
        protocol=json.loads((ROOT/"research"/spec["protocol_rel"]).read_text())
        class LoRA(nn.Module):
            def __init__(self,linear):
                super().__init__()
                self.base=linear
                self.lora_A=nn.Parameter(torch.zeros(8,linear.in_features))
                self.lora_B=nn.Parameter(torch.zeros(linear.out_features,8))
            def forward(self,x):
                return self.base(x)+((x@self.lora_A.T)@self.lora_B.T)*2.0
        for name in protocol["target_modules"]:
            old=model.get_submodule(name)
            if not isinstance(old,nn.Linear):
                raise RuntimeError("Unexpected LoRA module")
            parent,key=name.rsplit(".",1)
            setattr(model.get_submodule(parent),key,LoRA(old))
        weights=load_file(str(ES_ADAPTER))
        state=model.state_dict()
        expected={key for key in state if ".lora_" in key}
        if set(weights)!=expected:
            raise RuntimeError("Adapter incompatible")
        with torch.no_grad():
            for key,val in weights.items():
                if state[key].shape!=val.shape or not torch.isfinite(val).all():
                    raise RuntimeError("Bad adapter tensor")
                state[key].copy_(val)
        model.eval()

    def predict(inputs):
        import torch
        output=[]
        with torch.no_grad():
            for offset in range(0,len(inputs),4):
                x=tok(inputs[offset:offset+4],return_tensors="pt",padding=True,
                      truncation=True,max_length=128)
                generated=model.generate(**x,num_beams=4,max_new_tokens=96)
                output.extend(tok.batch_decode(generated,skip_special_tokens=True))
        return output
    return predict,attach_es


def evaluate_arm(rows,preds):
    from sacrebleu.metrics import CHRF,BLEU
    if len(rows)!=len(preds):
        raise ValueError("Prediction count changed")
    refs=[r["reference_synthetic"] for r in rows]
    scores={"chrf_synthetic_reference_not_accuracy":CHRF().corpus_score(preds,[refs]).score,
            "bleu_synthetic_reference_not_accuracy":BLEU().corpus_score(preds,[refs]).score,
            "n":len(rows)}
    grouped=defaultdict(lambda:([],[]))
    for row,pred in zip(rows,preds):
        a,b=grouped[row["category"]]
        a.append(pred);b.append(row["reference_synthetic"])
    scores["category_chrF_synthetic_not_accuracy"]={
        cat:{"n":len(pred),"chrF":CHRF().corpus_score(pred,[ref]).score}
        for cat,(pred,ref) in sorted(grouped.items())}
    scores["suspicious_development_error_tags"]={}
    counts=defaultdict(int)
    for row,pred in zip(rows,preds):
        for key in issue_tags(row["source"],row["reference_synthetic"],pred,
                              source_language=row["direction"][:2],
                              target_language=row["direction"][-2:],
                              prior_source=row["previous_source"]):
            counts[key]+=1
    scores["suspicious_development_error_tags"]=dict(counts)
    return scores


def run(out:Path):
    if out.exists():
        raise FileExistsError("Never overwrite actual stress testing receipts")
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    tests,sha=cases()
    start=time.monotonic()
    results={}
    rendered=[]
    for direction in ("es-en","en-es"):
        subset=[row for row in tests if row["direction"]==direction]
        predict,attach=load(direction=="es-en")
        plain=predict([row["source"] for row in subset])
        results[direction]={"original_base_no_context":evaluate_arm(subset,plain)}
        outputs={"original_base_no_context":plain}
        if direction=="es-en":
            attach()
            previous=predict([row["source"] for row in subset])
            contextual=predict([
                row["source"]+" | contexto previo: "+" / ".join(row["previous_source"][-2:])
                for row in subset])
            results[direction]["trained_v2_no_context"]=evaluate_arm(subset,previous)
            results[direction]["trained_v2_context_injected"]=evaluate_arm(subset,contextual)
            outputs["trained_v2_no_context"]=previous
            outputs["trained_v2_context_injected"]=contextual
            # Same exact utterance with different past context: does it change?
            def idx(which):
                return next(i for i,row in enumerate(subset) if row["case_id"]==which)
            a,b=idx("es_concert"),idx("es_storm")
            results[direction]["same_source_ambiguous_context_outputs_differ"]=\
                contextual[a]!=contextual[b]
        for i,row in enumerate(subset):
            rendered.append({
                "case_id":row["case_id"],"direction":direction,
                "category":row["category"],
                "original_synthetic_development_source":row["source"],
                "previous_source_only":row["previous_source"],
                "agent_authored_possible_reference_not_human_gold":row["reference_synthetic"],
                "hypotheses_for_dev_inspection":{k:outputs[k][i] for k in outputs},
                "not_human_quality_score":True,
            })
        del predict,attach
        gc.collect()
    report={
        "status":"real_context_slang_development_translation_experiment",
        "dev_stress_fixture_sha256":sha,
        "source_human_authored":False,
        "total_synthetic_development_cases":40,
        "spanish_to_english_model_sha256":ES_ADAPTER_SHA,
        "english_to_spanish_model_sha256":EN_BASE_SHA,
        "scores":results,
        "illustrative_translations_not_final_gold":rendered,
        "elapsed_seconds":time.monotonic()-start,
        "peak_rss_kib_linux":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "human_adequacy_95_verified":False,
        "test_or_final_holdout_used":False,
        "not_training_data":True,
        "research_only":True,
        "disclaimer":"All references authored by an AI assistant; natural variants "
                     "may score poorly under chrF despite conveying the right meaning. "
                     "No real-world meaning-preservation accuracy can be inferred.",
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"stress_development_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print("REAL_SPANISH_CONTEXT_SLANG_DEVELOPMENT_ABLATION",json.dumps({
        "scores":results,"elapsed_seconds":report["elapsed_seconds"],
        "human_adequacy_95_verified":False},ensure_ascii=False),flush=True)


if __name__=="__main__":
    run(ROOT/"research/es_en_v2/context_stress_receipt")
