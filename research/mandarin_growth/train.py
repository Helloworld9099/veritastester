#!/usr/bin/env python3
"""Train one shared M2M100 adapter on Mandarin + EN/ES/FR replay.

Research only. Preserve existing rank-4 parent, freeze its parameters, learn
new rank-8 slices. Evaluate all twelve routes; no automatic release.
"""
from __future__ import annotations
import hashlib,json,math,random,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.conversation_growth.round2 import embed_existing_rank4,lock_previously_learned_ranks,preserved_slices
from research.conversation_growth.data import build_new_parallel_training,sampled_dev,normalize
from research.multilingual_student.model import ROOT as MODEL_ROOT,read_config,load_shared_model,attach_shared_lora,sha256,only_adapter_state
from research.multilingual_student.train import encode_for_loss,evaluate,ensure_old_winners_untouched
from research.mandarin_growth.prepare import OUTPUT as DATA_OUT,prepare as prepare_mandarin

OUT=ROOT/"research/mandarin_growth/experiment_002"
PARENT=ROOT/"research/multilingual_student/experiment_002/shared_lora_extra20.safetensors"
PARENT_SHA="a00c7965211853df07ce3fa032a76283dfc984dcdaeac85e03263445029ffb8e"
ROUTES=("en-es","es-en","en-fr","fr-en","es-fr","fr-es","en-zh","zh-en","es-zh","zh-es","fr-zh","zh-fr")


def rows_from_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


MANDARIN_ROUTES = frozenset(("en-zh","zh-en","es-zh","zh-es","fr-zh","zh-fr"))


def evaluate_mandarin(model,tokenizer,dev,*,n_per_route=12):
    """Research-only true Mandarin evaluation without changing six-route app API.

    The older shared-model generate() intentionally rejects zh. This local
    evaluator exercises the native pinned M2M100 tokenizer's explicit zh ID.
    It NEVER silently aliases Cantonese, other Chinese varieties or unknown
    locale tags to Mandarin.
    """
    if set(dev) != MANDARIN_ROUTES or n_per_route < 1:
        raise ValueError("Only six explicit Mandarin development directions allowed")
    import torch
    from sacrebleu.metrics import CHRF,BLEU
    results={}
    model.eval()
    with torch.inference_mode():
        for route in sorted(dev):
            source,target=route.split("-")
            if source not in ("en","es","fr","zh") or target not in ("en","es","fr","zh"):
                raise ValueError("Unsupported source/target code")
            if not isinstance(tokenizer.get_lang_id(target),int):
                raise RuntimeError("Pinned M2M100 has no tokenizer id for "+target)
            samples=dev[route][:n_per_route]
            if len(samples)!=n_per_route:
                raise RuntimeError("Incomplete Mandarin development set: "+route)
            predictions=[]
            for row in samples:
                tokenizer.src_lang=source
                encoded=tokenizer(row["src"],return_tensors="pt",truncation=False)
                if int(encoded["input_ids"].shape[-1])>72:
                    raise ValueError("Mandarin development source exceeds 72-token safety bound")
                generated=model.generate(**encoded,
                    forced_bos_token_id=tokenizer.get_lang_id(target),
                    num_beams=2,max_new_tokens=72,use_cache=True)
                text=tokenizer.decode(generated[0],skip_special_tokens=True).strip()
                if not text:
                    raise RuntimeError("No translated text from shared Mandarin model")
                predictions.append(text)
            references=[row["tgt"] for row in samples]
            results[route]={
                "n":len(samples),
                "chrF_not_accuracy":round(CHRF().corpus_score(predictions,[references]).score,4),
                "BLEU_not_accuracy":round(BLEU().corpus_score(predictions,[references]).score,4),
                "empty_predictions":0,
                "source_and_reference_not_in_report":True
            }
            print("NATIVE_MANDARIN_DEV_"+route.upper().replace("-","_"),
                  json.dumps(results[route]),flush=True)
    return results


def schedule(mandarin,old,seed=2026112204):
    rng=random.Random(seed)
    schedule=[]
    for route in ROUTES:
        src,tgt=route.split("-")
        base=src+"-"+tgt if src<tgt else tgt+"-"+src
        if "zh" in route:
            # Dataset source orientation always <non-Chinese>-zh.
            base=next(x+"-zh" for x in ("en","es","fr") if x in route)
            pool=[r for r in mandarin if r["pair"]==base]
            steps=8
        else:
            pool=old[base]
            steps=4
        rng.shuffle(pool)
        if len(pool)<steps*2:
            raise RuntimeError("Insufficient unique licensed parallel rows for "+route)
        for i in range(steps):
            items=[]
            for r in pool[2*i:2*i+2]:
                if src==base.split("-")[0]:
                    a,b=r["source"],r["target"]
                else:
                    a,b=r["target"],r["source"]
                items.append({"src":a,"tgt":b,"group_id":base+":"+str(r.get("id",r.get("row_id")))})
            schedule.append((route,items))
    rng.shuffle(schedule)
    return schedule


def run():
    import torch
    from safetensors.torch import load_file,save_file
    if OUT.exists():
        raise FileExistsError("Do not overwrite a training run")
    if sha256(PARENT)!=PARENT_SHA:
        raise RuntimeError("Selected previous shared adapter missing or modified")
    old_winners=ensure_old_winners_untouched()
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    torch.manual_seed(2026112204)
    random.seed(2026112204)
    if not (DATA_OUT/"manifest.json").exists():
        prepare_mandarin()
    manifest=json.loads((DATA_OUT/"manifest.json").read_text())
    if manifest["base_model"]!="facebook/m2m100_418M" or manifest["no_final_test_used"] is not True:
        raise RuntimeError("Unapproved Mandarin data")
    mandarin=rows_from_jsonl(DATA_OUT/"train.jsonl")
    mandarin_dev=rows_from_jsonl(DATA_OUT/"dev.jsonl")
    old,old_manifest=build_new_parallel_training()
    old_dev,_=sampled_dev(36)
    old_dev_norm={normalize(v) for rows in old_dev.values() for r in rows for v in (r["src"],r["tgt"])}
    for r in mandarin+mandarin_dev:
        if normalize(r["source"]) in old_dev_norm or normalize(r["target"]) in old_dev_norm:
            raise RuntimeError("Mandarin data leaks old frozen DEV")
    plan=schedule(mandarin,old)
    model,tokenizer=load_shared_model()
    if tokenizer.get_lang_id("zh") is None:
        raise RuntimeError("Pinned model lacks Mandarin")
    cfg=read_config()["training"]
    architecture=attach_shared_lora(model,rank=8,alpha=16,
                                    encoder_last=cfg["last_encoder_layers"],
                                    decoder_last=cfg["last_decoder_layers"])
    original=load_file(str(PARENT))
    embed_existing_rank4(model,original)
    lock_previously_learned_ranks(model)
    old_baseline=evaluate(model,tokenizer,old_dev,n_per_route=36)
    def chinese_dev():
        routes={}
        for r in mandarin_dev:
            src,tgt=r["pair"].split("-")
            routes.setdefault(src+"-zh",[]).append({"src":r["source"],"tgt":r["target"]})
            routes.setdefault("zh-"+src,[]).append({"src":r["target"],"tgt":r["source"]})
        return routes
    zh_dev=chinese_dev()
    zh_baseline=evaluate_mandarin(model,tokenizer,zh_dev,n_per_route=12)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                          lr=0.00002,weight_decay=0)
    records=[]
    checkpoints={}
    for i,(route,items) in enumerate(plan,1):
        src,tgt=route.split("-")
        inputs,labels=encode_for_loss(tokenizer,items,src=src,tgt=tgt,max_tokens=96)
        model.train()
        opt.zero_grad(set_to_none=True)
        loss=model(**inputs,labels=labels).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
        loss.backward()
        opt.step()
        records.append({"step":i,"route":route,"loss":float(loss)})
        print("ACTUAL_MULTILINGUAL_CONTEXT_BASELINE_UPDATE",json.dumps(records[-1]),flush=True)
        if i not in (24,48,72):
            continue
        preserved_slices(model,original)
        OUT.mkdir(parents=True,exist_ok=True)
        file=OUT/f"shared_mandarin_rank8_step{i:02d}.safetensors"
        save_file(only_adapter_state(model),str(file))
        old_after=evaluate(model,tokenizer,old_dev,n_per_route=36)
        zh_after=evaluate_mandarin(model,tokenizer,zh_dev,n_per_route=12)
        old_deltas={k:round(old_after[k]["chrF_not_accuracy"]-old_baseline[k]["chrF_not_accuracy"],4) for k in old_dev}
        zh_deltas={k:round(zh_after[k]["chrF_not_accuracy"]-zh_baseline[k]["chrF_not_accuracy"],4) for k in zh_dev}
        eligible=all(v>=-1 for v in old_deltas.values()) and sum(zh_deltas.values())>0 and all(v>=-1 for v in zh_deltas.values())
        checkpoints[file.name]={"sha256":sha256(file),"old_deltas_chrF_not_accuracy":old_deltas,
                                "mandarin_deltas_chrF_not_accuracy":zh_deltas,
                                "eligible_research_only":eligible}
        print("TWELVE_ROUTE_GATE",json.dumps({"step":i,"eligible":eligible,"old":old_deltas,"mandarin":zh_deltas}),flush=True)
    winner=next((k for k in reversed(list(checkpoints)) if checkpoints[k]["eligible_research_only"]),None)
    report={"status":"trained_shared_M2M100_Mandarin_research_only",
            "optimizer_updates":len(records),"routes":dict(Counter(x["route"] for x in records)),
            "original_parent_sha256":PARENT_SHA,"original_parent_unchanged":sha256(PARENT)==PARENT_SHA,
            "legacy_champions_unchanged":ensure_old_winners_untouched()==old_winners,
            "architecture":architecture,"mandarin_data_manifest":manifest,
            "old_baseline":old_baseline,"mandarin_baseline":zh_baseline,
            "checkpoints":checkpoints,
            "selected_research_checkpoint":winner,
            "release_approved":False,"human_95_percent_verified":False,
            "actual_multi_turn_context_training":False,"actual_slang_supervision":False,
            "final_holdout_used":False,
            "warning":"Tatoeba sentences are not genuine multi-turn context or verified slang. Do not claim those capabilities were trained."}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("MANDARIN_SHARED_TRAINING_COMPLETED",json.dumps({"steps":len(records),"research_winner":winner}),flush=True)


if __name__=="__main__":
    run()
