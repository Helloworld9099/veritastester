#!/usr/bin/env python3
"""Real M2M100 tokenizer inspection + 8k-language catalog coverage audit.

This downloads only the pinned tokenizer, NOT the 1.94 GB model weights.
Native vocabulary != trained, reviewed or production-eligible translation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.global_languages.catalog import DocumentedCatalog,CATALOG_DIR
from research.global_languages.capabilities import GlobalLanguageRouter
from research.multilingual_student.model import read_config

REPORT=CATALOG_DIR/"real_tokenizer_capabilities.json"


def run():
    if REPORT.exists():
        raise FileExistsError("Never overwrite verified tokenizer capabilities")
    from transformers import M2M100Tokenizer
    cfg=read_config()
    tok=M2M100Tokenizer.from_pretrained(
        cfg["model"]["id"],revision=cfg["model"]["revision"],
        trust_remote_code=False)
    catalog=DocumentedCatalog.load()
    router=GlobalLanguageRouter(catalog,tok)
    real_codes=sorted(router.native_codes)
    if not 90<=len(real_codes)<=120:
        raise RuntimeError("Real M2M100 tokenizer coverage unexpectedly changed")
    for source,target in (("en","es"),("es","fr"),("fr","en"),
                          ("zh","en"),("de","fr")):
        decision=router.inspect(source,target)
        if not decision.can_run_native_model:
            raise RuntimeError("Known pretrained model pair is not routable: "+source+"-"+target)
        if decision.production_approved or decision.independently_meaning_verified:
            raise RuntimeError("Tokenizer availability was falsely promoted to quality")
    # Check authentic Glottolog language not present in this tokenizer.
    unknown_example=None
    for lang in catalog.by_glottocode.values():
        if lang.level=="language" and lang.iso639_3:
            decision=router.inspect("glotto:"+lang.glottocode,"en")
            if decision.model_coverage=="catalog_only":
                unknown_example=lang.glottocode
                break
    if not unknown_example:
        raise RuntimeError("Unexpectedly claims all world languages have M2M tokens")
    report={
        "status":"real_pinned_M2M100_native_tokenizer_checked_against_Glottolog",
        "base_model":cfg["model"]["id"],
        "base_revision":cfg["model"]["revision"],
        "real_native_language_codes":real_codes,
        "native_language_count":len(real_codes),
        "native_possible_pairs":len(real_codes)*(len(real_codes)-1),
        "Glottolog_documented_languages":catalog.stats()["documented_languages"],
        "Glottolog_documented_dialects":catalog.stats()["documented_dialects"],
        "Glottolog_not_native_example":unknown_example,
        "currently_shared_model_finetuned":["en","es","fr"],
        "independently_human_validated_new_language_pairs":0,
        "slang_or_context_accuracy_claim":False,
        "production_models_changed":False,
        "sourcing_note":"This is model tokenizer coverage, not a real-world translation evaluation.",
    }
    REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("REAL_NATIVE_MULTILINGUAL_TOKENIZER_VS_CATALOG",json.dumps({
        "native_model_languages":len(real_codes),
        "documented_languages":report["Glottolog_documented_languages"],
        "catalog_only_example":unknown_example,
        "production_ready_new_pairs":0,
    }),flush=True)


if __name__=="__main__":
    run()
