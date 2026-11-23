#!/usr/bin/env python3
"""Explore thousands of documented languages without starting a neural model.

Examples:
 PYTHONPATH=.:src python -m research.global_languages.cli --search Yoruba
 PYTHONPATH=.:src python -m research.global_languages.cli --source es-MX --target fr
 PYTHONPATH=.:src python -m research.global_languages.cli --source glotto:aari1239 --target en

Never reports cataloged entries as quality-certified translation support.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from .catalog import DocumentedCatalog,CATALOG_DIR
from .capabilities import GlobalLanguageRouter


def main(argv=None):
    parser=argparse.ArgumentParser(description="BridgeCaption world language research catalog")
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--search",help="Case-insensitive documented language/dialect name")
    action.add_argument("--source",help="Explicit BCP47 or glotto:<id> source language")
    action.add_argument("--summary",action="store_true",help="Show catalog and actual pretrained model coverage")
    parser.add_argument("--target",help="Explicit BCP47 or glotto:<id> target language")
    parser.add_argument("--limit",type=int,default=15)
    args=parser.parse_args(argv)
    if not 1<=args.limit<=100:
        raise ValueError("A search result cap from 1 to 100 is required")
    catalog=DocumentedCatalog.load()
    audit=CATALOG_DIR/"real_tokenizer_capabilities.json"
    if not audit.is_file():
        raise FileNotFoundError("Run the verified pinned tokenizer audit first")
    meta=json.loads(audit.read_text())
    if meta["status"]!="real_pinned_M2M100_native_tokenizer_checked_against_Glottolog":
        raise RuntimeError("Native tokenizer model provenance is not verified")
    tokenizer=SimpleNamespace(lang_code_to_id={
        code:i for i,code in enumerate(meta["real_native_language_codes"])})
    router=GlobalLanguageRouter(catalog,tokenizer)
    if args.search is not None:
        query=args.search.strip().casefold()
        if not query or len(query)>100:
            raise ValueError("Search query required")
        items=[{"glottocode":x.glottocode,"name":x.name,"level":x.level,
                "iso639_3":x.iso639_3}
               for x in catalog.by_glottocode.values() if query in x.name.casefold()]
        items=sorted(items,key=lambda x:(x["level"]!="language",
                                         x["name"].casefold()))[:args.limit]
        response={"query":args.search,"results":items,
                  "these_are_documented_not_translation_verified":True}
    elif args.source is not None:
        if not args.target:
            raise ValueError("--target required for source pair capability checks")
        if len(args.source)>80 or len(args.target)>80:
            raise ValueError("Language identifier too long")
        response=asdict(router.inspect(args.source,args.target))
    else:
        response=router.coverage_stats()
    print(json.dumps(response,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
