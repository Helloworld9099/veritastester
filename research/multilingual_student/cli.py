#!/usr/bin/env python3
"""Try the real downloaded SINGLE standalone multilingual model from the CLI.

Use the unchanged BridgeCaption app normally; this explicit research CLI
never installs a default model, downloads an API teacher, or accesses audio.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from bridgecaption.provider import TranslationRequest
from research.multilingual_student.standalone_runtime import (
    OneWeightMultilingualResearchTranslator
)


def main(argv=None):
    cli=argparse.ArgumentParser(description="One shared EN/ES/FR research translation model")
    cli.add_argument("--model-dir",type=Path,required=True,
                     help="Extracted full model checkpoint from GitHub Actions artifact")
    cli.add_argument("--source",choices=["en","es","fr"],required=True)
    cli.add_argument("--target",choices=["en","es","fr"],required=True)
    cli.add_argument("--text",required=True)
    cli.add_argument("--receipt",type=Path,default=None,
                     help="standalone_export.json (defaults to the branch's pinned report)")
    args=cli.parse_args(argv)
    kwargs={"receipt_path":args.receipt} if args.receipt is not None else {}
    engine=OneWeightMultilingualResearchTranslator(args.model_dir,**kwargs)
    output=engine.translate(TranslationRequest(
        args.text,source=args.source,target=args.target,context_enabled=False))
    print(output)


if __name__=="__main__":
    main()
