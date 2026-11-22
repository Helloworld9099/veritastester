#!/usr/bin/env python3
"""Reproducibly package Glottolog 5.4 and IANA identifiers, not model support.

The Glottolog CSV is CC BY 4.0. We keep source URL, raw archive hash,
extract checksum, dates, counts and license in the output manifest.
No external data is installed as Python dependencies or used for training.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import asdict
from pathlib import Path

from .catalog import CATALOG_DIR,from_glottolog_csv,parse_iana_subtags,sha256_bytes

GLOTTOLOG_VERSION="5.4"
GLOTTOLOG_URL="https://zenodo.org/records/24104033/files/glottolog_languoid.csv.zip?download=1"
IANA_URL="https://www.iana.org/assignments/language-subtag-registry/language-subtag-registry"
MAX_ARCHIVE_BYTES=8*1024*1024
MAX_CSV_BYTES=18*1024*1024
MAX_IANA_BYTES=5*1024*1024


def fetch_bytes(url:str,max_size:int)->bytes:
    req=urllib.request.Request(url,headers={"User-Agent":"BridgeCaption-Research-Data-Importer/1.0"})
    with urllib.request.urlopen(req,timeout=35) as response:
        raw=response.read(max_size+1)
    if len(raw)>max_size:
        raise RuntimeError("Upstream metadata exceeded expected bounded size")
    return raw


def build_catalog(glottolog_zip:bytes,iana_text:bytes,*,out:Path=CATALOG_DIR)->dict:
    if len(glottolog_zip)>MAX_ARCHIVE_BYTES or len(iana_text)>MAX_IANA_BYTES:
        raise ValueError("Unreasonably large catalog input rejected")
    with zipfile.ZipFile(io.BytesIO(glottolog_zip)) as archive:
        files=[n for n in archive.namelist() if n.endswith(".csv") and "/" not in n]
        if len(files)!=1:
            raise ValueError("Expected one flat Glottolog CSV file")
        info=archive.getinfo(files[0])
        if info.file_size>MAX_CSV_BYTES:
            raise ValueError("Zip-bomb protection: uncompressed CSV too large")
        with archive.open(info) as stream:
            csv_bytes=stream.read(MAX_CSV_BYTES+1)
        if len(csv_bytes)>MAX_CSV_BYTES:
            raise ValueError("Zip-bomb protection: actual CSV too large")
    rows=from_glottolog_csv(csv_bytes.decode("utf-8-sig"))
    if len(rows)<2:
        raise RuntimeError("Catalog contained no language records")
    iana=parse_iana_subtags(iana_text.decode("utf-8-sig"))
    canonical="\n".join(json.dumps(asdict(item),ensure_ascii=False,sort_keys=True)
                        for item in rows)+"\n"
    compressed=gzip.compress(canonical.encode("utf-8"),compresslevel=9,mtime=0)
    iana_blob=(json.dumps(iana,sort_keys=True,ensure_ascii=False,
                          separators=(",",":"))+"\n").encode("utf-8")
    by_level={"language":0,"dialect":0}
    for item in rows:
        by_level[item.level]+=1
    manifest={
        "schema_version":1,
        "source_name":"Glottolog",
        "source_version":GLOTTOLOG_VERSION,
        "source_url":GLOTTOLOG_URL,
        "source_archive_sha256":sha256_bytes(glottolog_zip),
        "source_csv_sha256":sha256_bytes(csv_bytes),
        "source_license":"CC-BY-4.0",
        "source_attribution":"Glottolog 5.4, Hammarstrom et al.; https://glottolog.org/download",
        "iana_source_url":IANA_URL,
        "iana_registry_raw_sha256":sha256_bytes(iana_text),
        "iana_registry_file_date":iana["file_date"],
        "iana_registry_license_note":"Official public language identifier registry",
        "catalog_sha256":sha256_bytes(compressed),
        "iana_sha256":sha256_bytes(iana_blob),
        "languoids_total":len(rows),
        "documented_languages":by_level["language"],
        "documented_dialects":by_level["dialect"],
        "iana_language_subtags":len(iana["languages"]),
        "mt_validated_pairs":0,
        "neural_model_extended_to_new_languages":False,
        "meaning_accuracy_verified":False,
        "translation_production_supported":False,
        "distinction":"Language catalog is not a claim of usable ASR or machine translation."
    }
    out=Path(out)
    if out.exists() and any(out.iterdir()):
        # Idempotently allow exactly identical result; never rewrite data
        # without a versioned and reviewed upstream source change.
        old_path=out/"manifest.json"
        if not old_path.is_file():
            raise FileExistsError("Output directory already has files without manifest")
        old=json.loads(old_path.read_text())
        if old!=manifest:
            raise FileExistsError("Upstream Glottolog/IANA changed; use a new versioned directory")
        return manifest
    out.mkdir(parents=True,exist_ok=True)
    (out/"languoids.jsonl.gz").write_bytes(compressed)
    (out/"iana_languages.json").write_bytes(iana_blob)
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--glottolog-zip",type=Path)
    parser.add_argument("--iana-registry",type=Path)
    parser.add_argument("--out",type=Path,default=CATALOG_DIR)
    args=parser.parse_args(argv)
    if bool(args.glottolog_zip)!=bool(args.iana_registry):
        raise ValueError("Supply both offline inputs, or neither")
    if args.glottolog_zip:
        glot=args.glottolog_zip.read_bytes()
        iana=args.iana_registry.read_bytes()
    else:
        glot=fetch_bytes(GLOTTOLOG_URL,MAX_ARCHIVE_BYTES)
        iana=fetch_bytes(IANA_URL,MAX_IANA_BYTES)
    report=build_catalog(glot,iana,out=args.out)
    if not 7000<=report["documented_languages"]<=15000:
        raise RuntimeError("Unexpected Glottolog documented-language count; refuse to publish")
    print("VERIFIED_REAL_DOCUMENTED_LANGUAGE_CATALOG",json.dumps({
        "languages":report["documented_languages"],
        "dialects":report["documented_dialects"],
        "iana_subtags":report["iana_language_subtags"],
        "catalog_sha256":report["catalog_sha256"],
        "translation_adequacy_claim":False
    }),flush=True)


if __name__=="__main__":
    main()
