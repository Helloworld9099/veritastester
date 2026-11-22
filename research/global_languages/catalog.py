"""Documented languages and dialects are identifiers, not translation guarantees.

Glottolog is a CC BY 4.0 catalog. IANA RFC 5646 subtags are identifiers.
This intentionally keeps catalog coverage distinct from model-tokenizer support,
language-pair training, bilingual adequacy, and speech recognition coverage.

No dependencies outside the Python standard library.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
CATALOG_DIR=ROOT/"research/global_languages/generated"
GLOTTOCODE=re.compile(r"[a-z]{4}[0-9]{4}\Z")
ISOCODE=re.compile(r"[a-z]{3}\Z")
LANGUAGE_SUBTAG=re.compile(r"[a-zA-Z]{2,8}\Z")
SCRIPT_SUBTAG=re.compile(r"[a-zA-Z]{4}\Z")
REGION_SUBTAG=re.compile(r"(?:[a-zA-Z]{2}|[0-9]{3})\Z")
VARIANT_SUBTAG=re.compile(r"(?:[a-zA-Z0-9]{5,8}|[0-9][a-zA-Z0-9]{3})\Z")


def sha256_bytes(value:bytes)->str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class DocumentedLanguage:
    glottocode:str
    name:str
    level:str
    iso639_3:str|None
    parent_glottocode:str|None

    def __post_init__(self):
        if not GLOTTOCODE.fullmatch(self.glottocode):
            raise ValueError("Unrecognized Glottocode")
        if self.level not in {"language","dialect"}:
            raise ValueError("Families must not be presented as languages")
        if self.iso639_3 is not None and not ISOCODE.fullmatch(self.iso639_3):
            raise ValueError("Invalid ISO 639-3 code")


@dataclass(frozen=True)
class LanguageTag:
    primary:str
    script:str|None
    region:str|None
    variants:tuple[str,...]
    canonical_tag:str
    is_private:bool=False


def parse_bcp47(tag:str)->LanguageTag:
    """Conservative BCP 47 subset: primary[-Script][-Region][-Variant...].

    Unsupported extensions/private use/grandfathered tags MUST use a
    separate IANA-compliant parser, not silently get downgraded to primary.
    """
    if not isinstance(tag,str) or not tag.strip() or len(tag)>80:
        raise ValueError("Language identifier is required")
    if tag.lower().startswith("x-") or tag.lower()=="auto":
        raise ValueError("Private/automatic language code cannot be mapped to a model")
    parts=tag.strip().split("-")
    if not LANGUAGE_SUBTAG.fullmatch(parts[0]):
        raise ValueError("Invalid primary BCP 47 language")
    primary=parts[0].lower()
    rest=parts[1:]
    script=None
    region=None
    if rest and SCRIPT_SUBTAG.fullmatch(rest[0]):
        script=rest.pop(0).title()
    if rest and REGION_SUBTAG.fullmatch(rest[0]):
        region=rest.pop(0).upper()
    variants=[]
    for value in rest:
        if not VARIANT_SUBTAG.fullmatch(value):
            raise ValueError("Unsupported extension/variant or malformed tag")
        if value.casefold() in variants:
            raise ValueError("Duplicate language variant")
        variants.append(value.lower())
    canonical="-".join([primary,*([script] if script else []),
                        *([region] if region else []),*variants])
    return LanguageTag(primary,script,region,tuple(variants),canonical)


def from_glottolog_csv(text:str)->list[DocumentedLanguage]:
    """Parse Glottolog 5.x released languoid CSV, preserving dialect entries."""
    reader=csv.DictReader(io.StringIO(text))
    fields=set(reader.fieldnames or [])
    required={"name","level"}
    if not required.issubset(fields) or not ({"glottocode","id"}&fields):
        raise ValueError("Unexpected Glottolog schema: "+", ".join(sorted(fields)))
    result=[]
    seen=set()
    for row in reader:
        level=str(row.get("level","")).strip().lower()
        if level not in {"language","dialect"}:
            continue
        glotto=str(row.get("glottocode") or row.get("id") or "").strip()
        name=str(row.get("name","")).strip()
        iso=str(row.get("iso639P3code") or row.get("iso639_3") or "").strip().lower()
        parent=str(row.get("parent_id") or row.get("parent") or "").strip()
        if not name or glotto in seen:
            raise ValueError("Empty name or duplicate Glottocode")
        if parent and not GLOTTOCODE.fullmatch(parent):
            parent=None
        language=DocumentedLanguage(
            glottocode=glotto,name=name,level=level,
            iso639_3=iso if iso and iso not in {"nan","none"} else None,
            parent_glottocode=parent or None)
        seen.add(glotto)
        result.append(language)
    return sorted(result,key=lambda r:r.glottocode)


def parse_iana_subtags(text:str)->dict:
    """Parse IANA language-subtag-registry; retain preferred replacements."""
    if not text.startswith("File-Date: "):
        raise ValueError("Expected original IANA Language Subtag Registry")
    language_codes={}
    preferred={}
    for block in text.split("%%"):
        lines=block.strip().splitlines()
        props={}
        key=None
        for line in lines:
            if line[:1].isspace() and key:
                props[key][-1]+=" "+line.strip()
            elif ": " in line:
                key,val=line.split(": ",1)
                props.setdefault(key,[]).append(val.strip())
        if props.get("Type")!=["language"]:
            continue
        code=(props.get("Subtag") or [None])[0]
        if not code or not LANGUAGE_SUBTAG.fullmatch(code):
            continue
        language_codes[code.lower()]={
            "name":(props.get("Description") or [code])[0],
            "deprecated":bool(props.get("Deprecated")),
            "macrolanguage":(props.get("Macrolanguage") or [None])[0],
        }
        if "Preferred-Value" in props:
            preferred[code.lower()]=props["Preferred-Value"][0].lower()
    if len(language_codes)<100:
        raise ValueError("IANA registry unexpectedly incomplete")
    return {"languages":language_codes,"preferred":preferred,
            "file_date":text.splitlines()[0].split(": ",1)[1]}


class DocumentedCatalog:
    def __init__(self,records:list[DocumentedLanguage],iana:dict):
        self.by_glottocode={record.glottocode:record for record in records}
        self.by_iso={}
        for record in records:
            if record.iso639_3 and record.level=="language":
                # A code can be shared/ambiguous; never silently pick from
                # two competing separate Glottolog language records.
                self.by_iso.setdefault(record.iso639_3,[]).append(record)
        self.iana=iana

    @classmethod
    def load(cls, directory:Path=CATALOG_DIR):
        directory=Path(directory)
        manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
        path=directory/"languoids.jsonl.gz"
        data=path.read_bytes()
        if sha256_bytes(data)!=manifest["catalog_sha256"]:
            raise RuntimeError("Documented language catalog checksum mismatch")
        iana_file=directory/"iana_languages.json"
        if sha256_bytes(iana_file.read_bytes())!=manifest["iana_sha256"]:
            raise RuntimeError("IANA language identifiers checksum mismatch")
        raw=json.loads(iana_file.read_text(encoding="utf-8"))
        records=[DocumentedLanguage(**json.loads(line))
                 for line in gzip.decompress(data).decode("utf-8").splitlines()]
        if len(records)!=manifest["languoids_total"]:
            raise RuntimeError("Catalog manifest count disagrees with contents")
        return cls(records,raw)

    def lookup(self, identifier:str)->DocumentedLanguage|None:
        if not isinstance(identifier,str):
            raise ValueError("Language identifier must be a string")
        if identifier.startswith("glotto:"):
            code=identifier.split(":",1)[1]
            if not GLOTTOCODE.fullmatch(code):
                raise ValueError("Malformed Glottocode")
            return self.by_glottocode.get(code)
        code=parse_bcp47(identifier).primary
        code=self.iana.get("preferred",{}).get(code,code)
        rows=self.by_iso.get(code,[])
        if len(rows)>1:
            raise ValueError("ISO 639-3 code is ambiguous in Glottolog")
        return rows[0] if rows else None

    def cataloged(self,identifier:str)->bool:
        if identifier.startswith("glotto:"):
            return self.lookup(identifier) is not None
        tag=parse_bcp47(identifier)
        if tag.primary in self.iana["languages"]:
            return True
        return self.lookup(identifier) is not None

    def stats(self)->dict:
        levels={"language":0,"dialect":0}
        for value in self.by_glottocode.values():
            levels[value.level]+=1
        return {"documented_languages":levels["language"],
                "documented_dialects":levels["dialect"],
                "documented_languoids":sum(levels.values()),
                "iana_language_subtags":len(self.iana["languages"]),
                "translation_ready":0,
                "notes":"Catalog coverage does not imply MT or ASR support"}
