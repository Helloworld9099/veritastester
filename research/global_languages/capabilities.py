"""Capability-aware routing for ALL cataloged languages, without false translation.

The present single multilingual M2M100 network has ~100 native codes (from
its *actual tokenizer*, NOT the number of Glottolog names). Broader
documented-language support requires new tokenizer embeddings/new pretrained
base/datasets and human verification, never a silent fallback to another
language. This layer anticipates that work without altering the installed app.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .catalog import DocumentedCatalog,LanguageTag,parse_bcp47

FINETUNED=frozenset(("en","es","fr"))
KNOWN_ISO_TO_M2M={
    # Explicit reviewed ISO 639-3 -> actual tokenizer code, never an
    # automatic guess from language name or glottocode.
    "eng":"en","spa":"es","fra":"fr",
}
VERSION="global_language_routes_v1"


class UnsupportedLanguageRoute(ValueError):
    pass


@dataclass(frozen=True)
class Capability:
    source_identifier:str
    target_identifier:str
    source_native_code:str|None
    target_native_code:str|None
    source_documented:bool
    target_documented:bool
    can_run_native_model:bool
    native_model_id:str
    model_coverage:str
    research_finetuned:bool
    regional_variant_evaluated:bool
    independently_meaning_verified:bool
    production_approved:bool
    reason:str


def native_m2m100_codes(tokenizer)->frozenset[str]:
    """Truth comes from exact loaded tokenizer, not arbitrary catalog names."""
    codes=getattr(tokenizer,"lang_code_to_id",None)
    if not isinstance(codes,Mapping) or not codes:
        raise RuntimeError("Tokenizer has no pinned language-ID vocabulary")
    valid=frozenset(str(key).strip().lower() for key,value in codes.items()
                    if isinstance(value,int) and str(key).strip())
    if not {"en","es","fr"}.issubset(valid) or len(valid)<90 or len(valid)>140:
        raise RuntimeError("M2M100 language vocabulary unexpectedly changed")
    return valid


def native_code(identifier:str,catalog:DocumentedCatalog,
                native_codes:frozenset[str])->tuple[str|None,bool,LanguageTag|None]:
    if identifier.startswith("glotto:"):
        record=catalog.lookup(identifier)
        if record is None:
            return None,False,None
        iso=record.iso639_3
        native=KNOWN_ISO_TO_M2M.get(iso) if iso else None
        if native not in native_codes:
            native=None
        return native,True,None
    tag=parse_bcp47(identifier)
    documented=catalog.cataloged(identifier)
    primary=catalog.iana.get("preferred",{}).get(tag.primary,tag.primary)
    if not documented:
        return None,False,tag
    # ISO639-3 languages only enter the existing model through an explicit
    # vetted identifier crosswalk. Reject unsupported script/macrolanguage
    # substitutions rather than auto-translating the wrong language.
    value=(primary if primary in native_codes
           else KNOWN_ISO_TO_M2M.get(primary))
    return value if value in native_codes else None,True,tag


class GlobalLanguageRouter:
    def __init__(self,catalog:DocumentedCatalog,tokenizer):
        self.catalog=catalog
        self.native_codes=native_m2m100_codes(tokenizer)
        self.tokenizer=tokenizer

    def inspect(self,source:str,target:str)->Capability:
        if not isinstance(source,str) or not isinstance(target,str):
            raise ValueError("Explicit source and target identifiers required")
        src,source_known,source_tag=native_code(source,self.catalog,self.native_codes)
        dst,target_known,target_tag=native_code(target,self.catalog,self.native_codes)
        possible=bool(src and dst and src!=dst)
        fine=possible and src in FINETUNED and dst in FINETUNED
        regional=bool((source_tag and (source_tag.region or source_tag.script or source_tag.variants))
                      or (target_tag and (target_tag.region or target_tag.script or target_tag.variants)))
        if not source_known or not target_known:
            coverage="unknown_identifier"
            explanation="Source or target is not in the versioned documented-language catalog"
        elif not src or not dst:
            coverage="catalog_only"
            explanation="Language is documented but the current M2M100 tokenizer has no verified native language token"
        elif src==dst:
            coverage="same_language"
            explanation="Source and target resolve to the same base language; translation not needed"
        elif regional:
            coverage="native_variant_unvalidated"
            explanation="Base language has a tokenizer token, but the requested script, region or dialect is not independently tested"
        elif fine:
            coverage="research_finetuned_unvalidated"
            explanation="One shared learned checkpoint exists but human semantic accuracy and production adequacy are unverified"
        else:
            coverage="native_pretrained_unvalidated"
            explanation="Native M2M100 token exists but no BridgeCaption fine-tuning or independent quality review"
        return Capability(
            source_identifier=source,target_identifier=target,
            source_native_code=src,target_native_code=dst,
            source_documented=source_known,target_documented=target_known,
            can_run_native_model=possible,
            native_model_id="facebook/m2m100_418M",
            model_coverage=coverage,
            research_finetuned=bool(fine),
            regional_variant_evaluated=False,
            independently_meaning_verified=False,
            production_approved=False,
            reason=explanation)

    def require_research_route(self,source:str,target:str)->tuple[str,str]:
        data=self.inspect(source,target)
        if not data.can_run_native_model:
            raise UnsupportedLanguageRoute(
                f"{data.model_coverage}: {data.reason}. "
                "No silent model/language substitution is permitted.")
        return data.source_native_code,data.target_native_code

    def coverage_stats(self)->dict:
        catalog=self.catalog.stats()
        return {**catalog,
                "native_m2m100_language_codes":len(self.native_codes),
                "native_language_pairs_theoretically_possible":
                    len(self.native_codes)*(len(self.native_codes)-1),
                "current_finetuned_base_languages":sorted(FINETUNED),
                "independently_human_validated_pairs":0,
                "production_multilingual_routes_activated":0,
                "documented_languages_not_native_model_ready":
                    "Catalog supports future language extensions; no fabricated count without verified crosswalk"}
