"""Scalable language-model backend contracts, without an 8,000-model switch.

Glottolog grows independently of pretrained tokenizer vocabularies. New
translation families, script-specific text tokenizers and sign/speech engines
must register explicit audited backend capabilities, rights and training
lineage. This pure-Python contract makes expansion a backend/plugin addition,
not rewriting every UI or model-route switch statement.

Important: only M2M100 is actually implemented and tested right now.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .catalog import DocumentedCatalog,parse_bcp47
from .capabilities import GlobalLanguageRouter,UnsupportedLanguageRoute

_SHA=re.compile(r"[0-9a-f]{40}\Z")
SAFE_LICENSES=frozenset(("mit","apache-2.0","cc0","cc0-1.0",
                          "cc-by-4.0","cc-by-2.0-fr","cc-by-sa-4.0"))


@dataclass(frozen=True)
class BackendIdentity:
    backend_id:str
    pretrained_model_id:str
    pinned_revision:str
    license:str
    modality:str
    single_shared_base:bool
    verified_human_meaning:bool=False
    released_to_production:bool=False

    def __post_init__(self):
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,50}",self.backend_id):
            raise ValueError("Unsafe backend identifier")
        if not self.pretrained_model_id or not _SHA.fullmatch(self.pinned_revision):
            raise ValueError("Every added model family must pin an immutable upstream revision")
        if self.license.lower() not in SAFE_LICENSES:
            raise ValueError("Noncommercial, unspecified or restricted model license")
        if self.modality not in {"text","speech","sign_video"}:
            raise ValueError("Unspecified input modality")
        if self.verified_human_meaning or self.released_to_production:
            raise ValueError("This research registry cannot grant human evaluation or release status")


@runtime_checkable
class ResearchBackend(Protocol):
    identity:BackendIdentity
    def native_codes(self)->frozenset[str]: ...
    def supports_pair(self,source:str,target:str)->bool: ...
    def translate_research(self,text:str,source:str,target:str)->str: ...


@dataclass(frozen=True)
class BackendDecision:
    source_identifier:str
    target_identifier:str
    modality:str
    eligible_backends:tuple[str,...]
    missing_model:bool
    production_approved:bool
    human_meaning_verified:bool
    reason:str


class WorldModelBackends:
    """Memory/runtime provider registry, no uncontrolled dynamic imports."""
    def __init__(self,catalog:DocumentedCatalog):
        self.catalog=catalog
        self.backends:dict[str,ResearchBackend]={}

    def add(self,backend:ResearchBackend):
        if not isinstance(backend,ResearchBackend):
            raise TypeError("Backend must implement the complete research translator contract")
        identity=backend.identity
        if identity.backend_id in self.backends:
            raise ValueError("Duplicate model family")
        native=backend.native_codes()
        if not native or any(not code.islower() for code in native):
            raise ValueError("Backend native tokenizer language code inventory invalid")
        if identity.modality!="text" and not identity.single_shared_base:
            # Additional speech and sign backends may be integrated later but
            # should be audited under modality-specific contracts, not routed
            # as text translations without signal alignment evidence.
            raise ValueError("Modality-specific runtime needs a separate approval path")
        self.backends[identity.backend_id]=backend

    def inspect(self,source:str,target:str,*,modality:str="text")->BackendDecision:
        if modality not in {"text","speech","sign_video"}:
            raise ValueError("Unknown modality")
        if not self.catalog.cataloged(source) or not self.catalog.cataloged(target):
            return BackendDecision(source,target,modality,(),True,False,False,
                                   "Unknown documented language identifier")
        eligible=[]
        for key,backend in sorted(self.backends.items()):
            if backend.identity.modality!=modality:
                continue
            if backend.supports_pair(source,target):
                eligible.append(key)
        return BackendDecision(
            source,target,modality,tuple(eligible),not bool(eligible),
            production_approved=False,human_meaning_verified=False,
            reason=("Pretrained research backend exists, but pair adequacy and "
                    "dialect coverage are unverified" if eligible else
                    "Catalog knows this language but no approved native tokenizer/model backend supports the pair"))

    def dispatch_research(self,source:str,target:str,text:str,*,backend_id:str|None=None)->str:
        decision=self.inspect(source,target)
        if not decision.eligible_backends:
            raise UnsupportedLanguageRoute(decision.reason)
        if backend_id is None:
            if len(decision.eligible_backends)!=1:
                raise ValueError("Multiple research backends require explicit benchmark selection")
            backend_id=decision.eligible_backends[0]
        if backend_id not in decision.eligible_backends:
            raise UnsupportedLanguageRoute("Requested backend cannot process source/target identifiers")
        result=self.backends[backend_id].translate_research(text,source,target)
        if not isinstance(result,str) or not result.strip():
            raise RuntimeError("Research backend produced no translation")
        return result

    def production_translate(self,*args,**kwargs)->str:
        raise PermissionError("Global models have not passed independent human accuracy, legal and latency gates")

    def stats(self)->dict:
        return {"catalog_languages":self.catalog.stats()["documented_languages"],
                "catalog_dialects":self.catalog.stats()["documented_dialects"],
                "registered_research_model_families":len(self.backends),
                "available_modalities":sorted(set(x.identity.modality for x in self.backends.values())),
                "independently_human_verified_models":0,
                "new_production_languages_activated":0,
                "notes":"A catalog entry, tokenizer code and production-quality MT are three distinct states"}
