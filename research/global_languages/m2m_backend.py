"""Actual optional M2M100 backend in the future world-model registry.

This is still ONE text model and the existing selected shared LoRA, not
separate English/French/Spanish weights. New language families are added by
implementing ResearchBackend; the original BridgeCaption app remains untouched.
"""
from __future__ import annotations

from .backend_registry import BackendIdentity,WorldModelBackends
from .catalog import DocumentedCatalog
from .capabilities import GlobalLanguageRouter
from .runtime import ExpandableResearchTranslator
from research.multilingual_student.model import read_config


class CurrentSharedM2M100Backend:
    def __init__(self,catalog:DocumentedCatalog,*,tokenizer=None,translator=None):
        cfg=read_config()["model"]
        self.identity=BackendIdentity(
            backend_id="shared_m2m100_418m",
            pretrained_model_id=cfg["id"],
            pinned_revision=cfg["revision"],
            license=cfg["license"],modality="text",
            single_shared_base=True)
        self.catalog=catalog
        if tokenizer is None:
            from transformers import M2M100Tokenizer
            tokenizer=M2M100Tokenizer.from_pretrained(
                cfg["id"],revision=cfg["revision"],trust_remote_code=False)
        self.routes=GlobalLanguageRouter(catalog,tokenizer)
        self.translator=translator or ExpandableResearchTranslator(catalog=catalog)

    def native_codes(self)->frozenset[str]:
        return self.routes.native_codes

    def supports_pair(self,source:str,target:str)->bool:
        return self.routes.inspect(source,target).can_run_native_model

    def translate_research(self,text:str,source:str,target:str)->str:
        from bridgecaption.provider import TranslationRequest
        self.routes.require_research_route(source,target)
        return self.translator.translate(TranslationRequest(
            text=text,source=source,target=target,
            context_enabled=False,previous=[]))


def make_current_world_research_backend(*,catalog=None)->WorldModelBackends:
    source=catalog or DocumentedCatalog.load()
    world=WorldModelBackends(source)
    world.add(CurrentSharedM2M100Backend(source))
    return world
