"""Provider-independent translation with optional grounded memory selection."""
from dataclasses import dataclass

from .context import Conversation
from .context_engine import ContextSelector
from .provider import TranslationRequest
from .providers import create_translator


@dataclass
class Caption:
    original: str
    translated: str
    source_language: str
    topic: str
    context_info: str = "none"


class Pipeline:
    def __init__(self, translator=None, selector=None, context_mode="selective"):
        self.translator = translator if translator is not None else create_translator()
        self.selector = selector if selector is not None else ContextSelector()
        if context_mode not in {"selective", "all"}:
            raise ValueError("context_mode must be selective or all")
        self.context_mode = context_mode
        self.history = Conversation()

    def translate(self, text, source="auto", target="en", topic="auto", glossary=None, contextual=True):
        text = " ".join(text.split())[:1500]
        if not text:
            raise ValueError("Blank transcript")
        selected_topic = "general" if topic == "auto" else topic
        supports_context = getattr(self.translator, "supports_context", True)
        with_context = bool(contextual and supports_context and source != target)
        if with_context:
            memory = self.history.snapshot()
            if self.context_mode == "all":
                previous = memory[-4:]
                reason = "always_include"
            else:
                result = self.selector.select(text, memory, language=source,
                                              topic=selected_topic)
                previous, reason = list(result.memories), result.mode
        else:
            previous, reason = [], "disabled"
        response = text if source == target else self.translator.translate(TranslationRequest(
            text=text, source=source, target=target, topic=selected_topic,
            previous=previous, glossary=(glossary or {}) if with_context else {},
            context_enabled=with_context))
        if with_context:
            self.history.add(text)
        else:
            self.history.clear()
        return Caption(text, response, source, selected_topic,
                       context_info=f"{reason}: {len(previous)} prior utterance(s)")

    def reset(self):
        self.history.clear()
