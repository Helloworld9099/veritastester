"""Audio-free translation orchestration with distinct baseline and context modes."""
from dataclasses import dataclass

from .context import Conversation
from .provider import GeminiTranslator, TranslationRequest

@dataclass
class Caption:
    original: str
    translated: str
    source_language: str
    topic: str

class Pipeline:
    def __init__(self, translator=None):
        self.translator = translator if translator is not None else GeminiTranslator()
        self.history = Conversation()

    def translate(self, text, source="auto", target="en", topic="auto", glossary=None, contextual=True):
        text = " ".join(text.split())
        if not text:
            raise ValueError("Blank transcript")
        previous = self.history.snapshot() if contextual else []
        # The synthetic topic classifier has no validated real-world performance.
        # "auto" safely uses a neutral topic until independent testing is available.
        selected = "general" if topic == "auto" else topic
        # Same-language captions require no API call; preserve raw spoken text.
        response = (text if source == target else self.translator.translate(TranslationRequest(
            text=text, source=source, target=target, topic=selected,
            previous=previous, glossary=(glossary or {}) if contextual else {},
            context_enabled=contextual)))
        if contextual:
            self.history.add(text)
        else:
            # Disabling context must also prevent later reuse of this utterance.
            self.history.clear()
        return Caption(text, response, source, selected)

    def reset(self):
        self.history.clear()
