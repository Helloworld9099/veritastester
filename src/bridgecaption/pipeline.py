"""Audio-free translation orchestration with distinct baseline and context modes."""
from dataclasses import dataclass

from .context import Conversation
from .provider import GeminiTranslator, TranslationRequest
from .training import infer_topic

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
        guessed = infer_topic(" ".join(self.history.snapshot()[-2:] + [text]))
        selected = guessed if topic == "auto" else topic
        response = self.translator.translate(TranslationRequest(
            text=text, source=source, target=target, topic=selected,
            previous=self.history.snapshot() if contextual else [],
            glossary=glossary or {} if contextual else {}, context_enabled=contextual))
        self.history.add(text)
        return Caption(text, response, source, selected)

    def reset(self):
        self.history.clear()
