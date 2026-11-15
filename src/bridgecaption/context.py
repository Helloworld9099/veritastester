"""Short, in-memory conversation context and context-sensitive slang hints."""
from collections import deque
from dataclasses import dataclass, field

SLANG = {
    "es": {
        "chido": "Mexican Spanish slang: cool/nice",
        "no manches": "Mexican Spanish: surprise or disbelief",
        "qué onda": "informal greeting; what is up",
        "guay": "slang in Spain: cool",
        "mola": "slang in Spain: something is cool",
        "vale": "can mean okay; meaning depends on context",
    },
    "en": {
        "no cap": "honestly; not lying",
        "bet": "can mean agreement rather than a wager",
        "fire": "can mean excellent rather than flames",
        "sick": "can mean awesome, but also ill",
        "ghosted": "stopped communicating",
        "lowkey": "quietly or somewhat",
        "mid": "mediocre, not necessarily a spatial middle",
        "sus": "suspicious",
    },
    "fr": {"ouf": "French slang meaning crazy/wild", "chelou": "strange/suspicious"},
    "zh": {"摸鱼": "slacking off at work", "牛": "may mean impressive", "内卷": "excessive competition"},
    "sw": {"poa": "colloquial cool/fine", "mambo": "informal greeting"},
}

def slang_hints(text: str, lang: str) -> list[str]:
    lowered = text.casefold()
    return [f"{term}: {meaning}" for term, meaning in SLANG.get(lang, {}).items()
            if term.casefold() in lowered][:5]

def parse_glossary(text: str) -> dict[str, str]:
    entries = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = [p.strip() for p in line.split("=", 1)]
        if key and value and len(key) <= 80 and len(value) <= 100:
            entries[key] = value
        if len(entries) >= 20:
            break
    return entries

@dataclass
class Conversation:
    previous: deque = field(default_factory=lambda: deque(maxlen=4))

    def add(self, text: str) -> None:
        text = " ".join(text.split())[:500]
        if text:
            self.previous.append(text)

    def snapshot(self) -> list[str]:
        return list(self.previous)

    def clear(self) -> None:
        self.previous.clear()
