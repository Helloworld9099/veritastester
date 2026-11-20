"""Short, in-memory conversation context and context-sensitive slang hints."""
from collections import deque
from dataclasses import dataclass, field
import re

SLANG = {
    "es": {
        "chido": "Mexican Spanish slang: cool/nice",
        "no manches": "Mexican Spanish: surprise or disbelief",
        "qué onda": "informal greeting; what is up",
        "guay": "slang in Spain: cool",
        "mola": "slang in Spain: something is cool",
        "vale": "can mean okay; meaning depends on context",
        "ahorita": "may mean now, soon or later depending on region and context; never invent an exact time",
        "guagua": "regional ambiguity: bus in Canary Islands/Caribbean, baby in Chile; resolve from geography and conversation",
        "chamba": "regional informal job/work, not a literal object",
        "pana": "Venezuelan and other informal usage: buddy/friend; in other contexts may mean corduroy",
        "parce": "Colombian informal address: friend",
        "chévere": "regional compliment: cool, excellent, nice",
        "bacán": "regional compliment: excellent/cool",
        "brutal": "in some contexts enthusiastic praise; can also literally describe violence",
        "lana": "in some Mexican contexts money, rather than wool",
        "me quedé sin lana": "Mexican colloquial expression: I ran out of money",
        "me dejó en visto": "messaging idiom: left me on read, not literally in sight",
        "me echó la mano": "idiom: helped me out, not shook hands",
        "se puso las pilas": "idiom: got motivated/started making an effort, not physically obtained batteries",
        "de pelos": "some regional praise: awesome, great",
        "tomando el pelo": "idiom: joking, teasing, pulling someone's leg",
        "en las nubes": "idiom: daydreaming or distracted, not literally in clouds",
        "tirar": "context-dependent: pull, throw, toss or discard; use object and situation",
        "tocar": "context-dependent: touch, play music, be one's turn; use topic",
        "volante": "flyer or steering wheel depending on context",
        "bola": "in ice-cream context scoop; otherwise ball or rounded object",

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
        "i'm down": "with an invitation may mean I agree/I'm interested; in mental-health context may mean I feel low",
        "i’m down": "with an invitation may mean agreement; in other contexts may describe sadness or physical position",
        "cooked": "colloquial: I am in trouble or done for; may also literally mean cooked food",
        "slaps": "of a song or performance: it's excellent or impressive; may otherwise literally mean hits",
        "dead": "in reaction to a joke may mean laughing hard; otherwise literal or figurative death",
        "spill the tea": "informal: share gossip or news, not pour a beverage",
        "break a leg": "as encouragement before a performance means good luck",
        "stay on top of things": "idiom: keep responsibilities organized and under control",
        "understood the assignment": "informal praise for doing something especially well",
        "catch a ride": "get transportation with someone; not physically catch a ride",
        "tag": "playground game meaning chase and touch someone; can mean label in other domains",
        "slide": "playground slide or presentation slide depending on topic",

    },
    "fr": {"ouf": "French slang meaning crazy/wild", "chelou": "strange/suspicious"},
    "zh": {"摸鱼": "slacking off at work", "牛": "may mean impressive", "内卷": "excessive competition"},
    "sw": {"poa": "colloquial cool/fine", "mambo": "informal greeting"},
}

def slang_hints(text: str, lang: str) -> list[str]:
    """Supply possible interpretations, never replace the speaker's words."""
    lowered = text.casefold()
    hints = []
    language = lang.strip().lower().split("-", 1)[0] if isinstance(lang, str) else ""
    for term, meaning in SLANG.get(language, {}).items():
        needle = term.casefold()
        if re.search(r"[a-z]", needle):
            matched = bool(re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", lowered))
        else:
            # CJK phrases do not always use whitespace between words.
            matched = needle in lowered
        if matched:
            hints.append(f"{term}: {meaning}")
    return hints[:5]

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
