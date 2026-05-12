"""Content safety guard for Companion outputs.

Content guard for Companion outputs. The bar:

- Companion personae are sarcastic and dry. That is voice, not unsafe.
- Companion is never sexist, racist, ableist, homo/transphobic,
  partisan, or cruel about real people. It does not instigate crime,
  glorify violence, or joke about self-harm.

A hit returns a `SafetyHit` carrying the matched category. The API
layer surfaces a neutral fallback instead of the raw text and records
both the raw text and the reason in the JSONL log for review.

This file is deliberately conservative — false positives are easy to
fix (rename a phrase). False negatives are not.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_PATTERNS: list[tuple[str, str]] = [
    # sex / gender / orientation
    ("sexist", r"\b(weiber|tussi|schlampe|hure|nutte|fotze|votze|"
               r"frauen sind|maenner sind|typisch frau|typisch mann)\b"),
    ("sexist", r"\b(blondine|blond witz|frau am steuer)\b"),
    ("homophobic", r"\b(schwuchtel|schwuchteln|tunte|kampflesbe|homo witz)\b"),
    ("transphobic", r"\b(tranny|trannie|shemale|es war mal ein mann|"
                    r"biologisch eine frau|biologisch ein mann)\b"),
    # race / ethnicity / religion
    ("racist", r"\b(neger|nigger|mohr|kanake|schlitzauge|chinesen koennen|"
               r"alle juden|alle muslime|alle christen|judenwitz|juden sind)\b"),
    ("racist", r"\b(asiaten alle|afrikaner alle|zigeuner|sinti raus|roma raus)\b"),
    ("xenophobic", r"\b(auslaender raus|auslander raus|abschieben alle|"
                   r"umvolkung|grosser austausch|remigration)\b"),
    # ableism
    ("ableist", r"\b(spast|spasti|spasten|mongo|mongoloid|behinderter witz|"
                r"geisteskrank weil|kruepp el|kruppel)\b"),
    ("ableist", r"\b(autisten alle|autisten sind|adhs witz)\b"),
    # crime instigation / violence
    ("crime_instigation", r"\b(stiehl|klau das|schlag ihn|schlag sie|"
                          r"toete ihn|toete sie|umbringen|abstechen|erschiessen|"
                          r"bauanleitung bombe|wie baut man eine bombe|drogen kaufen wo)\b"),
    ("violence_glorify", r"\b(sollen alle sterben|gehoert vergast|"
                         r"haengen sollte man|ans messer liefern)\b"),
    # self-harm
    ("self_harm", r"\b(bring dich um|toete dich selbst|haeng dich auf|"
                  r"spring vom dach|schneide dich|ritzen tut)\b"),
    # partisan politics
    ("partisan", r"\b(afd|cdu|spd|gruene waehl|fdp|linke partei|"
                 r"trump|biden|harris|merkel|scholz|merz)\b"),
    # broad real-person attack template
    ("real_person_attack", r"\b(\w+ ist ein idiot|\w+ ist eine idiotin|"
                           r"\w+ gehoert weg)\b"),
]

_COMPILED = [(cat, re.compile(pat, flags=re.IGNORECASE)) for cat, pat in _PATTERNS]


@dataclass(frozen=True)
class SafetyHit:
    category: str
    pattern: str
    text: str

    def __str__(self) -> str:
        return f"[{self.category}] {self.text!r} matched /{self.pattern}/"


def _normalise(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    no_accent = "".join(c for c in nfkd if not unicodedata.combining(c))
    return no_accent.lower()


def check(text: str) -> SafetyHit | None:
    """Return a SafetyHit if text trips a guard, else None."""
    if not text:
        return None
    needle = _normalise(text)
    for category, regex in _COMPILED:
        if regex.search(needle):
            return SafetyHit(category=category, pattern=regex.pattern, text=text)
    return None
