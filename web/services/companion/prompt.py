"""Prompt builder for the Companion.

Keeps the system prompt and the message construction in one place so
adapters never have to know about personae, language, or tone tags.

The prompt encodes the locked v1 contract from ``docs/COMPANION_V1.md``:
three personae (Frieda / Pip / Walther), DE or EN as separate modes,
kid_friendly or adult_dry tone, 60-150 token output, no meta-language.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Language = Literal["de", "en"]
Tone = Literal["kid_friendly", "adult_dry"]


@dataclass(frozen=True)
class CompanionContext:
    language: Language = "de"
    tone: Tone = "kid_friendly"
    time_of_day: str | None = None
    weather: str | None = None
    recent_events: tuple[str, ...] = ()
    last_window_summary: str | None = None
    household_context: tuple[str, ...] = ()

    def echo(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "tone": self.tone,
            "time": self.time_of_day,
            "weather": self.weather,
            "recent_events": list(self.recent_events),
            "last_window_summary": self.last_window_summary,
            "household_context": list(self.household_context),
        }


_SYSTEM_PROMPT_DE = (
    "Du bist der WatchMyBirds-Companion. Du sprichst aus drei Stimmen: "
    "Frieda (neugierig, fragt, bemerkt Details), Pip (schnell, schnoddrig, "
    "kurz), Walther (ruhig, sachkundig, trockener Humor). Jede Stimme ist "
    "aus dem Stil allein erkennbar.\n"
    "Regeln: max. zwei Saetze pro Aussage. Keine Erklaerungen, keine "
    "Aufzaehlungen, keine Emojis, kein KI-Ton. Du benutzt den gegebenen "
    "Kontext implizit, du listest ihn nicht auf. Du erfindest keine "
    "Fakten, die nicht im Kontext stehen."
)

_SYSTEM_PROMPT_EN = (
    "You are the WatchMyBirds companion. You speak with three voices: "
    "Frieda (curious, asks, notices small details), Pip (quick, slightly "
    "snippy, short), Walther (calm, knowledgeable, dry wit). Each voice "
    "is recognisable from style alone.\n"
    "Rules: at most two sentences per utterance. No explanations, no "
    "lists, no emojis, no AI tone. Use the given context implicitly, "
    "do not enumerate it. Do not invent facts that are not in the "
    "context."
)


def build_system_prompt(ctx: CompanionContext) -> str:
    base = _SYSTEM_PROMPT_DE if ctx.language == "de" else _SYSTEM_PROMPT_EN
    tone_line = (
        "Ton: spielerisch, fuer Kinder ab 5 verstaendlich, sanfter Humor."
        if ctx.tone == "kid_friendly" and ctx.language == "de"
        else "Tone: playful, easy for a 5-year-old to follow, gentle humour."
        if ctx.tone == "kid_friendly"
        else "Ton: trocken, lakonisch, erwachsen ohne Schaerfe."
        if ctx.language == "de"
        else "Tone: dry, understated, adult without edge."
    )
    return f"{base}\n{tone_line}"


def build_user_message(ctx: CompanionContext, *, message: str) -> str:
    """Compose the user-side payload: a compact context blob plus the
    operator's message or an event description.
    """
    parts: list[str] = []
    if ctx.time_of_day:
        parts.append(f"time={ctx.time_of_day}")
    if ctx.weather:
        parts.append(f"weather={ctx.weather}")
    if ctx.recent_events:
        events = "; ".join(ctx.recent_events[:5])
        parts.append(f"recent_events=[{events}]")
    if ctx.last_window_summary:
        parts.append(f"window={ctx.last_window_summary}")
    if ctx.household_context:
        parts.append("household=" + ",".join(ctx.household_context))
    context_blob = " | ".join(parts) if parts else "(no_context)"
    return f"Context: {context_blob}\nUser: {message}"


def build_messages(ctx: CompanionContext, *, message: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": build_user_message(ctx, message=message)}]
