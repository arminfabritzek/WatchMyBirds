"""Output cleaner for Companion model responses.

Strips ``<think>...</think>``
blocks, trims chatty role prefixes, and caps the response at the v1
length contract (60-150 tokens; we approximate via sentence count and
character length since adapters do not all return token counts).
"""

from __future__ import annotations

import re

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_ROLE_PREFIX_RE = re.compile(
    r"^\s*(assistant|companion|frieda|pip|walther)\s*:\s*",
    flags=re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# 720 chars ≈ the 150-token contract ceiling at 3-4 chars/token (DE/EN) with
# headroom; anything longer means the model ignored the brevity rule — trim.
MAX_CHARS = 720
MAX_SENTENCES = 6  # 4 dialog turns × ~1.5 sentences, generous buffer


def clean_model_text(text: str) -> str:
    """Return cleaned, length-capped text. Empty string if input empty."""
    if not text:
        return ""
    stripped = _THINK_RE.sub("", text)
    stripped = _ROLE_PREFIX_RE.sub("", stripped.strip())
    collapsed = " ".join(stripped.split())
    sentences = _SENTENCE_SPLIT_RE.split(collapsed)
    capped_sentences = " ".join(sentences[:MAX_SENTENCES]).strip()
    return capped_sentences[:MAX_CHARS]
