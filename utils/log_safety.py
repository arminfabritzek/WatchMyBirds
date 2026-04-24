"""Twin of :func:`web.security.safe_log_value` for non-web modules.

Kept separate to avoid a ``utils -> web`` cyclic import.
"""

from __future__ import annotations


def safe_log_value(value: object, max_len: int = 200) -> str:
    """Neutralize CR/LF/tab so user-controlled data cannot forge log lines."""
    text = str(value)
    text = text.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
    text = "".join(c if (c.isprintable() or c == " ") else "?" for c in text)
    if len(text) > max_len:
        text = text[:max_len] + "...[truncated]"
    return text
