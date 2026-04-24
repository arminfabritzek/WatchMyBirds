"""Log-output sanitisation for modules outside ``web/``.

Mirror of :func:`web.security.safe_log_value`, duplicated here so
modules in ``utils/`` and ``core/`` can use it without forming a
cyclic import (``utils -> web -> ...``).

Closes CodeQL py/log-injection at the I/O boundary: any value coming
from external sources (request bodies, JSON registries, network
discovery, ...) gets CR/LF/tab replaced by visible escapes and other
C0 controls stripped to ``?`` before the log handler ever sees it.
A poisoned ``model_id = "abc\\n[ERROR] admin escalated"`` therefore
cannot forge a fake log line.
"""

from __future__ import annotations


def safe_log_value(value: object, max_len: int = 200) -> str:
    """Neutralise CR/LF/tab in *value* before emitting to logs.

    See :func:`web.security.safe_log_value` for the full rationale —
    this is a deliberate copy to keep ``utils/`` and ``core/``
    independent of the Flask layer.
    """
    text = str(value)
    text = text.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
    text = "".join(c if (c.isprintable() or c == " ") else "?" for c in text)
    if len(text) > max_len:
        text = text[:max_len] + "...[truncated]"
    return text
