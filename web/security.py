"""Shared security helpers for web blueprints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flask import jsonify

from logging_config import get_logger

if TYPE_CHECKING:
    from flask import Response

logger = get_logger(__name__)


def safe_log_value(value: object, max_len: int = 200) -> str:
    """Neutralize CR/LF/tab so user-controlled data cannot forge log lines."""
    text = str(value)
    text = text.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
    text = "".join(c if (c.isprintable() or c == " ") else "?" for c in text)
    if len(text) > max_len:
        text = text[:max_len] + "...[truncated]"
    return text


def error_response(
    public_message: str, exc: BaseException, status: int = 500
) -> tuple[Response, int]:
    """5xx JSON response with ``{"status": "error", "message": ...}``.

    The exception only goes to the log via exc_info; the client sees
    just the public message.
    """
    logger.error("%s [%s]", public_message, type(exc).__name__, exc_info=True)
    return jsonify({"status": "error", "message": public_message}), status


def error_response_simple(
    public_message: str, exc: BaseException, status: int = 500
) -> tuple[Response, int]:
    """Like :func:`error_response` but emits ``{"error": ...}`` (older blueprints)."""
    logger.error("%s [%s]", public_message, type(exc).__name__, exc_info=True)
    return jsonify({"error": public_message}), status


def safe_validation_message(
    exc: BaseException,
    allowed_prefixes: tuple[str, ...],
    fallback: str = "Invalid input",
) -> str:
    """Whitelist-filter for exception messages that need to reach the UI.

    Some endpoints (notably the PTZ empirical-probe wizard) rely on a
    specific validation message — e.g. "Overview preset is not
    configured" — being surfaced to the operator. We don't want the
    raw ``str(exc)`` to flow into the response, because an unrelated
    ValueError elsewhere in the call chain could then leak internal
    detail.

    This helper accepts the exception only if ``str(exc)`` starts with
    one of the ``allowed_prefixes``. Anything else collapses to the
    generic ``fallback`` — the message that leaves this function is
    statically bounded by the prefix list.

    Usage::

        except ValueError as exc:
            msg = safe_validation_message(
                exc,
                allowed_prefixes=("Overview preset",),
                fallback="Probe start rejected",
            )
            return jsonify({"status": "error", "message": msg}), 400
    """
    text = str(exc)
    for prefix in allowed_prefixes:
        if text.startswith(prefix):
            # Cap length so an attacker-controllable suffix cannot
            # smuggle large payloads through a short prefix.
            return text[:200]
    return fallback
