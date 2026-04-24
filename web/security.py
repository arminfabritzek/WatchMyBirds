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
