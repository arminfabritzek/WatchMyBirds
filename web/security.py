"""Shared security helpers for web blueprints.

Two choke points used across every Flask route:

- :func:`safe_log_value` — neutralize CR/LF/tab/controls before
  emitting user-controlled data to logs. Closes CodeQL
  py/log-injection everywhere the helper is used.

- :func:`error_response` — build a 5xx JSON body that carries only
  a caller-supplied public message while the full exception goes to
  the server log. Closes CodeQL py/stack-trace-exposure.

Previously lived inline in web/blueprints/api_v1.py. Moved here so
every blueprint (backup, review, ingest, etc.) can funnel through
the same sanitization without duplicating the logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flask import jsonify

from logging_config import get_logger

if TYPE_CHECKING:
    from flask import Response

logger = get_logger(__name__)


def safe_log_value(value: object, max_len: int = 200) -> str:
    """Neutralize CR/LF/tab in values before emitting them to logs.

    Without this, a request body with
    ``"model_id": "x\\n[ERROR] admin logged in"`` would forge a second
    log line. We replace the control chars with visible escape
    sequences, strip everything else outside printable ASCII, and cap
    the length so log lines stay bounded.
    """
    text = str(value)
    text = text.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
    text = "".join(c if (c.isprintable() or c == " ") else "?" for c in text)
    if len(text) > max_len:
        text = text[:max_len] + "...[truncated]"
    return text


def error_response(
    public_message: str, exc: BaseException, status: int = 500
) -> tuple[Response, int]:
    """Shared 5xx response builder (api_v1 style).

    The full traceback goes to the server log (``exc_info=True``);
    the client receives only the caller-supplied high-level message.
    Previously many routes returned ``str(exc)`` which could leak file
    paths, SQL fragments, or other implementation details that help
    attackers enumerate internals.

    Body shape: ``{"status": "error", "message": <public_message>}``
    — matches the convention used by ``/api/v1/*`` routes.
    """
    logger.error(f"{public_message}: {exc}", exc_info=True)
    return jsonify({"status": "error", "message": public_message}), status


def error_response_simple(
    public_message: str, exc: BaseException, status: int = 500
) -> tuple[Response, int]:
    """Same as :func:`error_response` but emits ``{"error": ...}``.

    Several older blueprints (backup, trash, inbox, ...) ship a JSON
    contract with a single ``error`` key. Use this variant there so
    the existing client code keeps working while still scrubbing the
    exception text from the response.
    """
    logger.error(f"{public_message}: {exc}", exc_info=True)
    return jsonify({"error": public_message}), status
