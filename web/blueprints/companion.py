"""Companion v1 API endpoints.

Mounted under ``/api/v1/companion/*``. All routes require auth via the
existing ``login_required`` decorator. Mutating endpoints inherit the
same CSRF semantics as the rest of the v1 API.

Service injection: ``init_companion_bp(service)`` is called once from
``web_interface`` after the lease and the inference adapter have been
wired. The blueprint stores the service reference on its own object so
each request handler can read it. When ``COMPANION_ENABLED=false`` the
service still exists but every generation path returns ``disabled``,
so the API surface stays predictable for clients.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from logging_config import get_logger
from web.blueprints.auth import login_required

logger = get_logger(__name__)
companion_bp = Blueprint("companion_v1", __name__, url_prefix="/api/v1/companion")


def init_companion_bp(service) -> None:
    companion_bp.companion_service = service  # type: ignore[attr-defined]


def _service():
    svc = getattr(companion_bp, "companion_service", None)
    if svc is None:
        return None
    return svc


def _bad_request(reason: str):
    return jsonify({"error": "bad_request", "reason": reason}), 400


def _unavailable(reason: str, *, extra: dict | None = None):
    body = {"error": "service_unavailable", "reason": reason}
    if extra:
        body.update(extra)
    return jsonify(body), 503


def _busy(reason: str):
    return jsonify({"error": "busy", "reason": reason}), 409


def _coerce_lang(value) -> str:
    lang = str(value or "de").strip().lower()
    return "en" if lang == "en" else "de"


def _coerce_tone(value) -> str:
    tone = str(value or "kid_friendly").strip().lower()
    return "adult_dry" if tone == "adult_dry" else "kid_friendly"


@companion_bp.route("/state", methods=["GET"])
@login_required
def state():
    svc = _service()
    if svc is None:
        return _unavailable("companion service not initialised")
    try:
        return jsonify(svc.state())
    except Exception as exc:
        logger.error("companion /state error [%s]", type(exc).__name__, exc_info=True)
        return jsonify({"error": "internal", "reason": "state lookup failed"}), 500


@companion_bp.route("/recent", methods=["GET"])
@login_required
def recent():
    svc = _service()
    if svc is None:
        return _unavailable("companion service not initialised")
    try:
        limit_raw = request.args.get("limit", default="50")
        try:
            limit = int(limit_raw)
        except (TypeError, ValueError):
            return _bad_request("limit must be an integer")
        if limit < 1 or limit > 200:
            return _bad_request("limit must be in [1, 200]")
        return jsonify({"entries": svc.recent(limit=limit)})
    except Exception as exc:
        logger.error("companion /recent error [%s]", type(exc).__name__, exc_info=True)
        return jsonify({"error": "internal", "reason": "recent lookup failed"}), 500


@companion_bp.route("/chat", methods=["POST"])
@login_required
def chat():
    svc = _service()
    if svc is None:
        return _unavailable("companion service not initialised")
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message") or "").strip()
    if not message:
        return _bad_request("message is required")
    if len(message) > 1000:
        return _bad_request("message too long (max 1000 chars)")
    language = _coerce_lang(payload.get("language"))
    tone = _coerce_tone(payload.get("tone"))
    try:
        result = svc.chat(message=message, language=language, tone=tone)  # type: ignore[arg-type]
    except Exception as exc:
        logger.error("companion /chat error [%s]", type(exc).__name__, exc_info=True)
        return jsonify({"error": "internal", "reason": "chat failed"}), 500
    return _format_result(result)


@companion_bp.route("/event", methods=["POST"])
@login_required
def event():
    svc = _service()
    if svc is None:
        return _unavailable("companion service not initialised")
    payload = request.get_json(silent=True) or {}
    species = str(payload.get("species") or "").strip()
    if not species:
        return _bad_request("species is required")
    try:
        count = int(payload.get("count", 1))
    except (TypeError, ValueError):
        return _bad_request("count must be an integer")
    rare = bool(payload.get("rare", False))
    language = _coerce_lang(payload.get("language"))
    tone = _coerce_tone(payload.get("tone"))
    try:
        result = svc.event(
            species=species,
            count=count,
            rare=rare,
            language=language,  # type: ignore[arg-type]
            tone=tone,  # type: ignore[arg-type]
        )
    except Exception as exc:
        logger.error("companion /event error [%s]", type(exc).__name__, exc_info=True)
        return jsonify({"error": "internal", "reason": "event failed"}), 500
    return _format_result(result)


@companion_bp.route("/feedback", methods=["POST"])
@login_required
def feedback():
    svc = _service()
    if svc is None:
        return _unavailable("companion service not initialised")
    payload = request.get_json(silent=True) or {}
    trigger_id = str(payload.get("trigger_id") or "").strip()
    if not trigger_id:
        return _bad_request("trigger_id is required")
    vote_raw = payload.get("vote")
    if vote_raw is None or vote_raw == "":
        vote = None
    elif vote_raw in ("up", "down"):
        vote = vote_raw
    else:
        return _bad_request("vote must be 'up', 'down', or null")
    try:
        updated = svc.feedback(trigger_id=trigger_id, vote=vote)
    except ValueError as exc:
        return _bad_request(str(exc))
    except Exception as exc:
        logger.error(
            "companion /feedback error [%s]", type(exc).__name__, exc_info=True
        )
        return jsonify({"error": "internal", "reason": "feedback failed"}), 500
    if updated is None:
        return jsonify({"error": "not_found", "reason": "trigger_id unknown"}), 404
    return jsonify({"ok": True, "entry": updated})


def _format_result(result: dict):
    """Map service result dict to HTTP response."""
    status = result.get("status")
    if result.get("ok"):
        return jsonify(result)
    if status == "disabled":
        return _unavailable("companion disabled", extra=result)
    if status == "busy":
        return _busy(result.get("reason") or "lease busy")
    # unreachable / timeout / non_ok
    return _unavailable(result.get("reason") or "inference failed", extra=result)
