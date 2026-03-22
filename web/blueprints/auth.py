"""
Authentication Blueprint.

Handles login/logout routes and the login_required decorator.
"""

import logging
import time
from collections import defaultdict
from functools import wraps

from flask import Blueprint, redirect, render_template, request, session, url_for

from web.services import auth_service

logger = logging.getLogger(__name__)

auth_bp = Blueprint("auth", __name__)

# ---------------------------------------------------------------------------
# Simple in-memory rate limiter (no external dependency)
# ---------------------------------------------------------------------------
_login_attempts: dict[str, list[float]] = defaultdict(list)
_RATE_LIMIT_MAX = 5  # attempts
_RATE_LIMIT_WINDOW = 300  # seconds (5 minutes)


def _is_rate_limited(ip: str) -> bool:
    """Return True if the IP has exceeded the login attempt limit."""
    now = time.monotonic()
    # Prune old entries
    _login_attempts[ip] = [t for t in _login_attempts[ip] if now - t < _RATE_LIMIT_WINDOW]
    return len(_login_attempts[ip]) >= _RATE_LIMIT_MAX


def _record_attempt(ip: str) -> None:
    """Record a failed login attempt for the given IP."""
    _login_attempts[ip].append(time.monotonic())


def login_required(f):
    """Decorator to require authentication for Flask routes."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect(url_for("auth.login", next=request.path))
        return f(*args, **kwargs)

    return decorated_function


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    """Login page and authentication handler."""
    error = None
    next_url = auth_service.get_redirect_target(request.args.get("next"))

    if request.method == "POST":
        ip = request.remote_addr

        if _is_rate_limited(ip):
            error = "Too many login attempts. Please try again later."
            logger.warning("Login RATE-LIMITED ip=%s", ip)
            return render_template("login.html", error=error, next_url=next_url)

        password = request.form.get("password", "")
        next_url = auth_service.get_redirect_target(request.form.get("next"))

        if auth_service.authenticate(password):
            session["authenticated"] = True
            session.permanent = True
            # Clear failed attempts on success
            _login_attempts.pop(ip, None)
            logger.info("Login success ip=%s", ip)
            return redirect(next_url)
        else:
            _record_attempt(ip)
            remaining = _RATE_LIMIT_MAX - len(_login_attempts[ip])
            error = "Invalid password. Please try again."
            if remaining <= 2:
                error += f" ({remaining} attempts remaining)"
            logger.warning("Login FAILED ip=%s attempts=%d", ip, len(_login_attempts[ip]))

    return render_template("login.html", error=error, next_url=next_url)


@auth_bp.route("/logout")
def logout():
    """Logout and clear session."""
    session.pop("authenticated", None)
    return redirect("/gallery")
