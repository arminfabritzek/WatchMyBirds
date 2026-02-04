"""
Authentication Blueprint.

Handles login/logout routes and the login_required decorator.
"""

import logging
from functools import wraps

from flask import Blueprint, redirect, render_template, request, session, url_for

from config import get_config

logger = logging.getLogger(__name__)

auth_bp = Blueprint("auth", __name__)


def get_edit_password():
    """Gets the current EDIT_PASSWORD from config."""
    cfg = get_config()
    return cfg.get("EDIT_PASSWORD", "")


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
    next_url = request.args.get("next", "/gallery")

    if request.method == "POST":
        password = request.form.get("password", "")
        next_url = request.form.get("next", "/gallery")
        edit_password = get_edit_password()

        if password == (edit_password or ""):
            session["authenticated"] = True
            logger.info("User authenticated successfully.")
            return redirect(next_url)
        else:
            error = "Invalid password. Please try again."
            logger.warning("Failed login attempt.")

    return render_template("login.html", error=error, next_url=next_url)


@auth_bp.route("/logout")
def logout():
    """Logout and clear session."""
    session.pop("authenticated", None)
    return redirect("/gallery")
