"""
Authentication Blueprint.

Handles login/logout routes and the login_required decorator.
"""

import logging
from functools import wraps

from flask import Blueprint, redirect, render_template, request, session, url_for

from web.services import auth_service

logger = logging.getLogger(__name__)

auth_bp = Blueprint("auth", __name__)


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
        password = request.form.get("password", "")
        next_url = auth_service.get_redirect_target(request.form.get("next"))

        if auth_service.authenticate(password):
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
