"""Unit tests for the api_v1 security helpers.

Guards against CodeQL py/log-injection (#305-307) and
py/stack-trace-exposure (#302-304). The two helpers
(_safe_log_value + _error_response) are the choke points every route
handler in api_v1.py now funnels through. Regressions here would
reintroduce the vulnerabilities.
"""

from __future__ import annotations

import json
import logging

import pytest
from flask import Flask

from web.security import error_response as _error_response
from web.security import safe_log_value as _safe_log_value


class TestSafeLogValue:
    def test_passes_plain_text_unchanged(self):
        assert _safe_log_value("Parus_major") == "Parus_major"

    def test_escapes_newline(self):
        # Without this, an attacker could forge a fake log line by
        # injecting \n followed by a realistic-looking [ERROR] entry.
        assert _safe_log_value("x\nINJECTED") == "x\\nINJECTED"

    def test_escapes_crlf(self):
        assert _safe_log_value("x\r\nevil") == "x\\r\\nevil"

    def test_escapes_tab(self):
        assert _safe_log_value("tab\there") == "tab\\there"

    def test_drops_other_control_chars(self):
        # NUL and BEL are not printable; they become '?'
        assert _safe_log_value("\x00\x07nul-bel") == "??nul-bel"

    def test_truncates_oversized_input(self):
        got = _safe_log_value("a" * 500, max_len=100)
        assert got.startswith("a" * 100)
        assert got.endswith("...[truncated]")
        assert len(got) == 100 + len("...[truncated]")

    def test_handles_non_string_input(self):
        assert _safe_log_value(None) == "None"
        assert _safe_log_value(42) == "42"
        assert _safe_log_value({"k": "v"}) == "{'k': 'v'}"

    def test_preserves_unicode_printables(self):
        # Bird names with umlauts or Latin diacritics must survive.
        assert _safe_log_value("Wacholderdrossel") == "Wacholderdrossel"
        assert _safe_log_value("Motacilla_flava") == "Motacilla_flava"


class TestErrorResponse:
    @pytest.fixture
    def flask_app(self):
        app = Flask(__name__)
        yield app

    def test_returns_public_message_not_exception_str(self, flask_app, caplog):
        # The raw exception carries a sensitive detail — a file path,
        # a SQL snippet, etc. The client must only see the high-level
        # public message.
        secret_exc = RuntimeError("sqlite3 error: no such table at /opt/app/data/images.db")
        with flask_app.test_request_context(), caplog.at_level(logging.ERROR):
            response, status = _error_response("models/detector GET failed", secret_exc)

        assert status == 500
        body = json.loads(response.get_data(as_text=True))
        assert body["status"] == "error"
        assert body["message"] == "models/detector GET failed"
        # Critical: the exception's str() must NOT appear in the body.
        assert "sqlite3" not in body["message"]
        assert "/opt/app" not in body["message"]

    def test_logs_full_exception_on_server_side(self, flask_app, caplog):
        secret_exc = RuntimeError("sqlite3 error: no such table")
        with flask_app.test_request_context(), caplog.at_level(logging.ERROR):
            _error_response("models/detector GET failed", secret_exc)

        # The server operator still needs the full context in the log.
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "models/detector GET failed" in joined
        assert "sqlite3 error" in joined

    def test_custom_status_code(self, flask_app):
        with flask_app.test_request_context():
            _, status = _error_response("bad gateway", RuntimeError("x"), status=502)
        assert status == 502
