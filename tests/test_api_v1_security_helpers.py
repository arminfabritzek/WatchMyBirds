"""Unit tests for safe_log_value + error_response.

These two helpers are the choke points for log-injection and
stack-trace-exposure across every Flask route — regressions here
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
        # Raise inside the with-block so exc_info=True picks the
        # active traceback up — that mirrors how every real call site
        # invokes the helper (from inside an ``except`` clause).
        with flask_app.test_request_context(), caplog.at_level(logging.ERROR):
            try:
                raise RuntimeError("sqlite3 error: no such table")
            except RuntimeError as caught:
                _error_response("models/detector GET failed", caught)

        # The formatted message line carries the public message and the
        # exception class only — never the str(exc) text. The full
        # traceback (including str(exc)) still goes to the structured
        # exc_info on the LogRecord, so log handlers configured to
        # render tracebacks keep the operator-facing detail.
        records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert any("models/detector GET failed" in r.getMessage() for r in records)
        assert any("RuntimeError" in r.getMessage() for r in records)
        # The raw exception text is in exc_info, not the formatted message.
        assert all("sqlite3 error" not in r.getMessage() for r in records)
        assert any(
            r.exc_info and isinstance(r.exc_info[1], RuntimeError)
            for r in records
        )

    def test_custom_status_code(self, flask_app):
        with flask_app.test_request_context():
            _, status = _error_response("bad gateway", RuntimeError("x"), status=502)
        assert status == 502
