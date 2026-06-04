"""Tests for the always-visible footer version line.

Two regressions are guarded here:

1. The commit/date must render from server-side template context, not a
   ``@login_required`` fetch — so it shows logged out as well as logged in.
2. The ``"Unknown"`` sentinel from ``read_build_metadata()`` must be
   normalized to an empty string so the footer simply omits the version
   rather than printing the literal word "Unknown".
"""

from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
from jinja2 import Environment

# The footer markup as it lives in templates/base.html. Kept as a literal
# here so the render logic (the select|join filter and the GitHub-link
# conditionals) is exercised in isolation, without standing up the full
# Flask app and its partial includes.
_FOOTER_SNIPPET = (
    "{% set _footer_ver = [footer_commit, footer_build_date] "
    "| select | join(' · ') %}"
    '{% if _footer_ver %}<small class="footer-version">'
    "{{ _footer_ver }}</small>{% endif %}"
    '<a href="'
    "{% if footer_commit %}"
    "https://github.com/arminfabritzek/WatchMyBirds/commit/{{ footer_commit }}"
    '{% else %}https://github.com/arminfabritzek/WatchMyBirds{% endif %}"'
    ' title="'
    "{% if footer_commit %}Commit {{ footer_commit }}"
    '{% else %}View on GitHub{% endif %}">gh</a>'
)


def _render(**context: object) -> str:
    env = Environment(autoescape=True)
    return env.from_string(_FOOTER_SNIPPET).render(**context)


class TestFooterRender:
    """The footer version block renders from context, independent of auth."""

    def test_shows_commit_and_date(self) -> None:
        html = _render(footer_commit="abc1234", footer_build_date="2026-06-04")
        assert "abc1234 · 2026-06-04" in html
        # GitHub link points at the specific commit.
        assert "/commit/abc1234" in html
        assert 'title="Commit abc1234"' in html

    def test_logged_out_context_still_renders_version(self) -> None:
        # is_authenticated is irrelevant — the footer block never reads it.
        html = _render(
            footer_commit="abc1234",
            footer_build_date="2026-06-04",
            is_authenticated=False,
        )
        assert "abc1234" in html

    def test_commit_only_when_date_empty(self) -> None:
        html = _render(footer_commit="abc1234", footer_build_date="")
        assert "abc1234" in html
        assert "·" not in html  # no separator with a single part

    def test_date_only_when_commit_empty(self) -> None:
        html = _render(footer_commit="", footer_build_date="2026-06-04")
        assert "2026-06-04" in html
        assert "·" not in html
        # No commit → generic repo link, generic title.
        assert 'href="https://github.com/arminfabritzek/WatchMyBirds"' in html
        assert 'title="View on GitHub"' in html

    def test_both_empty_omits_version_small(self) -> None:
        html = _render(footer_commit="", footer_build_date="")
        assert "footer-version" not in html
        # Link still present, just generic.
        assert "github.com/arminfabritzek/WatchMyBirds" in html


class TestUnknownNormalization:
    """The context processor maps the 'Unknown' sentinel to ''.

    Mirrors the normalization in web/web_interface.py inject_security_context
    so the template never has to special-case the literal word.
    """

    @staticmethod
    def _normalize(commit: str, date: str) -> tuple[str, str]:
        # Same expressions as web_interface.py.
        footer_commit = "" if commit == "Unknown" else commit
        footer_build_date = "" if date == "Unknown" else date[:10]
        return footer_commit, footer_build_date

    def test_unknown_becomes_empty(self) -> None:
        commit, date = self._normalize("Unknown", "Unknown")
        assert commit == ""
        assert date == ""
        # And so the footer omits the version entirely.
        assert "footer-version" not in _render(
            footer_commit=commit, footer_build_date=date
        )

    def test_real_values_pass_through(self) -> None:
        commit, date = self._normalize("abc1234", "2026-06-04T10:00:00Z")
        assert commit == "abc1234"
        # Full timestamp is sliced to the date portion.
        assert date == "2026-06-04"


# ---------------------------------------------------------------------------
# End-to-end: the logged-out login page must carry the footer version.
# This is the actual Bug #1 regression — the version used to come from a
# @login_required fetch, so it was blank when logged out.
# ---------------------------------------------------------------------------


@pytest.fixture
def logged_out_app(monkeypatch, tmp_path):
    import config
    from utils import path_manager
    from utils.db import connection as db_connection
    from web.web_interface import create_web_interface

    output_dir = tmp_path / "output"
    ingest_dir = tmp_path / "ingest"
    output_dir.mkdir()
    ingest_dir.mkdir()
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("INGEST_DIR", str(ingest_dir))
    monkeypatch.setenv("EDIT_PASSWORD", "test-password")
    config._CONFIG = None
    db_connection._schema_initialized_paths.clear()
    path_manager._instance = None

    detection_manager = MagicMock()
    detection_manager.frame_lock = nullcontext()
    detection_manager.latest_raw_timestamp = 0.0
    detection_manager.last_good_frame_timestamp = 0.0
    detection_manager._first_frame_received = False

    with (
        patch(
            "web.services.auth_service.should_require_password_setup",
            return_value=False,
        ),
        patch("web.services.auth_service.is_default_password", return_value=False),
        # Pin build metadata so the assertion is deterministic regardless of
        # the host's git state / build files. The factory imports the helper
        # function-locally, so patch it at the source module.
        patch(
            "utils.deploy_info.read_build_metadata",
            return_value={
                "app_version": "9.9.9",
                "git_commit": "deadbee",
                "build_date": "2026-06-04T00:00:00Z",
                "deploy_type": "dev",
            },
        ),
    ):
        app = create_web_interface(detection_manager)
        app.config["TESTING"] = True
        yield app


def test_login_page_shows_footer_version_when_logged_out(logged_out_app):
    with logged_out_app.test_client() as client:
        resp = client.get("/login")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)

    # The pinned commit + sliced date render server-side, no auth required.
    assert "deadbee" in html
    assert "2026-06-04" in html
    # GitHub link points at the commit, not the bare repo.
    assert "/commit/deadbee" in html
