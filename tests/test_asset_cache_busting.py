"""The editor's cache-buster must move when the editor does.

``base.html`` loads ``bird_editor.js`` with a ``?v=`` token. Browsers key
their cache on the full URL, so shipping new JS under an unchanged token
leaves every returning visitor on the old file — the markup updates from
the template while the behaviour silently does not.

This pins the token against the file's content hash: change the script,
and the test tells you to move the token in the same commit.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_BASE = _ROOT / "templates/base.html"
_EDITOR = _ROOT / "assets/js/bird_editor.js"

# Update BOTH when bird_editor.js changes: bump the ?v= token in base.html,
# then paste the digest the failure message prints.
_EXPECTED_EDITOR_DIGEST = "ee99517fb25dbc44"
_EXPECTED_TOKEN = "20260921-box-verdict-in-view-v8"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _token() -> str:
    match = re.search(r"bird_editor\.js\?v=([^\"']+)", _BASE.read_text())
    assert match, "base.html no longer loads bird_editor.js with a ?v= token"
    return match.group(1)


def test_editor_cache_token_matches_the_shipped_file():
    digest = _digest(_EDITOR)
    token = _token()

    assert (digest, token) == (_EXPECTED_EDITOR_DIGEST, _EXPECTED_TOKEN), (
        "bird_editor.js changed without moving its cache-buster (or vice versa).\n"
        f"  current digest: {digest}\n"
        f"  current token:  {token}\n"
        "Bump the ?v= token in templates/base.html, then update both constants "
        "in this test."
    )
