"""Run the page-change feedback script suite with the regular test run."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_nav_feedback_script_suite() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    result = subprocess.run(
        [node, "--test", "tests/nav_feedback.test.cjs"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
