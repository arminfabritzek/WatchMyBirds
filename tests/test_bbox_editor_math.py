"""Executable geometry contracts for direct-on-image bbox correction."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MATH_MODULE = ROOT / "assets" / "js" / "bbox_editor_math.js"


def _node_eval(expression: str) -> object:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    script = (
        f"const math = require({json.dumps(str(MATH_MODULE))});"
        f"process.stdout.write(JSON.stringify({expression}));"
    )
    completed = subprocess.run(
        [node, "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


@pytest.mark.parametrize(
    ("handle", "dx", "dy", "expected"),
    [
        ("n", 0.0, -0.05, {"x": 0.2, "y": 0.15, "w": 0.3, "h": 0.35}),
        ("s", 0.0, 0.05, {"x": 0.2, "y": 0.2, "w": 0.3, "h": 0.35}),
        ("e", 0.05, 0.0, {"x": 0.2, "y": 0.2, "w": 0.35, "h": 0.3}),
        ("w", -0.05, 0.0, {"x": 0.15, "y": 0.2, "w": 0.35, "h": 0.3}),
        ("ne", 0.05, -0.05, {"x": 0.2, "y": 0.15, "w": 0.35, "h": 0.35}),
        ("nw", -0.05, -0.05, {"x": 0.15, "y": 0.15, "w": 0.35, "h": 0.35}),
        ("se", 0.05, 0.05, {"x": 0.2, "y": 0.2, "w": 0.35, "h": 0.35}),
        ("sw", -0.05, 0.05, {"x": 0.15, "y": 0.2, "w": 0.35, "h": 0.35}),
    ],
)
def test_each_edge_and_corner_resizes_only_its_sides(
    handle: str,
    dx: float,
    dy: float,
    expected: dict[str, float],
) -> None:
    result = _node_eval(
        f"math.resizeBox({{x:0.2,y:0.2,w:0.3,h:0.3}},"
        f"{json.dumps(handle)},{dx},{dy},0.01)"
    )

    assert result == pytest.approx(expected)


def test_move_clamps_the_whole_box_inside_the_image() -> None:
    result = _node_eval(
        "math.resizeBox({x:0.8,y:0.8,w:0.15,h:0.15},'move',0.4,0.4,0.01)"
    )

    assert result == pytest.approx({"x": 0.85, "y": 0.85, "w": 0.15, "h": 0.15})


def test_resize_preserves_a_minimum_visible_box_at_frame_edge() -> None:
    result = _node_eval(
        "math.resizeBox({x:0.2,y:0.2,w:0.3,h:0.3},'nw',0.5,0.5,0.04)"
    )

    assert result == pytest.approx({"x": 0.46, "y": 0.46, "w": 0.04, "h": 0.04})
