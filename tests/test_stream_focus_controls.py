from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_stream_mounts_capability_gated_focus_controls():
    html = (ROOT / "templates" / "stream.html").read_text(encoding="utf-8")

    assert 'id="ptzFocusControls"' in html
    assert 'data-axis="focus"' in html
    assert 'id="ptzAutoFocus"' in html
    assert "/focus/capabilities" in html
    assert "/focus/move" in html
    assert "/focus/stop" in html
    assert "/focus/auto" in html
    assert "capabilities.relative ? 'relative' : 'continuous'" in html
    assert "distance: dir * FOCUS_STEP" in html
