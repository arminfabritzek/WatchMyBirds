from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_stream_exposes_accessible_click_to_aim_controls():
    html = (ROOT / "templates" / "stream.html").read_text(encoding="utf-8")

    assert 'id="ptzAimToggle"' in html
    assert 'title="Aim camera at a point in the live view"' in html
    assert 'aria-label="Aim camera at a point in the live view"' in html
    assert 'id="ptzAimOverlay"' in html
    assert 'id="ptzAimReticle"' in html
    assert 'role="status" aria-live="polite"' in html
    assert "event.key === 'Escape'" in html
    assert "JSON.stringify({ x: x, y: y })" in html
    assert "aimStatusTextEl.textContent = 'Stopping'" in html
    assert "aimStatusTextEl.textContent = 'Stopped'" in html
    assert "PTZ stop requested" in html
