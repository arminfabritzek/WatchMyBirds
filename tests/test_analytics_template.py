from pathlib import Path


def test_station_report_species_rhythm_includes_hour_axis():
    project_root = Path(__file__).resolve().parent.parent
    template_path = project_root / "templates" / "analytics.html"
    content = template_path.read_text(encoding="utf-8")

    assert "species-rhythm__axis" in content
    assert "<span>00</span>" in content
    assert "<span>24</span>" in content


def test_station_report_uses_event_led_language_and_scoped_retention():
    project_root = Path(__file__).resolve().parent.parent
    template_path = project_root / "templates" / "analytics.html"
    content = template_path.read_text(encoding="utf-8")

    assert "Station Report" in content
    assert "Station activity, without photo inflation" in content
    assert "Verified events" in content
    assert "Every bar counts event starts, never individual photos" in content
    assert "requires measured observation hours" in content
    assert "No files are changed" in content
    assert "Chao1" not in content
    assert "Total Observations" not in content
    assert "photos become optional evidence" not in content
    assert "Object-level reviews captured; event-level verification is still pending" in content
    assert "human object-level species decisions" in content
    assert "<strong>Preliminary</strong>" in content
    assert "model-scored detections" in content
