from pathlib import Path


def test_analytics_species_table_includes_hour_axis():
    project_root = Path(__file__).resolve().parent.parent
    template_path = project_root / "templates" / "analytics.html"
    content = template_path.read_text(encoding="utf-8")

    assert "species-table__axis-row" in content
    assert "species-table__axis-labels" in content
    assert content.count("range(24)") >= 2


def test_analytics_template_includes_event_intelligence_section():
    project_root = Path(__file__).resolve().parent.parent
    template_path = project_root / "templates" / "analytics.html"
    content = template_path.read_text(encoding="utf-8")

    assert "Event Intelligence" in content
    assert "Representative Budget" in content
    assert "event_intelligence.largest_events" in content
    assert "event_intelligence.species_pressure" in content
