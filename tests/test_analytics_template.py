from pathlib import Path


def test_analytics_species_table_includes_hour_axis():
    project_root = Path(__file__).resolve().parent.parent
    template_path = project_root / "templates" / "analytics.html"
    content = template_path.read_text(encoding="utf-8")

    assert "species-table__axis-row" in content
    assert "species-table__axis-labels" in content
    assert content.count("range(24)") >= 2
