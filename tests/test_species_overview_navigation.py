from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _read_text(rel_path: str) -> str:
    return (_project_root() / rel_path).read_text(encoding="utf-8")


def test_stream_tiles_link_to_species_overview():
    content = _read_text("templates/stream.html")

    assert "/species/overview?species_key=" in content
    assert 'data-bs-target="#modal-summary-' not in content


def test_species_tiles_link_to_species_overview():
    content = _read_text("templates/species.html")

    assert "/species/overview?species_key=" in content
    assert 'data-bs-target="#modal-species_summary-' not in content


def test_species_overview_route_registered():
    content = _read_text("web/web_interface.py")

    assert '"/species/overview"' in content
    assert 'endpoint="species_overview"' in content


def test_species_overview_template_structure():
    content = _read_text("templates/species_overview.html")

    assert (
        "{% extends 'base.html' %}" in content or '{% extends "base.html" %}' in content
    )
    assert "render_modal(det, 'species_overview')" in content
    assert "species-overview-grid" in content
