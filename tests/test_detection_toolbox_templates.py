from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _read(relative_path: str) -> str:
    return (_project_root() / relative_path).read_text(encoding="utf-8")


def test_tile_toolbox_supports_details_href():
    content = _read("templates/partials/tile_toolbox.html")

    assert "details_href=none" in content
    assert "modal_target or details_href" in content
    assert "data-details-href" in content


def test_subgallery_cover_and_filmstrip_use_toolbox():
    content = _read("templates/subgallery.html")

    assert "detection_id=obs.cover_detection.detection_id" in content
    assert (
        "modal_target='#modal-obs' ~ obs.observation_id ~ '-' ~ obs.cover_detection.detection_id"
        in content
    )
    assert 'class="obs-filmstrip__media wm-toolbox-host"' in content
    assert "detection_id=det.detection_id" in content
    assert (
        "modal_target='#modal-obs' ~ obs.observation_id ~ '-' ~ det.detection_id"
        in content
    )
    assert "filmstrip.scrollIntoView" not in content
    assert "partials/rating_badge.html" not in content


def test_subgallery_observation_modals_define_global_nav_scope():
    content = _read("templates/subgallery.html")

    assert "nav_scope='subgallery-all-observations'" in content
    assert "nav_index=det.nav_index" in content


def test_detection_modal_supports_optional_nav_scope_and_index():
    content = _read("templates/components/detection_modal.html")

    assert "{% macro render_modal(det, group_id, nav_scope=none, nav_index=none) %}" in content
    assert 'data-nav-scope="{{ nav_scope }}"' in content
    assert 'data-nav-index="{{ nav_index }}"' in content


def test_species_templates_use_toolbox_without_legacy_badge():
    species_content = _read("templates/species.html")
    overview_content = _read("templates/species_overview.html")

    assert "{% from 'partials/tile_toolbox.html' import tile_toolbox %}" in species_content
    assert "details_href=('/gallery/' ~ det.gallery_date" in species_content
    assert "source-link-badge" in species_content
    assert "wikipedia_species_url(" in species_content
    assert "cover-jump-badge" not in species_content
    assert "partials/rating_badge.html" not in species_content
    assert "{{ tile_toolbox(" in overview_content
    assert "partials/rating_badge.html" not in overview_content


def test_stream_detection_previews_use_toolbox():
    content = _read("templates/stream.html")

    assert "{% from 'partials/tile_toolbox.html' import tile_toolbox %}" in content
    assert content.count("{{ tile_toolbox(") >= 2
    assert "quiet-preview__item fade-in-item wm-toolbox-host" in content
    assert "best-species-board__featured" in content
    assert 'class="best-species-card__media wm-toolbox-host"' in content
    assert "best-species-board__grid" in content
    assert "details_href='/gallery/' ~ det.gallery_date" in content
    assert "filter: drop-shadow(0 2px 4px rgba(0,0,0,0.5));" not in content


def test_edit_tiles_use_toolbox_and_skip_selection_when_using_it():
    content = _read("templates/edit.html")

    assert "{{ tile_toolbox(" in content
    assert "partials/thumb_view_toggle.html" in content
    assert "details_href='/gallery/' ~ date_iso" in content
    assert "event.target.closest('.wm-tile__select, .wm-toolbox')" in content


def test_detection_toolbox_css_and_js_support_scroll_strip_and_sync():
    css = _read("assets/design-system.css")
    js = _read("assets/js/gallery_utils.js")

    assert ".wm-toolbox-host" in css
    assert "grid-auto-flow: column;" in css
    assert "overflow-x: auto;" in css
    assert ".obs-filmstrip__media" in css
    assert '.wm-toolbox__fav[data-detection-id="${detectionId}"]' in js
    assert "const navScope = currentModalEl.getAttribute('data-nav-scope');" in js
    assert '.gallery-modal[data-nav-scope="${navScope}"]' in js
    assert "data-nav-index" in js
    assert "let modalNavigationInFlight = false;" in js
    assert "hidden.bs.modal" in js
    assert "shown.bs.modal" in js


def test_subgallery_focus_styles_live_in_design_system():
    css = _read("assets/design-system.css")

    assert ".wm-tile--focus-target" in css
    assert ".wm-tile--focus-target-fade" in css
    assert ".obs-filmstrip__item--focus-target" in css
    assert ".obs-filmstrip__item--focus-target-fade" in css
    assert "@keyframes subgallery-focus-pulse" in css
