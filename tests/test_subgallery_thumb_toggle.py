from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_subgallery_all_observation_images_support_thumb_toggle():
    content = (_project_root() / "templates" / "subgallery.html").read_text(
        encoding="utf-8"
    )

    assert 'data-thumb-src="{{ obs.cover_detection.display_url }}"' in content
    assert 'data-full-src="{{ obs.cover_detection.full_url }}"' in content
    assert 'data-thumb-src="{{ det.display_url }}"' in content
    assert 'data-full-src="{{ det.full_url }}"' in content


def test_subgallery_can_focus_observation_from_focus_query():
    content = (_project_root() / "templates" / "subgallery.html").read_text(
        encoding="utf-8"
    )

    assert "const focusObservationId = {{ focus_observation_id | tojson }};" in content
    assert "const focusDetectionId = {{ focus_detection_id | tojson }};" in content
    assert "document.getElementById('observation-' + focusObservationId)" in content
    assert "highlightObservationTarget(targetEl);" in content
    assert "openObsFilmstrip(targetEl);" in content
    assert "targetEl.getAttribute('data-has-filmstrip') === 'true'" in content
    assert "document.getElementById('filmstrip-item-' + focusDetectionId)" in content
    assert "highlightFocusedDetection(detectionEl);" in content
    assert "wm-tile--focus-target" in content
    assert "wm-tile--focus-target-fade" in content
    assert "obs-filmstrip__item--focus-target" in content
    assert "obs-filmstrip__item--focus-target-fade" in content
    assert "10000" in content


def test_subgallery_uses_today_specific_heading():
    content = (_project_root() / "templates" / "subgallery.html").read_text(
        encoding="utf-8"
    )

    assert "All observations from today" in content
    assert "All Observations" not in content


def test_single_image_observations_open_modal_instead_of_filmstrip():
    content = (_project_root() / "templates" / "subgallery.html").read_text(
        encoding="utf-8"
    )

    assert "function openObservationModal(card)" in content
    assert "function handleObservationCardClick(card, event)" in content
    assert "if (card.getAttribute('data-has-filmstrip') === 'true')" in content
    assert "openObservationModal(card);" in content
    assert 'data-observation-card="true"' in content
    assert "event.target.closest('[data-observation-card=\"true\"]')" in content
    assert 'data-close-filmstrip="{{ obs.observation_id }}"' in content
    assert "closeFilmstrip(closeBtn.dataset.closeFilmstrip);" in content
    assert 'onclick="handleObservationCardClick(' not in content
    assert 'onclick="event.stopPropagation(); closeFilmstrip(' not in content
    assert "{% if obs.photo_count > 1 %}" in content


def test_subgallery_filmstrips_use_detection_timestamps_and_nav_uses_gallery_order():
    content = (_project_root() / "web" / "web_interface.py").read_text(
        encoding="utf-8"
    )

    assert '"image_timestamp": ts,' in content
    assert 'all_dets_enriched.sort(' in content
    assert 'det.get("image_timestamp", "")' in content
    assert "nav_index_by_detection_id" in content
    assert "for obs in enriched_observations" in content
    assert "for det in obs[\"all_detections\"]" in content
    assert "Modal navigation should follow the visible gallery sequence" in content


def test_subgallery_time_sort_uses_observation_end_time():
    content = (_project_root() / "web" / "web_interface.py").read_text(
        encoding="utf-8"
    )

    assert 'observations_all.sort(key=lambda o: o["end_time"])' in content
    assert 'observations_all.sort(key=lambda o: o["end_time"], reverse=True)' in content
