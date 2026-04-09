from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _read(relative_path: str) -> str:
    return (_project_root() / relative_path).read_text(encoding="utf-8")


def test_tile_toolbox_supports_details_href():
    content = _read("templates/partials/tile_toolbox.html")

    assert "details_href=none" in content
    assert "allow_favorite=true" in content
    assert "allow_details=true" in content
    assert "allow_change_species=true" in content
    assert "allow_move_to_trash=true" in content
    assert "allow_review_no_bird=true" in content
    assert "modal_target or details_href" in content
    assert "allow_details and (modal_target or details_href)" in content
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
    assert "has_manual_species_review = (det.species_source == 'manual') and det.manual_species_override" in content
    assert "has_manual_review_approval = has_manual_species_review and (det.review_status == 'confirmed_bird')" in content
    assert "title_species = det.species_key or det.manual_species_override or det.cls_class_name or det.od_class_name" in content
    assert "ai_status_label = '🤖 KI bestaetigt'" in content
    assert "🧑👍 manuell bestaetigt" in content
    assert "🤖 KI unbekannt" in content
    assert "wikipedia_species_url(det.common_name, title_species)" in content


def test_detection_info_hides_decision_badges_after_manual_species_review():
    content = _read("templates/components/modal_detection_info.html")

    assert "sib_has_manual_species_review = (sib.species_source == 'manual') and sib.manual_species_override" in content
    assert "sib_has_manual_review_approval = sib_has_manual_species_review and (sib.review_status == 'confirmed_bird')" in content
    assert "{% if sib_has_manual_review_approval %}" in content
    assert "{% elif sib.decision_state and not sib_has_manual_review_approval %}" in content
    assert "det_has_manual_species_review = (det.species_source == 'manual') and det.manual_species_override" in content
    assert "det_has_manual_review_approval = det_has_manual_species_review and (det.review_status == 'confirmed_bird')" in content
    assert "{% if det_has_manual_review_approval %}" in content
    assert "{% elif det.decision_state and not det_has_manual_review_approval %}" in content
    assert "🤖 KI bestaetigt" in content
    assert "🤖 KI unsicher" in content
    assert "🤖 KI unbekannt" in content
    assert "🧑👍 manuell bestaetigt" in content


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
    assert "source-link-badge" in overview_content
    assert "wikipedia_species_url(" in overview_content
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


def test_review_modal_uses_quick_review_layout():
    content = _read("templates/components/orphan_modal.html")
    stage_content = _read("templates/components/review_stage_panel.html")
    css = _read("assets/design-system.css")
    review_page = _read("templates/orphans.html")
    review_js = _read("assets/js/review_workspace.js")

    assert "review-stage-panel__canvas" in content
    assert 'render_orphan_modal(orphan, true)' in stage_content
    assert 'data-review-panel-action="trash"' in content
    assert 'data-review-panel-action="approve_review"' in content
    assert 'data-review-panel-action="select_species"' in content
    assert 'data-review-panel-action="confirm_species"' in content
    assert 'data-review-panel-action="open_species_picker"' in content
    assert 'data-review-panel-action="deep_scan"' in content
    assert 'data-review-nav="-1"' in content
    assert 'data-review-nav="1"' in content
    assert 'data-review-facts-toggle' in content
    assert 'data-review-facts-panel' in content
    assert 'aria-expanded="true"' in content
    assert 'data-review-global-metric="auto_accepted"' in content
    assert 'data-review-global-metric="manual_confirmed"' in content
    assert 'data-review-global-metric="policy_confirmed"' in content
    assert 'data-review-global-metric="review_queue"' in content
    assert "render_image_viewer(" in content
    assert "tile_toolbox(" in content
    assert "show_viewer_tools=true" in content
    assert "show_boxes=has_bbox" in content
    assert "review-stage-panel__decision-rail" in content
    assert "review-stage-panel__species-strip" in content
    assert "review-stage-panel__species-actions" in content
    assert "species.thumb_url" in content
    assert "review-stage-panel__species-image" in content
    assert "Choose another species" in content
    assert 'data-default-suggestion="{% if species.scientific == orphan.default_species %}1{% else %}0{% endif %}"' in content
    assert "Default suggestion · Click to confirm" not in content
    assert 'data-review-viewer-tool="zoom"' in content
    assert 'data-review-viewer-tool="bbox"' in content
    assert 'data-bbox-review-toggle' in content
    assert 'data-bbox-review-copy' in content
    assert "review-stage-panel__image-frame" in content
    assert "review-stage-panel__facts-toggle" in css
    assert "review-stage-panel__facts-panel" in css
    assert "review-stage-panel__facts-label" in css
    assert ".review-stage-panel__facts-divider" in css
    assert ".review-stage-panel__facts-grid--global" in css
    assert ".review-stage-panel__viewer-media" in css
    assert ".review-stage-panel__decision-rail" in css
    assert ".review-stage-panel__species-strip" in css
    assert ".review-stage-panel__species-actions" in css
    assert ".review-stage-panel__species-media" in css
    assert ".review-stage-panel__species-image" in css
    assert ".review-stage-panel__species-overlay" in css
    assert '.review-stage-panel__species-btn[data-default-suggestion="1"]' in css
    assert '.review-stage-panel__species-btn[data-default-suggestion="1"]::before' in css
    assert ".review-stage-panel__species-btn.is-selected::after" in css
    assert ".review-stage-panel__action--picker" in css
    assert ".review-stage-panel__toggle.is-correct" in css
    assert ".review-stage-panel__toggle.is-wrong" in css
    assert "overflow-wrap: anywhere;" in css
    assert "font-size: 0.72rem;" in css
    assert "{{ img.formatted_date }}" not in review_page
    assert 'onclick="modalAction(' not in content
    assert 'onclick="analyzeAction(' not in content
    assert 'onclick="reviewQuickSpecies(' not in content
    assert 'onclick="stepReviewItem(' not in content
    assert 'onclick="setReviewBboxState(' not in content
    assert 'data-review-controls' in content
    assert ".review-workspace" in css
    assert ".review-stage-panel__species-strip" in css
    assert ".review-stage-panel__action" in css
    assert "grid-template-columns: repeat(2, minmax(0, 1fr));" in css
    assert "aspect-ratio: 4 / 3;" in css
    assert "object-fit: contain;" in css
    assert ".review-stage-panel__image-frame" in css
    assert ".review-stage-panel__viewer-trigger" in css
    assert "overflow: hidden;" in css
    assert "min-height: clamp(360px, 62vh, 820px);" in css
    assert "width: min(100%, 1180px);" in css
    assert "aspect-ratio: 3 / 2;" in css
    assert "grid-template-columns: minmax(180px, 228px) minmax(0, 1fr);" in css
    assert "grid-template-columns: 72px 1fr;" in css
    assert "display: block;" in css
    assert "position: absolute;" in css
    assert "width: 44px;" in css
    assert "height: 44px;" in css
    assert "statusEl.hidden = true;" in review_js
    assert "/assets/js/review_workspace.js?v=4" in review_page
    assert "data-review-item" in review_page
    assert 'id="dsManualConfirmed"' in review_page
    assert 'onchange="toggleOrphanFilter(this)"' not in review_page
    assert "Current · CLS {{ species.score_pct }}%" in content
    assert "CLS {{ species.score_pct }}%" in content
    assert "/assets/js/species_picker.js" in review_page
    assert "/assets/js/tile_actions.js?v=1" in review_page
    assert 'id="reviewActivePanel"' in review_page
    assert 'onclick="selectReviewItem(' not in review_page
    assert "action === 'select_species'" in review_js
    assert "action === 'confirm_species'" in review_js
    assert "/api/review/bbox-review" in review_js
    assert "/api/review/panel/" in review_js
    assert "encodeURIComponent(filename)}?force=1" in review_js
    assert "[data-review-panel-action]" in review_js
    assert "[data-review-nav]" in review_js
    assert "[data-review-facts-toggle]" in review_js
    assert "event.target.closest('[data-review-item]')" in review_js
    assert "event.target.closest('[data-review-nav]')" in review_js
    assert "toggleReviewFacts(factsToggle);" in review_js
    assert "event.target.closest('[data-bbox-review-toggle]')" in review_js
    assert "function getNextReviewBboxState(currentState)" in review_js
    assert "event.target.closest('#hide-orphans')" in review_js
    assert "selectReviewItem(reviewItem.dataset.itemKey);" in review_js
    assert "Manual confirmed " in review_js
    assert "Policy confirmed " in review_js
    assert "data.manual_confirmed_count || 0" in review_js
    assert "function applyDecisionStatsToReviewMetrics(root = document)" in review_js
    assert "let reviewMetricsExpanded = localStorage.getItem('reviewMetricsExpanded') !== 'false';" in review_js
    assert "function applyReviewMetricsState(root = document)" in review_js
    assert "localStorage.setItem('reviewMetricsExpanded', reviewMetricsExpanded ? 'true' : 'false');" in review_js
    assert "applyReviewMetricsState(panel);" in review_js
    assert "applyReviewMetricsState(document);" in review_js
    assert "latestDecisionStats = data;" in review_js
    assert "applyDecisionStatsToReviewMetrics(panel);" in review_js
    assert "applyDecisionStatsToReviewMetrics(document);" in review_js
    assert "const REVIEW_PENDING_SPECIES_KEY = 'reviewPendingSpeciesV1';" in review_js
    assert "function getPendingReviewSpecies(itemKey)" in review_js
    assert "function setPendingReviewSpecies(itemKey, species)" in review_js
    assert "function clearPendingReviewSpecies(itemKey)" in review_js
    assert "persistPending: Boolean(pendingSpecies)" in review_js
    assert "new Image()" in review_js
    assert "hydrateReviewSpeciesThumbs(panel);" in review_js
    assert "async function hydrateReviewSpeciesThumbs()" in review_js
    assert "reviewApprove(" in review_js
    assert "async function confirmReviewSpeciesSelection(actionBtn, panel, itemKey, filename)" in review_js
    assert "prefetchReviewPanel(" in review_js
    assert "prefetchReviewImage(" in review_js
    assert "scheduleReviewPrefetch(itemKey);" in review_js
    assert "requestIdleCallback" in review_js
    assert "[data-review-viewer-tool]" in review_js
    assert "toggleSmartZoom(viewerToolBtn);" in review_js
    assert "toggleBboxOverlay(viewerToolBtn);" in review_js
    assert "action === 'approve_review'" in review_js
    assert "const pendingSpecies = getPendingReviewSpecies(itemKey);" in review_js
    assert "clearPendingReviewSpecies(itemKey);" in review_js
    assert "applyReviewSpeciesUi(controls, species, { origin: 'pending' });" in review_js
    assert "currentPendingSpecies && currentPendingSpecies === species" in review_js
    assert "document.addEventListener('dblclick'" in review_js
    assert "Species selected. Click again to confirm." in review_js
    assert "Species confirmed. Approve when the review is complete." in review_js
    assert "action === 'trash' || action === 'no_bird'" in review_js
    assert "modalAction(itemKey, filename, action, detectionId);" in review_js
    assert "waitForReviewDetectionControls(itemKey, filename);" in review_js
    assert "Deep Scan finished. BBox and species review are now available." in review_js
    assert "Status unavailable" not in review_js
    assert "window.initReviewDefaultBboxes = initReviewDefaultBboxes;" in review_js
    gallery_js = _read("assets/js/gallery_utils.js")
    assert "function getViewerScope(el)" in gallery_js
    assert "function isReviewViewerScope(scope)" in gallery_js
    assert "el.closest('.wm-viewer-scope') || el.closest('.modal')" in gallery_js
    assert "'wmb_review_bbox_pref'" in gallery_js
    assert "'wmb_review_zoom_pref'" in gallery_js
    toolbox = _read("templates/partials/tile_toolbox.html")
    assert "show_viewer_tools=false" in toolbox
    assert "show_boxes=false" in toolbox
    assert "current_bbox_json='{}'" in toolbox
