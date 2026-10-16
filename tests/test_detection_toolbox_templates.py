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

    assert (
        "{% macro render_modal(det, group_id, nav_scope=none, nav_index=none) %}"
        in content
    )
    assert 'data-nav-scope="{{ nav_scope }}"' in content
    assert 'data-nav-index="{{ nav_index }}"' in content
    assert "wikipedia_species_url(" in content
    assert "Manually identified" in content
    # The AI-status labels that used to be pinned here were assigned but never
    # rendered; the string assertions kept the dead block alive. Anything about
    # the head's visible text belongs in the rendering tests named below.


# The head's species/provenance/confidence rules are asserted on rendered
# output in test_detail_modal_unknown_species_head.py: pinned as source
# strings they passed while the head showed a withdrawn species.


def test_detection_modal_uses_one_shared_bird_editor():
    modal = _read("templates/components/detection_modal.html")
    viewer = _read("templates/components/modal_image_viewer.html")
    toolbar = _read("templates/components/bird_editor_toolbar.html")
    editor = _read("assets/js/bird_editor.js")

    assert "render_bird_editor_toolbar(det, can_moderate=can_moderate)" in modal
    assert "tile_toolbox(" not in modal
    assert "data-current-detection=" in viewer
    assert "data-bird-editor-layer" in viewer
    assert "data-editor-object-select" in toolbar
    assert 'data-editor-action="adjust"' in toolbar
    assert 'data-editor-action="add"' in toolbar
    assert "window.WmSpeciesPicker.pickSpecies" in editor
    assert "allowUnknown: true" in editor


def test_detection_modals_defer_full_image_load_until_modal_open():
    detection_modal = _read("templates/components/detection_modal.html")
    viewer = _read("templates/components/modal_image_viewer.html")
    css = _read("assets/design-system.css")
    js = _read("assets/js/gallery_utils.js")

    assert "defer_src=true" in detection_modal
    assert "data-deferred-src" in viewer
    assert "wm-image-viewer--loading" in viewer
    assert ".wm-image-viewer--loading" in css
    assert "loadDeferredViewerImages" in js
    assert "'show.bs.modal'" in js


def test_deferred_viewer_images_use_same_origin_sanitizer():
    js = _read("assets/js/gallery_utils.js")

    assert "function safeSameOriginImagePath" in js
    assert "parsed.origin !== window.location.origin" in js
    assert "img.src = safeTarget" in js
    assert "img.setAttribute('src', target)" not in js


def test_tile_details_navigation_uses_same_origin_sanitizer():
    # safeSameOriginPath now lives in gallery_utils.js (loaded first) so it can
    # be modelled as a CodeQL barrier; tile_actions.js consumes it.
    shared = _read("assets/js/gallery_utils.js")
    js = _read("assets/js/tile_actions.js")

    assert "function safeSameOriginPath" in shared
    assert "parsed.origin !== window.location.origin" in shared
    assert "safeSameOriginPath(" in js
    assert "window.location.assign(safeDetailsPath)" in js
    assert "window.location.href = detailsHref" not in js


def test_tile_actions_pages_load_gallery_utils_first():
    """tile_actions.js depends on safeSameOriginPath from gallery_utils.js.

    Every template that loads tile_actions.js must load gallery_utils.js before
    it, or the shared sanitizer is undefined at runtime.
    """
    templates_dir = _project_root() / "templates"
    offenders = []
    for tpl in templates_dir.rglob("*.html"):
        text = tpl.read_text(encoding="utf-8")
        ta = text.find('src="/assets/js/tile_actions.js')
        if ta == -1:
            continue
        gu = text.find('src="/assets/js/gallery_utils.js')
        if gu == -1 or gu > ta:
            offenders.append(str(tpl.relative_to(_project_root())))
    assert not offenders, (
        "These templates load tile_actions.js without gallery_utils.js first:\n  "
        + "\n  ".join(offenders)
    )


def test_modal_action_bar_data_actions_are_dispatched():
    action_bar = _read("templates/components/modal_action_bar.html")
    viewer = _read("templates/components/modal_image_viewer.html")
    detection_modal = _read("templates/components/detection_modal.html")
    js = _read("assets/js/tile_actions.js")

    # The bar has an in_header variant
    # (class="modal-action-bar modal-action-bar--in-header"), so the
    # historical exact-match assertion would only catch the legacy
    # footer-rendered form. Match the class as a prefix and verify the
    # in-header variant lives on the same element.
    assert 'class="modal-action-bar' in action_bar
    assert "modal-action-bar--in-header" in action_bar
    assert 'data-action="toggle-menu"' in action_bar
    assert 'data-action="toggle-smart-zoom"' not in action_bar
    assert 'data-action="toggle-bbox-overlay"' in action_bar
    assert 'data-action="toggle-modal-maximize"' in action_bar
    assert 'data-action="move-trash"' not in action_bar
    toolbar = _read("templates/components/bird_editor_toolbar.html")
    assert 'data-action="set-smart-zoom"' in toolbar
    assert 'data-view-mode="zoom"' in toolbar
    assert 'data-view-mode="full"' in toolbar
    assert "show_view_mode=false" in detection_modal
    assert "show_more=false" in detection_modal
    assert "actionEl.closest('.modal-action-bar')" in js
    assert "actionEl.closest('.wm-view-mode-toggle')" in js
    assert "case 'set-smart-zoom':" in js
    assert "case 'toggle-bbox-overlay':" in js
    assert "case 'toggle-modal-maximize':" in js


def test_gallery_view_mode_is_an_explicit_segmented_choice():
    toggle = _read("templates/partials/thumb_view_toggle.html")
    css = _read("assets/design-system.css")

    assert 'data-thumb-view-mode="crop"' in toggle
    assert 'data-thumb-view-mode="full"' in toggle
    assert "window.setThumbView" in toggle
    assert "wmb_thumb_view" in toggle
    assert ".wm-view-mode-toggle--gallery" in css
    assert ".wm-view-mode-toggle--viewer" in css


def test_toolbox_escape_closes_only_the_open_menu():
    js = _read("assets/js/tile_actions.js")

    assert "if (event.key === 'Escape' && openDropdown)" in js
    assert "event.stopPropagation();" in js
    assert "toggle.focus();" in js
    assert "}, true); // Capture before Bootstrap modal Escape handling" in js


def test_species_picker_escape_does_not_reach_the_parent_modal():
    picker = _read("assets/js/species_picker.js")
    escape_branch = picker.split("if (e.key === 'Escape' || e.key === 'ArrowLeft')")[
        1
    ].split("return;", 1)[0]

    assert "e.preventDefault();" in escape_branch
    assert "e.stopImmediatePropagation();" in escape_branch
    assert escape_branch.index("e.stopImmediatePropagation();") < escape_branch.index(
        "cancel();"
    )
    assert "document.addEventListener('keydown', onKey, true);" in picker
    assert "document.removeEventListener('keydown', onKey, true);" in picker


def test_detection_info_hides_decision_badges_after_manual_species_review():
    content = _read("templates/components/modal_detection_info.html")

    # The single-detection branch is gone (the header subtitle in
    # detection_modal.html now carries species + confidence + AI-status
    # for the entry-point detection). Only the multi-bird sibling-strip
    # branch remains; the manual-species-review guard still has to
    # suppress decision badges per sibling when human approval has
    # landed.
    assert (
        "sib_has_manual_species_review = (sib.human_species_source == 'manual') and sib.human_species_key"
        in content
    )
    assert "sib_has_manual_review_approval = sib_has_manual_species_review" in content
    assert "{% if sib_has_manual_review_approval %}" in content
    assert (
        "{% elif sib.decision_state and not sib_has_manual_review_approval %}"
        in content
    )
    assert "🤖 AI confirmed" in content
    assert "🤖 AI uncertain" in content
    assert "🤖 AI unknown" in content
    assert "🧑👍 manually confirmed" in content


def test_species_templates_use_toolbox_without_legacy_badge():
    species_content = _read("templates/species.html")
    overview_content = _read("templates/species_overview.html")

    assert (
        "{% from 'partials/tile_toolbox.html' import tile_toolbox %}" in species_content
    )
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
    assert content.count("{{ tile_toolbox(") >= 1
    assert "quiet-preview__item fade-in-item wm-toolbox-host" in content
    assert "best-species-board__featured" in content
    assert 'class="best-species-card__media wm-toolbox-host"' in content
    assert "best-species-board__grid" in content
    assert "details_href='/gallery/' ~ frame.gallery_date" in content
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


def test_no_bird_action_renders_on_browse_and_review_surfaces():
    """The full-image action is available where its whole-frame scope is supported.

    Species is intentionally excluded: the user is already inside a
    single-species view and the action is frame-wide, which is the
    wrong mental model on that surface.
    """
    toolbox = _read("templates/partials/tile_toolbox.html")

    # Gate covers Review-Queue, Gallery, and detail modals — Species omitted by design.
    assert "surface in ('review', 'gallery', 'detail_modal')" in toolbox
    # Action attribute and filename binding are present
    assert 'data-action="review-no-bird"' in toolbox
    assert 'data-filename="{{ filename }}"' in toolbox
    assert "No birds in full image" in toolbox
    # allow_review_no_bird parameter still guards the button
    assert "allow_review_no_bird" in toolbox
    # Trash and Species surfaces are excluded (not in the allowed tuple)
    gate_body = toolbox.split("surface in ('review', 'gallery', 'detail_modal')")[
        1
    ].split("endif")[0]
    assert "'trash'" not in gate_body
    assert "'species'" not in gate_body


def test_no_bird_action_not_on_trash_or_species_surface():
    """Trash and Species tiles must never show the no-bird button.

    Asserts that the surface-gate literal for the no-bird action does
    NOT contain 'trash' or 'species'. A future widening of the gate
    must consciously rewrite this assertion.
    """
    toolbox = _read("templates/partials/tile_toolbox.html")
    gate = "surface in ('review', 'gallery', 'detail_modal')"
    assert gate in toolbox, "no-bird gate literal moved or changed shape"
    assert "'trash'" not in gate
    assert "'species'" not in gate


def test_detail_modal_keeps_editing_in_the_fixed_editor_toolbar():
    modal = _read("templates/components/detection_modal.html")
    toolbar = _read("templates/components/bird_editor_toolbar.html")
    editor = _read("assets/js/bird_editor.js")
    base = _read("templates/base.html")

    assert 'data-editor-action="adjust"' in toolbar
    assert 'data-editor-action="add"' in toolbar
    assert 'data-editor-action="cancel"' in toolbar
    assert 'data-editor-action="save"' in toolbar
    assert 'data-filename="{{ image_filename }}"' in toolbar
    assert "bbox_editor_math.js" in base
    assert "bird_editor.js" in base
    assert "render_bird_editor_toolbar" in modal
    assert "'/api/manual-objects'" in editor
    assert "'/api/labels/answer'" in editor


def test_saved_existing_box_changes_only_the_geometry_axis():
    editor = _read("assets/js/bird_editor.js")
    save_flow = editor.split("async function save()", 1)[1].split(
        "async function retractManualObject()", 1
    )[0]
    payload = save_flow.split("'/api/labels/answer'", 1)[1].split("});", 1)[0]

    assert "bbox_correction" in editor
    assert "object_bird_presence" not in payload
    assert "bbox_quality" not in payload


def test_bbox_arrow_keys_do_not_reach_modal_navigation():
    """Moving a focused box must not also switch to another detection."""
    editor = _read("assets/js/gallery_utils.js")
    keydown_handler = editor.split("boxEl.addEventListener('keydown'")[1].split(
        "});", 1
    )[0]

    assert "event.preventDefault();" in keydown_handler
    assert "event.stopPropagation();" in keydown_handler


def test_gallery_utils_cache_key_reaches_every_image_surface():
    """Every image surface must request the current shared viewer logic."""
    templates = (
        "templates/edit.html",
        "templates/review_grid.html",
        "templates/canonical_dataset_box_walkthrough.html",
        "templates/species.html",
        "templates/species_overview.html",
        "templates/stream.html",
        "templates/subgallery.html",
        "templates/trash.html",
    )

    for template in templates:
        assert "gallery_utils.js?v={{ asset_version }}" in _read(template), template


def test_offered_box_labels_choose_visible_space_around_the_box():
    """Interactive species controls must not always cover the box at top-left."""
    js = _read("assets/js/gallery_utils.js")

    assert "function _positionInteractiveBboxLabel" in js
    assert "name: 'above'" in js
    assert "name: 'below'" in js
    assert "name: 'right'" in js
    assert "name: 'left'" in js
    assert "candidate.space > best.space" in js
    assert "label.dataset.placement = placement.name" in js
    assert "viewport.right - labelRect.width - inset" in js
    assert "candidate.fits && !candidate.overlapsBox && !candidate.overlapsLabel" in js
    assert "layer.style.transform = '';" in js
    assert "const renderedScaleX = imageRect.width / geometry.elementW;" in js
    assert "labelLayer.style.transform = '';" in js

    css = _read("assets/design-system.css")
    assert "@media (hover: hover) and (pointer: fine)" in css
    assert ".wm-bbox-label__confirm:hover" in css
    assert ".wm-bbox-label__picker:hover" in css
    assert "border: 1px solid #f5c84c" in css
    assert "const strokeCurrent = 1 * inv;" in js
    assert "const strokeOther = 0.75 * inv;" in js

    editor = _read("assets/js/bird_editor.js")
    assert "function positionShapeLabel" in editor
    assert "const previousPlacement = label.dataset.placement;" in editor
    assert "candidate.name === previousPlacement" in editor
    assert "img.addEventListener('transitionend', syncFinishedTransform);" in editor
    assert (
        "candidate.fits && !candidate.overlapsBox && !candidate.overlapsLabel" in editor
    )
    assert "--wm-bird-editor-stroke" in editor
    assert "--wm-bird-editor-soft-stroke" in editor
    assert "label.style.transform = 'scale(' + inverse + ')'" in editor
    assert "button.wm-bird-editor__box-label-name:hover" in css
    assert ".wm-bird-editor__box-label-picker:hover" in css


def test_adjust_box_preserves_focus_and_handles_keep_screen_size():
    editor = _read("assets/js/bird_editor.js")
    css = _read("assets/design-system.css")
    begin_edit = editor.split("function beginEdit()", 1)[1].split(
        "function beginAdd()", 1
    )[0]
    begin_add = editor.split("function beginAdd()", 1)[1].split(
        "function cancelEdit", 1
    )[0]

    assert "forceFull();" not in begin_edit
    assert "forceFull();" in begin_add
    assert "--wm-bird-editor-handle-scale" in editor
    assert "transform: scale(var(--wm-bird-editor-handle-scale, 1));" in css
    assert "transform-origin: center;" in css


def test_no_bird_js_fallback_is_surface_agnostic():
    """tile_actions.js must contain the noBirdFrame fallback for non-review surfaces."""
    js = _read("assets/js/tile_actions.js")
    base = _read("templates/base.html")
    dialog = _read("templates/partials/no_bird_confirm_dialog.html")
    css = _read("assets/design-system.css")

    assert "async function noBirdFrame(" in js
    assert "'/api/labels/answer'" in js
    assert "image_bird_presence: 'absent'" in js
    assert "window.wmConfirmFullImageNoBird = confirmFullImageNoBird" in js
    assert "noBirdConfirmed" not in js
    assert "NO_BIRD_KEY" not in js
    # Confirmation flow must surface the frame-wide effect in plain
    # language (multi-detection count or generic "all detections").
    # Tolerates both the modal-with-full-frame UX and the legacy
    # confirm() fallback path used by old browsers.
    assert "Every detected object will be answered No bird." in js
    # Modal must include a full-frame preview to prevent the UX trap
    # where the operator confirms based on the cropped tile view.
    assert "data-full-src" in js
    assert "data-deferred-src" in js
    assert "Full image preview is unavailable. Nothing was changed." in js
    assert "A text-only fallback would recreate the crop-context trap." in js
    assert "{% include 'partials/no_bird_confirm_dialog.html' %}" in base
    assert 'class="wm-modal wm-no-bird-confirm"' in dialog
    assert "Confirm no birds in full image" in dialog
    assert 'aria-label="Cancel full-image no-bird confirmation"' in dialog
    assert 'aria-label="Confirm no birds in full image"' in dialog
    no_bird_dialog_flow = js.split("function confirmFrameNoBird", 1)[1].split(
        "async function confirmFullImageNoBird", 1
    )[0]
    assert "document.createElement('dialog')" not in no_bird_dialog_flow
    assert "style.cssText" not in no_bird_dialog_flow
    # The safety preview must use the available viewport instead of shrinking
    # the evidence into the former 720 px confirmation-dialog limit.
    assert ".wm-no-bird-confirm" in css
    assert "width: 95vw" in css
    assert "max-width: none" in css
    assert "min(720px, 95vw)" not in css
    # Tile fade-out on success
    assert "closest('.wm-tile')" in js
    assert "tile.style.opacity = '0'" in js
    # Every surface must use the same safety gate; no legacy callback may bypass it.
    assert "singleAction(filename, 'no_bird')" not in js
    assert "noBirdFrame(filename, actionEl)" in js


def test_tile_action_cache_key_reaches_every_image_surface():
    templates = (
        "templates/edit.html",
        "templates/review_grid.html",
        "templates/canonical_dataset_box_walkthrough.html",
        "templates/species.html",
        "templates/species_overview.html",
        "templates/stream.html",
        "templates/subgallery.html",
        "templates/trash.html",
    )

    for template in templates:
        assert "tile_actions.js?v={{ asset_version }}" in _read(template), template


def test_box_walkthrough_is_a_shared_viewer_scope():
    gallery = _read("assets/js/gallery_utils.js")

    assert "el.closest('.wm-box-walkthrough')" in gallery


def test_bird_editor_keeps_selection_confirmation_and_picker_separate():
    editor = _read("assets/js/bird_editor.js")
    toolbar = _read("templates/components/bird_editor_toolbar.html")
    tile_actions = _read("assets/js/tile_actions.js")

    assert "data-editor-object-select" in toolbar
    assert 'data-editor-action="species"' not in toolbar
    assert "data-editor-confirm" in editor
    assert "object_bird_presence: 'present'" in editor
    assert "species_identity: 'confirmed'" in editor
    assert "data-editor-species" in editor
    assert "wm-species-picker-overlay" in editor
    assert "actionEl.closest('.wm-bird-editor')" in tile_actions
    assert "const savingDraft = clone(draft)" in editor
    assert 'data-editor-action="retract"' in toolbar


def test_homepage_species_cards_share_the_fixed_photo_grid():
    content = _read("templates/stream.html")
    assert "{% set frames = det.story_frames or [det] %}" in content
    assert "{% for frame in frames %}" in content
    assert "best-species-card__photos--single" in content
    assert "variant == 'featured' and det.story_frames" not in content
    assert '<details class="best-species-board__more">' in content
    assert "<summary>Show more</summary>" in content
