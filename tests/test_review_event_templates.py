"""Static template / JS / CSS contract tests for the event-based Review surface.

The Queue / Bulk Review mode switch was removed and replaced with a
single biological event panel.
"""

from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _read(relative_path: str) -> str:
    return (_project_root() / relative_path).read_text(encoding="utf-8")


def test_design_system_defines_wong_species_palette():
    """Eight deterministic Wong slots live in :root."""
    content = _read("assets/design-system.css")
    for slot in range(8):
        assert f"--species-colour-{slot}:" in content
    # Reference colours from the Wong (2011) palette.
    assert "#0072B2" in content
    assert "#E69F00" in content
    assert "#009E73" in content


def test_gallery_utils_js_carries_species_palette_and_responsive_label_helper():
    """The JS palette, label helper, and bbox colour wiring are present."""
    content = _read("assets/js/gallery_utils.js")

    # JS-side mirror of the CSS Wong palette so the canvas (which has
    # no DOM ancestor with the CSS var when detached) can colour bboxes.
    assert "const SPECIES_COLOURS" in content
    assert "#0072B2" in content
    assert "#E69F00" in content
    assert "#009E73" in content

    # Responsive label helper.
    assert "function truncateBboxLabel" in content
    assert "ctx.measureText" in content
    assert "window.innerWidth < 640" in content

    # The draw site reads the species slot off each box.
    assert "function resolveBboxColour" in content
    assert "box.speciesColour" in content


def test_bbox_draw_uses_rendered_image_geometry_for_object_fit():
    """BBox overlays must align with the visible bitmap, not the img box."""
    gallery_js = _read("assets/js/gallery_utils.js")
    review_grid_js = _read("assets/js/review_grid.js")

    assert "function getWmRenderedImageGeometry" in gallery_js
    assert "getComputedStyle(img).objectFit" in gallery_js
    assert "contentX: (elementW - contentW) * position.x" in gallery_js
    assert "const geometry = getWmRenderedImageGeometry(img);" in gallery_js
    assert "geometry.contentX + box.x * scaleX" in gallery_js

    assert "window.getWmRenderedImageGeometry(img)" in review_grid_js
    assert "geometry.contentX + x * geometry.contentW" in review_grid_js


def test_review_grid_card_surfaces_start_end_window():
    """Grid card headers must make event boundaries scannable."""
    content = _read("templates/components/review_grid_card.html")

    assert "(event.window_time or '').split(' - ')" in content
    assert "review-grid__card-window" in content
    assert "review-grid__card-window-chip--start" in content
    assert "review-grid__card-window-chip--end" in content
    assert 'review-grid__card-window-label">Start' in content
    assert 'review-grid__card-window-label">End' in content

    # The raw range must not stay buried in the muted meta line.
    assert (
        '<span class="review-grid__card-time">{{ event.window_time }}</span>'
        not in content
    )


def test_review_grid_event_cards_have_stronger_boundaries():
    """Event cards need a visibly larger break and a bottom end cap."""
    content = _read("assets/design-system.css")

    stack_start = content.index(".review-grid__stack {")
    stack_end = content.index("}", stack_start)
    stack_block = content[stack_start:stack_end]

    assert "gap: clamp(16px, 2vw, 24px);" in stack_block
    assert ".review-grid__card::after" in content
    assert "height: 4px;" in content
    assert ".review-grid__card-window-chip" in content
    assert ".review-grid__card-window-chip--end" in content


def test_review_grid_js_can_invert_card_selection():
    """Invert selection stays scoped to the current event card."""
    content = _read("assets/js/review_grid.js")

    assert "function invertCardSelection(card)" in content
    assert "const checkboxes = tileCheckboxes(card);" in content
    assert "setTileSelectionState(cb, !cb.checked);" in content
    assert "syncCardSelectionUi(card);" in content
    assert "verb === 'invert_selection'" in content


def test_gallery_context_frames_can_draw_bbox_without_toolbox_toggle():
    content = _read("assets/js/gallery_utils.js")

    assert "host?.dataset.contextOnly === '1'" in content
    assert "const x = parseFloat(container.dataset.bboxX);" in content
    # The context-only branch must still call drawBoundingBoxes — but
    # the call now accepts either the legacy single-box list or a
    # multi-sibling list. Match the call by
    # function name + canvas+img args, not the literal `[{` opener.
    assert "drawBoundingBoxes(canvas, img, boxes, detectionId" in content


def test_gallery_detail_modal_auto_renders_companion_bboxes():
    """Detail-modal auto-render must paint every active sibling bbox
    when more than one is present, regardless of the saved
    bbox-overlay user pref."""
    content = _read("assets/js/gallery_utils.js")

    # Force-on multi-bird path triggered by container.dataset.siblings
    assert "container.dataset.siblings" in content
    assert "multiBird" in content
    # The auto-render context branch must read siblings JSON and map
    # them to a box list with `isCurrent` set on the entry-point id.
    assert "sib.detection_id === detectionId" in content


# Additional review-surface assertions for highlighting, receipt
# rendering, zoom state, copy cleanup, mini-map placement, and
# canvas-side species-colour handling.


def test_smart_zoom_toggle_pressed_state_and_state_aware_title():
    """Smart zoom syncs both segmented and legacy button presentations."""
    js = _read("assets/js/gallery_utils.js")

    # The shared state setter exists and reflects all three contract bits.
    assert "function applySmartZoomToggleState" in js
    assert "wm-view-mode-toggle" in js
    assert "[data-view-mode]" in js
    assert "is-active" in js
    assert "wm-toolbox__btn--toggled" in js
    assert "aria-pressed" in js
    assert "Show full image" in js
    assert "Zoom into bird" in js
    # Graceful degradation when the new viewer has no bbox.
    assert "No bounding box — zoom unavailable for this frame" in js

    # Both toggleSmartZoom and initSmartZoom call the state setter so
    # the pressed state is consistent on click + on initial load.
    assert js.count("applySmartZoomToggleState(") >= 3

    css = _read("assets/design-system.css")
    assert ".wm-toolbox__btn--toggled" in css


def test_review_grid_zoom_toggle_syncs_the_shared_scope():
    """Multi-frame Review events use one shared zoom intent per grid.

    Toggling smart zoom inside any cell must update every viewer in the
    workbench scope, not just the first `.wm-image-viewer` in DOM order.
    """
    js = _read("assets/js/gallery_utils.js")

    assert "function applySmartZoomPreferenceToScope" in js
    assert "scope.querySelectorAll('.wm-toolbox-host')" in js
    assert "scope.querySelectorAll('.bbox-toggle.active')" in js
    assert "localStorage.setItem(prefs.zoom, nextPref)" in js


def test_bbox_overlay_init_and_redraw_stay_on_the_local_viewer_host():
    """The shared Review grid persists bbox state per workbench, but the
    actual canvas redraw must stay on the clicked host, never the first
    viewer in the shared scope.
    """
    js = _read("assets/js/gallery_utils.js")

    assert "function resolveViewerHost(scope, el) {" in js
    assert "const host = resolveViewerHost(scope, img);" in js
    assert "const btn = resolveViewerToolButton(host, scope, '.bbox-toggle');" in js
    assert "const host = resolveViewerHost(scope, btn);" in js
    assert "const container = host?.querySelector('.modal-image-viewer');" in js
    assert "scope.querySelector('.modal-image-viewer')" not in js


def test_canvas_draw_forwards_species_colour_from_currentbbox():
    """Canvas redraw forwards species colour through each bbox draw call."""
    content = _read("assets/js/gallery_utils.js")

    # Both code paths (siblings + single bbox) carry speciesColour
    # through to the draw call.
    assert "currentBbox.speciesColour" in content
    assert "[data-species-colour]" in content
    assert "speciesColour: currentSlot" in content
    assert "speciesColour: sibSlot" in content
    # The host fallback is documented and explicit.
    assert "fallbackSlot" in content


# Dark-mode palette assertions for base tokens, species-colour slots,
# and review-surface overrides that need explicit rules.


def test_design_system_defines_dark_scope_with_wong_palette():
    """The dark scope redefines the core token family and Wong palette."""
    content = _read("assets/design-system.css")

    # The scope is keyed on the OS preference so no JS toggle
    # infrastructure is needed. If a manual theme toggle is added
    # later the same rules can live in a :root[data-theme=dark]
    # selector. Match the `{` to skip the comment mention that lives
    # inside the explainer block above the first scope.
    assert "@media (prefers-color-scheme: dark) {" in content

    # All 8 Wong slots must have a dark variant. The first `@media
    # ... {` opens the :root token scope.
    dark_scope_start = content.index("@media (prefers-color-scheme: dark) {")
    dark_scope = content[dark_scope_start : dark_scope_start + 6000]
    for slot in range(8):
        assert f"--species-colour-{slot}:" in dark_scope, (
            f"Dark scope missing --species-colour-{slot} override"
        )

    # Highlight rules ride on --color-primary*. The dark scope must
    # redefine the primary family so active accents follow the theme.
    for token in (
        "--color-primary:",
        "--color-primary-dark:",
        "--color-primary-light:",
    ):
        assert token in dark_scope, f"Dark scope missing {token}"

    # Core base tokens that carry text / surfaces / borders must also
    # flip so Review-surface rules that use them cascade correctly.
    for token in (
        "--color-bg:",
        "--color-surface:",
        "--color-surface-2:",
        "--color-text:",
        "--color-text-muted:",
        "--color-border:",
    ):
        assert token in dark_scope, f"Dark scope missing {token}"


def test_js_species_palette_reads_from_css_custom_properties():
    """The canvas-side species palette follows live CSS custom properties."""
    content = _read("assets/js/gallery_utils.js")

    # Fallback palette is still present (for SSR / detached scenarios)
    # but the live getter is the primary read path.
    assert "SPECIES_COLOURS_FALLBACK" in content
    assert "function getSpeciesColours" in content

    # Live read against :root custom properties.
    assert "getComputedStyle(document.documentElement)" in content
    assert "'--species-colour-' + idx" in content

    # Cache invalidation on OS theme change.
    assert "matchMedia('(prefers-color-scheme: dark)')" in content
    assert "_speciesColoursCache = null" in content

    # Backwards-compat: the old static SPECIES_COLOURS constant name is
    # still exposed (via a Proxy) so existing draw code + tests keep
    # working with SPECIES_COLOURS[slot] indexing.
    assert "const SPECIES_COLOURS = new Proxy" in content


# ─────────────────────────────────────────────────────────────────────
# Mixed-event direct actions
# ─────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────
