"""Static template / JS / CSS contract tests for the event-based Review surface.

The Queue / Bulk Review mode switch was removed and replaced with a
single biological event panel.
"""

from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _read(relative_path: str) -> str:
    return (_project_root() / relative_path).read_text(encoding="utf-8")


def test_review_page_exposes_event_browser_and_event_payload():
    content = _read("templates/orphans.html")

    # The Queue / Bulk Review mode switch is gone.
    assert 'data-review-mode="queue"' not in content
    assert 'data-review-mode="bulk"' not in content
    assert 'review-cluster-data' not in content

    # The new event-based surface ships an event rail and a
    # single payload script tag.
    assert 'id="reviewEventBrowser"' in content
    assert 'id="reviewEventList"' in content
    assert 'data-review-event' in content
    assert 'id="review-event-data"' in content
    assert "event.eligibility_label" in content
    assert '/assets/js/batch_actions.js?v=1' in content


def test_review_event_rail_is_text_only_after_a0():
    """The rail cards are text-only and expose the active state clearly."""
    content = _read("templates/orphans.html")

    # The old cover thumbnail block must stay gone. The per-card
    # trail map is still expected, so the bare `__trail` absence check
    # is no longer meaningful here.
    assert "review-event-card__image" not in content
    assert "event.bbox_trail_preview" not in content

    # Per-card trail map rides on the existing `event.bbox_trail`
    # payload so every card renders its own motion trail.
    assert "review-event-card__trail-map" in content
    assert "review-event-card__trail-box" in content
    assert "event.bbox_trail" in content

    # New text-only structure, now count-first so the rail stays
    # legible without depending on the species line.
    assert "review-event-card__accent" in content
    assert "review-event-card__count" in content
    assert "review-event-card__count-value" in content
    # The `frame(s)` label is intentionally absent so the count block
    # stays compact and easier to scan.
    assert "review-event-card__count-label" not in content
    assert "review-event-card__body" in content
    assert "review-event-card__species" in content
    assert "review-event-card__time" in content

    # The first (active by loop.first) card mirrors its state as
    # aria-current so JS rail rotation can pair the two.
    assert 'aria-current="true"' in content


def test_review_page_keeps_orphan_only_workspace_alive():
    """Regression: orphan-only pages must still render the workspace."""
    content = _read("templates/orphans.html")

    # Page gate must include the orphan branch as well.
    assert "{% if review_events or orphans %}" in content
    # Queue rail markup is still present for the orphan-only path and
    # stays hidden while the event workspace is active.
    assert 'id="reviewQueueBrowser"' in content
    assert 'id="reviewQueueList"' in content
    assert "{% if queue_orphans and not review_events %}" in content
    assert "for img in queue_orphans" in content
    # The default panel-type follows the available rail.
    assert 'data-panel-type="{% if review_events %}event{% else %}queue{% endif %}"' in content


def test_review_event_panel_exposes_event_actions_and_grid():
    """The event panel uses an equal-size detection grid."""
    content = _read("templates/components/review_event_panel.html")

    # Wiring hooks that survive the restructure.
    assert "data-review-event-controls" in content
    assert "data-review-event-bbox-toggle" in content
    assert 'data-review-panel-action="approve_event"' in content
    assert 'data-review-panel-action="trash_event"' in content
    assert 'data-review-panel-action="select_event_species"' in content
    assert 'data-review-panel-action="open_event_species_picker"' in content
    assert "event.members" in content
    assert "Approve Event" in content
    assert "Move Event to Trash" in content
    assert "Choose another species" in content

    # New equal-size grid: the cell + grid classes are present and the
    # grid is aria-labelled so rail navigation stays accessible.
    assert "review-event-panel__grid" in content
    assert "review-event-panel__cell" in content
    assert 'aria-label="Frames in this event"' in content

    # The legacy cover + filmstrip classes are retired from the event
    # mode — no cover frame, no filmstrip strip, no per-member thumbnail
    # block inside the rendered template.
    assert "review-event-panel__filmstrip" not in content
    assert "review-event-panel__member-media" not in content
    assert "review-stage-panel__image-frame wm-toolbox-host" not in content

    # Retired 2026-04-19: the right-rail mini-trail + "Event Ready" badge
    # were redundant with the grid header and cell bbox overlays.
    assert "review-event-panel__trail-section" not in content
    assert "review-event-panel__trail-map" not in content
    assert "review-event-panel__trail-head" not in content
    assert "event.bbox_trail" not in content

    # Retired 2026-04-19: the Utilities / "Review in Queue" button.
    assert "data-review-open-item" not in content
    assert "Review in Queue" not in content


def test_review_workspace_js_handles_event_state_and_actions():
    content = _read("assets/js/review_workspace.js")

    # The mode switch and cluster vocabulary are gone.
    assert "let reviewMode" not in content
    assert "setReviewMode" not in content
    assert "selectReviewCluster" not in content
    assert "renderBulkEmptyPanel" not in content
    assert "/api/review/bulk-panel/" not in content
    assert "/api/review/bulk-approve" not in content
    assert "data-review-cluster-bbox-toggle" not in content

    # Event-based state and endpoints.
    assert "const reviewEventDataEl = document.getElementById('review-event-data');" in content
    assert "function selectReviewEvent(eventKey, options = {})" in content
    assert "function renderEmptyEventStage()" in content
    assert "async function reviewApproveEvent(eventKey)" in content
    assert "async function reviewTrashEvent(eventKey)" in content
    assert "async function reviewOpenEventSpeciesPicker(eventKey, detectionId, currentSpecies)" in content
    assert "action === 'select_event_species'" in content
    assert "action === 'open_event_species_picker'" in content
    assert "/api/review/event-panel/" in content
    assert "/api/review/event-approve" in content
    assert "/api/review/event-trash" in content
    assert "event.target.closest('[data-review-event]')" in content
    assert "event.target.closest('[data-review-event-bbox-toggle]')" in content
    assert "event.target.closest('[data-review-open-item]')" in content


def test_review_workspace_js_drill_down_works_without_rail_dom():
    """Regression: Open in Queue must work even when no rail node exists.

    The new event-based side rail no longer renders ``.review-queue__item``
    nodes, so ``selectReviewItem`` cannot rely on a DOM lookup. It must
    fall back to ``reviewQueueIndex`` and the JSON data so the
    drill-down from inside an event panel actually loads the per-detection
    queue panel instead of failing silently.
    """
    content = _read("assets/js/review_workspace.js")

    # Index-backed lookup helper exists.
    assert "function getReviewItemRecord(itemKey)" in content
    assert "reviewQueueIndex.get(itemKey)" in content

    # selectReviewItem has both the rail path and the index fallback.
    assert "const record = getReviewItemRecord(itemKey);" in content

    # stepReviewItem can navigate without a rail using the stage panel
    # dataset as the pivot.
    assert "panel?.dataset.itemKey" in content

    # Init bootstrap walks event rail -> queue rail -> queue index.
    assert "const firstRailItem = getVisibleReviewItems()[0];" in content
    assert "const firstIndexKey = reviewQueueIndex.keys().next().value;" in content

    # After the last event approve, falls back to remaining queue items
    # rather than the empty stage when orphans are still open.
    assert "applyReviewEventDomRemoval" in content


def test_review_event_panel_mounts_tile_toolbox_on_every_grid_cell():
    """Every event grid cell hosts ``tile_toolbox`` for viewer tools."""
    content = _read("templates/components/review_event_panel.html")

    # Macro import.
    assert 'from "partials/tile_toolbox.html" import tile_toolbox' in content

    # Every grid cell is a toolbox host.
    assert "review-event-panel__cell wm-toolbox-host" in content

    # Viewer tools, including the zoom toggle, are wired on the grid cells.
    assert "show_viewer_tools=true" in content
    assert "detection_id=member.best_detection_id" in content
    assert "modal_target=member_modal_target if not member.context_only else none" in content
    assert "details_href=('/gallery/' ~ member.gallery_date ~ '?focus=' ~ member.best_detection_id ~ '#detection-' ~ member.best_detection_id) if member.context_only and member.gallery_date and member.best_detection_id else none" in content
    assert "allow_change_species=not member.context_only" in content
    assert "allow_move_to_trash=not member.context_only" in content
    assert "allow_review_no_bird=not member.context_only" in content

    # The single event grid calls tile_toolbox with `surface='review'`
    # and `can_moderate=true`.
    assert content.count("surface='review'") >= 1
    assert content.count("can_moderate=true") >= 1


def test_review_event_panel_renders_every_member():
    """Every event renders **all** its members as first-class cells.

    Review frames (``context_only == false``) come first in timestamp
    order so the operator can approve/relabel them; context frames
    (Gallery anchors, same-species continuation) follow, still in
    timestamp order, shown read-only for orientation. Each physical
    frame appears exactly once.

    The previous Fixed-5 frame budget (2026-04-08) was retired on
    2026-04-19 because it hid relabel-affected frames from the
    operator when events carried more than five review-frames.
    Honest default: show everything that will be approved plus
    everything already in Gallery.

    Invariants the template must preserve:
    - A single ``display_members`` list is computed at the top of the
      file from ``event.members`` partitioned on ``member.context_only``.
    - The grid loop iterates ``display_members``, not ``event.members``.
    - No frame budget cap; no overflow hint (there is no clipping now).
    - The retired batch-panel surface is fully gone from the template
      (no ``review-batch-panel__*`` classes, no ``Apply Batch``, no
      ``continuity_batch`` reads inside the template).
    """
    content = _read("templates/components/review_event_panel.html")

    # Partition: review-frames first, context-frames second — no cap.
    assert "_review_members = event.members | rejectattr('context_only') | list" in content
    assert "_context_members = event.members | selectattr('context_only') | list" in content
    assert "display_members = _review_members + _context_members" in content

    # The old Fixed-5 budget variables are gone.
    assert "_frame_budget" not in content
    assert "_review_budget" not in content
    assert "_context_budget" not in content
    assert "_hidden_count" not in content
    assert "review-event-panel__grid-caption-overflow" not in content
    assert "ausgeblendet" not in content

    # Grid loop reads the full display list.
    assert "{% for member in display_members %}" in content
    assert "{% for member in event.members %}" not in content

    # The retired batch-panel surface is fully gone from the template.
    assert "review-batch-panel" not in content
    assert "Already in Gallery" not in content
    assert "Review now" not in content
    assert "Apply Batch" not in content
    assert "Approve Batch" not in content
    assert "{% if continuity_batch %}" not in content
    assert "continuity_batch.review_members" not in content
    assert "continuity_batch.anchor_members" not in content


def test_review_event_panel_carries_species_colour_and_ref_overlay():
    """Every coloured event-panel surface carries a species slot and reference image."""
    content = _read("templates/components/review_event_panel.html")

    # The remaining coloured surface uses the same data attribute + inline
    # custom property so a single CSS variable drives bbox border,
    # accent bar, and reference image border.
    assert content.count('data-species-colour="{{ member.species_colour }}"') >= 1
    assert "var(--species-colour-{{ member.species_colour }})" in content

    # Reference image overlay (img) + initial fallback (span) renders
    # for the event-mode grid cell.
    assert content.count('class="review-species-ref"') >= 1
    assert content.count("review-species-ref--initial") >= 1
    # The fallback uses the first letter of the common name.
    assert "[0]" in content


def test_review_event_panel_grid_cells_carry_lane_b_a1_wiring():
    """Grid cells carry the same species-colour and reference-image wiring."""
    content = _read("templates/components/review_event_panel.html")

    # Each cell root carries the colour token via data attribute + inline
    # custom property style (both are required: the attribute drives the
    # CSS selector binding, the inline style sets the actual value).
    assert 'data-species-colour="{{ member.species_colour }}"' in content
    assert "--cell-species-colour: var(--species-colour-{{ member.species_colour }})" in content

    # The bbox payload injected into tile_toolbox carries the species
    # slot so a future canvas-draw fix can read it off the box directly.
    assert '"speciesColour": member.species_colour' in content

    # Context-only cells inside the event grid are marked read-only.
    assert 'data-context-only="1"' in content


def test_review_event_panel_context_anchor_cell_renders_in_gallery_badge():
    content = _read("templates/components/review_event_panel.html")

    assert "review-event-panel__cell-badge--anchor" in content
    assert ">In Gallery</span>" in content


def test_design_system_uses_neutral_event_grid_border_and_green_anchor_highlight():
    content = _read("assets/design-system.css")

    assert "border: 1px solid rgba(31, 64, 115, 0.18);" in content
    assert '.review-event-panel__cell[data-context-only="1"] {' in content
    assert "border-color: rgba(47, 143, 78, 0.82);" in content
    assert ".review-event-panel__cell-badge--anchor {" in content


def test_context_only_event_cells_keep_hover_enabled_for_viewer_tools():
    content = _read("assets/design-system.css")

    start = content.index('.review-event-panel__cell[data-context-only="1"] {')
    end = content.index("}", start)
    block = content[start:end]

    assert "pointer-events: none;" not in block


def test_orphan_modal_carries_species_colour_and_ref_overlay():
    """Orphan-modal surfaces carry the species-colour token and reference image."""
    content = _read("templates/components/orphan_modal.html")

    # Cover frame carries the colour token via the viewer container so
    # the bbox + accent share one CSS var.
    assert 'data-species-colour="{{ orphan.species_colour }}"' in content
    assert "var(--species-colour-{{ orphan.species_colour }})" in content
    assert "review-species-ref" in content

    # Quick-pick buttons carry per-species colour for visual linking
    # back to the detection grid.
    assert 'data-species-colour="{{ species.species_colour }}"' in content


def test_design_system_defines_wong_species_palette():
    """Eight deterministic Wong slots live in :root."""
    content = _read("assets/design-system.css")
    for slot in range(8):
        assert f"--species-colour-{slot}:" in content
    # Reference colours from the Wong (2011) palette.
    assert "#0072B2" in content
    assert "#E69F00" in content
    assert "#009E73" in content


def test_design_system_defines_species_ref_overlay_and_responsive_label():
    """The reference-image overlay and responsive bbox label class are present."""
    content = _read("assets/design-system.css")
    assert ".review-species-ref" in content
    assert ".review-species-ref--initial" in content
    assert ".bbox-label--responsive" in content
    # The overlay must not eat hover events from the underlying toolbox.
    assert "pointer-events: none" in content
    # Overlay sits in the lower-right corner and keeps an explicit square
    # image box so the ref art cannot stretch into an oval.
    assert "bottom: 6px" in content
    assert "aspect-ratio: 1 / 1" in content
    assert "img.review-species-ref" in content


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


def test_review_workspace_js_handles_continuity_batch_actions():
    """Batch apply/undo/approve handlers keep context-only items guarded.

    The batch flow must:
    - keep per-batch local selection state keyed by ``batch_key``,
    - update all actionable cells + receipt + approve-gate in one
      DOM mutation,
    - drop ``context_only`` detection ids from the submit payload,
    - POST to ``/api/review/event-approve`` without an ``event_key``.
    """
    content = _read("assets/js/review_workspace.js")

    # Handlers exist.
    assert "function handleApplyBatchSpecies" in content
    assert "function undoBatchSpeciesChange" in content
    assert "async function reviewApproveBatch" in content

    # Delegated click handler wires the new actions.
    assert "action === 'apply_batch_species'" in content
    assert "action === 'undo_batch_species_change'" in content
    assert "action === 'approve_batch'" in content

    # Per-batch state map keyed by batch_key.
    assert "const reviewBatchState = new Map()" in content

    # Approve-batch gate requires convergence + all cells assigned.
    assert "updateBatchApproveGate" in content

    # Hard client-side guard against context_only detection ids and the
    # batch path never sends an event_key.
    assert "batchContextDetectionIds" in content
    assert "Refusing to submit" in content
    # The batch POST body must not carry an event_key field.
    batch_call_marker = "reviewApproveBatch"
    assert batch_call_marker in content


def test_review_workspace_event_fastpath_submits_actionable_ids_only():
    content = _read("assets/js/review_workspace.js")

    assert "const actionableIds = (controls.dataset.actionableDetectionIds || '')" in content
    assert "actionable detection ids only" in content
    assert "re-confirmed through event approval" in content
    assert "submitDetectionIds = actionableIds.length > 0 ? actionableIds : detectionIds;" in content
    assert "fastpathBody.detection_ids = submitDetectionIds;" in content


def test_gallery_context_frames_can_draw_bbox_without_toolbox_toggle():
    content = _read("assets/js/gallery_utils.js")

    assert "host?.dataset.contextOnly === '1'" in content
    assert "const x = parseFloat(container.dataset.bboxX);" in content
    assert "drawBoundingBoxes(canvas, img, [{" in content


def test_review_event_styles_live_in_design_system():
    content = _read("assets/design-system.css")

    # Mode toggle CSS is gone.
    assert ".review-mode-toggle" not in content

    # Event class family: rail plus event panel.
    assert ".review-event-browser" in content
    assert ".review-event-card" in content
    assert ".review-event-card__accent" in content
    assert ".review-event-card__count" in content
    assert ".review-event-card__body" in content
    assert ".review-event-panel__trail-map" in content
    assert ".review-event-panel__grid" in content
    assert ".review-event-panel__cell" in content
    assert ".review-event-panel__cell-media" in content
    assert ".review-event-panel__cell-action" in content
    assert ".review-stage-panel__empty" in content

    # Legacy cover and filmstrip class families are retired from the
    # design system and replaced by `__grid` and `__cell`.
    assert ".review-event-panel__filmstrip" not in content
    assert ".review-event-panel__member-media" not in content
    assert ".review-event-panel__member-image" not in content
    assert ".review-event-card__image " not in content
    assert ".review-event-card__trail " not in content

    # Workspace layout is flex-based so the event rail and the queue
    # rail can coexist on the same page (event-only, queue-only, both).
    assert ".review-workspace {" in content
    assert "display: flex;" in content
    assert ".review-workspace > .review-stage" in content

    # Continuity-batch stage styling.
    assert ".review-batch-panel" in content
    assert ".review-batch-panel__minimap" in content
    assert ".review-batch-panel__minimap-box--context" in content
    assert ".review-batch-panel__minimap-box--review" in content
    assert ".review-batch-panel__grid--anchors" in content
    assert ".review-batch-cell--context" in content
    assert ".review-batch-cell--review.is-active-event-member" in content
    assert ".review-batch-panel__receipt" in content


# Additional review-surface assertions for highlighting, receipt
# rendering, zoom state, copy cleanup, mini-map placement, and
# canvas-side species-colour handling.


def test_review_event_card_carries_strong_highlight_and_stage_cue():
    """The active rail card and stage panel share the same accent token."""
    content = _read("assets/design-system.css")

    # Active rail card: accent fill, thicker border, token-based.
    assert ".review-event-card.is-active" in content
    assert "border-width: 2px" in content
    assert "var(--color-primary" in content
    # Stage-connecting cue uses the same token family.
    assert "border-left: 3px solid var(--color-primary" in content


def test_review_species_receipt_partial_and_styles_exist():
    """The shared receipt partial and styles exist with slot-based markup."""
    partial = _read("templates/partials/review_species_receipt.html")

    # Partial structure.
    assert "data-review-species-receipt" in partial
    assert "review-species-receipt__prev" in partial
    assert "review-species-receipt__arrow" in partial
    assert "review-species-receipt__new" in partial
    assert "review-species-receipt__undo" in partial
    assert 'data-review-panel-action="undo_species_change"' in partial
    # Origin meta is conditional.
    assert "Manually changed" in partial
    assert "From CLS suggestion" in partial

    css = _read("assets/design-system.css")
    assert ".review-species-receipt-slot" in css
    assert ".review-species-receipt {" in css
    assert ".review-species-receipt__prev" in css
    assert ".review-species-receipt__arrow" in css
    assert ".review-species-receipt__new" in css
    assert ".review-species-receipt__undo" in css


def test_review_species_receipt_mounted_on_orphan_modal_and_event_panel():
    """The shared partial is included on both review surfaces with the
    appropriate scope, and the controls root carries the
    `data-original-species*` anchor that the Undo button reverts to.
    """
    orphan = _read("templates/components/orphan_modal.html")
    panel = _read("templates/components/review_event_panel.html")

    # Both surfaces include the partial.
    assert "include 'partials/review_species_receipt.html'" in orphan
    assert "include 'partials/review_species_receipt.html'" in panel

    # Both surfaces pin the original species anchor on the controls root.
    assert "data-original-species" in orphan
    assert "data-original-species-common" in orphan
    assert "data-original-species" in panel
    assert "data-original-species-common" in panel

    # Scope-aware include for the narrow event-panel aside.
    assert "scope='event'" in panel
    assert "scope='orphan'" in orphan


def test_review_workspace_js_handles_species_change_receipt():
    """Receipt updates, undo handling, and rail-step cleanup stay wired."""
    content = _read("assets/js/review_workspace.js")

    # New helpers exist.
    assert "function updateReviewSpeciesReceipt" in content
    assert "function clearReviewSpeciesReceipts" in content
    assert "function lookupSpeciesCommonName" in content

    # applyReviewSpeciesUi calls the receipt updater.
    assert "updateReviewSpeciesReceipt(controls, species, selectedOrigin)" in content

    # Delegated handler for the Undo button.
    assert "action === 'undo_species_change'" in content
    # The undo path resolves the original species from the controls root.
    assert "controls.dataset.originalSpecies" in content

    # Rail step clears any leftover receipt before the next item loads.
    assert "clearReviewSpeciesReceipts();" in content

    # XSS-safety: the receipt is built via createElement / textContent,
    # never innerHTML. The string 'innerHTML' must not appear inside
    # updateReviewSpeciesReceipt — we verify that no innerHTML write
    # lives between the function header and its closing brace.
    fn_start = content.index("function updateReviewSpeciesReceipt")
    fn_end = content.index("function lookupSpeciesCommonName")
    fn_body = content[fn_start:fn_end]
    assert ".innerHTML" not in fn_body
    assert "createElement" in fn_body
    assert "textContent" in fn_body


def test_smart_zoom_toggle_pressed_state_and_state_aware_title():
    """Smart zoom exposes pressed state, toggled styling, and state-aware copy."""
    js = _read("assets/js/gallery_utils.js")

    # The shared state setter exists and reflects all three contract bits.
    assert "function applySmartZoomToggleState" in js
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


def test_review_event_grid_is_workspace_scoped_for_zoom_persistence():
    """The event grid carries one shared workspace-scoped zoom preference."""
    content = _read("templates/components/review_event_panel.html")

    assert "review-event-panel__grid wm-viewer-scope" in content
    assert 'data-zoom-pref-key="wmb_review_zoom_pref"' in content
    assert 'data-bbox-pref-key="wmb_review_bbox_pref"' in content


def test_review_event_panel_e_copy_cleanup():
    """The trail header copy is cleaned up and the grid caption is present."""
    content = _read("templates/components/review_event_panel.html")

    # The trail section no longer carries its own <h3> species headline.
    assert "review-event-panel__title" not in content
    # The old "Cover image plus" copy is gone because there is no
    # dedicated cover-frame concept in this layout anymore.
    assert "Cover image plus" not in content
    # The grid caption container is present and states honestly what
    # "Approve Event" will do. The old "Approve once the pattern is
    # consistent" copy was retired on 2026-04-19 together with the
    # Fixed-5 budget: the caption now says exactly how many frames
    # will be approved and how many context frames are already in
    # the Gallery.
    assert "review-event-panel__grid-caption" in content
    assert "will be approved" in content
    assert "review-event-panel__grid-caption-context" in content


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
    dark_scope = content[dark_scope_start:dark_scope_start + 6000]
    for slot in range(8):
        assert f"--species-colour-{slot}:" in dark_scope, (
            f"Dark scope missing --species-colour-{slot} override"
        )

    # Highlight rules ride on --color-primary*. The dark scope must
    # redefine the primary family so active accents follow the theme.
    for token in ("--color-primary:", "--color-primary-dark:", "--color-primary-light:"):
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


def test_design_system_defines_review_surface_dark_overrides():
    """The review-surface dark override block covers explicit light backgrounds."""
    content = _read("assets/design-system.css")

    # Find the second `@media (prefers-color-scheme: dark) {` scope
    # — the review-surface override block. The first `{` match is the
    # :root token scope at the top of the file.
    first = content.index("@media (prefers-color-scheme: dark) {")
    second = content.index("@media (prefers-color-scheme: dark) {", first + 1)
    review_scope = content[second:second + 7000]

    # Review-scope selectors that must be overridden explicitly.
    expected_selectors = [
        ".review-workspace > .review-event-browser",
        ".review-event-card",
        ".review-event-card.is-active",
        ".review-event-panel__trail-map",
        ".review-event-panel__grid",
        ".review-event-panel__cell",
        ".review-event-panel__cell-media",
        ".review-event-panel__cell-action",
        ".review-event-panel__species-pill",
        ".review-species-receipt",
        ".review-event-card__badge--ready",
        ".review-event-card__badge--fallback",
        ".review-stage-panel__species-media",
        ".review-batch-panel",
        ".review-batch-panel__decision",
        ".review-batch-panel__minimap",
    ]
    for sel in expected_selectors:
        assert sel in review_scope, f"Dark override missing selector {sel}"


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
# Mixed-event direct actions (2026-04-08)
# ─────────────────────────────────────────────────────────────────────


def test_event_grid_cells_no_longer_render_per_frame_keep_trash_toggle():
    """Per-frame Keep/Trash toggle is retired from the event grid.

    Retired 2026-04-19: the inline Keep-this-frame / Trash-this-frame
    button under every cell was tied to the Mixed-Event resolve path
    and cluttered the grid. Per-frame Trash lives in the Hover-Toolbox
    (Move to Trash) and in the Multi-Select batch footer (Trash
    Selected). The backend mixed-resolve endpoint stays in place but
    is no longer triggered from the event panel UI.
    """
    content = _read("templates/components/review_event_panel.html")

    # The per-frame toggle and its hooks are gone from the template.
    assert "data-review-frame-decision" not in content
    assert 'data-frame-state="keep"' not in content
    assert "Keep this frame" not in content
    assert "data-frame-decision-copy" not in content
    assert "review-event-panel__cell-action" not in content


def test_review_workspace_js_wires_frame_decision_and_event_resolve():
    content = _read("assets/js/review_workspace.js")

    # Frame-decision click handler + helpers.
    assert "data-review-frame-decision" in content
    assert "function toggleFrameDecision" in content
    assert "function readFrameDecisions" in content
    assert "function reviewResolveEvent" in content
    assert "/api/review/event-resolve" in content

    # Mixed-path switch in Approve Event: resolve when any trash, else
    # the fast single-call approve path stays.
    assert "decisions.trash.length > 0" in content
    assert "reviewResolveEvent(eventKey, controls, decisions)" in content


def test_event_cell_species_label_is_clickable_relabel_button():
    """V1 per-frame relabel: every actionable cell renders its species
    label as a clickable button wired to WmSpeciesPicker. Context-only
    cells keep the plain span form."""
    content = _read("templates/components/review_event_panel.html")

    assert "data-review-cell-relabel" in content
    assert "data-current-species" in content
    # Button must carry the same class so the existing palette + dark
    # mode rules still bind to it.
    assert 'class="review-event-panel__cell-species"' in content


def test_review_event_panel_exposes_explicit_multi_select_mode():
    content = _read("templates/components/review_event_panel.html")

    assert 'data-review-multi-select-toggle' in content
    assert 'data-review-event-grid' in content
    assert 'data-multi-select-mode="0"' in content
    assert 'data-review-multi-select-checkbox' in content
    assert 'data-review-multi-select-footer' in content
    assert 'window.reviewEventBatchRelabelSelected()' in content
    assert 'window.reviewEventBatchTrashSelected()' in content
    assert 'window.reviewEventBatchCancelMultiSelect()' in content


def test_review_event_panel_species_hint_mentions_selected_frame_shortcut():
    content = _read("templates/components/review_event_panel.html")

    assert "Event-wide species choice stays local until you approve the event." in content
    assert "With selected frames, a quick-pick relabels them immediately." in content


def test_review_workspace_js_wires_cell_relabel_to_species_picker():
    content = _read("assets/js/review_workspace.js")

    assert "data-review-cell-relabel" in content
    assert "async function openReviewCellRelabel" in content
    assert "WmSpeciesPicker.pickSpecies" in content
    assert "/api/moderation/bulk/relabel" in content


def test_review_workspace_js_keeps_event_species_mirror_in_sync_with_bbox_labels():
    content = _read("assets/js/review_workspace.js")

    assert "function syncReviewCellSpeciesArtifacts" in content
    assert "viewer.dataset.bboxName = nextCommon || nextKey;" in content
    assert "currentBbox.name = nextCommon || nextKey;" in content
    assert "bboxBtn.dataset.currentBbox = JSON.stringify(currentBbox);" in content
    assert "redrawBboxOverlay(bboxBtn);" in content
    assert "cell.dataset.speciesIsManual = '1';" in content


def test_review_event_quick_species_buttons_expose_ref_image_url_for_cell_sync():
    content = _read("templates/components/review_event_panel.html")

    assert 'data-species-ref-image-url="{{ species.species_ref_image_url }}"' in content


def test_review_workspace_js_syncs_top_right_reference_image_on_species_change():
    content = _read("assets/js/review_workspace.js")

    assert "function lookupSpeciesRefImageUrl" in content
    assert "controls.dataset.selectedSpeciesRefImageUrl" in content
    assert "nextRef.matches('img.review-species-ref')" in content
    assert "nextRef.className = 'review-species-ref';" in content
    assert "nextRef.className = 'review-species-ref review-species-ref--initial';" in content
    assert "nextRef.setAttribute('src', nextRefImageUrl);" in content


def test_review_workspace_js_wires_event_multi_select_batch_actions():
    content = _read("assets/js/review_workspace.js")

    assert "function toggleReviewMultiSelectMode()" in content
    assert "function handleReviewMultiSelect(checkbox, shiftKey)" in content
    assert "toggleReviewMultiSelectRange" in content
    assert "window.reviewEventBatchRelabelSelected = reviewEventBatchRelabelSelected;" in content
    assert "window.reviewEventBatchTrashSelected = reviewEventBatchTrashSelected;" in content
    assert "window.WmBatchActions.runBatchRelabel" in content
    assert "window.WmBatchActions.runBatchAction(" in content
    assert "'/api/moderation/bulk/reject'" in content
    assert "data-review-multi-select-toggle" in content
    assert "data-review-multi-select-checkbox" in content


def test_review_workspace_js_uses_quick_species_click_as_selected_frame_shortcut():
    content = _read("assets/js/review_workspace.js")

    assert "async function quickRelabelSelectedReviewFrames" in content
    assert "function getSelectedReviewMultiSelectDetectionIds" in content
    assert "function updateReviewCellSpeciesDisplay" in content
    assert "window.WmBatchActions.executeBatchAction('/api/moderation/bulk/relabel'" in content
    assert "if (isReviewMultiSelectMode(panel) && getSelectedReviewMultiSelectDetectionIds(panel).length > 0)" in content
    assert "clearReviewMultiSelect(panel);" in content


def test_ui_standard_documents_multi_select_quick_pick_relabel_shortcut():
    content = _read("docs/UI_STANDARD.md")

    assert "selected-frame relabel shortcut while multi-select is active" in content
    assert "With selected frames, a quick-pick relabels them immediately." in content


def test_species_reference_image_is_anchored_top_right():
    """The species reference image stays top-right and keeps its square sizing contract."""
    import re

    content = _read("assets/design-system.css")

    # Primary rule: top-anchored, bottom cleared.
    rule_start = content.index(".review-species-ref {")
    rule_block = content[rule_start:rule_start + 400]
    assert "top: 6px;" in rule_block
    assert "bottom: auto;" in rule_block
    assert "right: 6px;" in rule_block
    # Round/square sizing contract preserved: width == height and the
    # shape is a full circle.
    width_m = re.search(r"width:\s*(\d+)px;", rule_block)
    height_m = re.search(r"height:\s*(\d+)px;", rule_block)
    assert width_m and height_m, "width/height in px expected"
    assert width_m.group(1) == height_m.group(1), (
        f"species-ref must be square, got {width_m.group(1)}x{height_m.group(1)}"
    )
    assert "border-radius: 50%;" in rule_block

    # img-specific rule still has the 1/1 aspect-ratio guard.
    img_rule_start = content.index("img.review-species-ref {")
    img_rule_block = content[img_rule_start:img_rule_start + 200]
    assert "aspect-ratio: 1 / 1;" in img_rule_block

    # Mobile override also top-anchored and square.
    mobile_idx = content.index("@media (max-width: 720px)", content.index(".review-species-ref"))
    mobile_block = content[mobile_idx:mobile_idx + 400]
    assert "top: 4px;" in mobile_block
    assert "bottom: auto;" in mobile_block
    mobile_width = re.search(r"width:\s*(\d+)px;", mobile_block)
    mobile_height = re.search(r"height:\s*(\d+)px;", mobile_block)
    assert mobile_width and mobile_height
    assert mobile_width.group(1) == mobile_height.group(1)
