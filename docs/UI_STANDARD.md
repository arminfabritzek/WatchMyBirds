# WatchMyBirds UI Standard

This file defines the frontend conventions and the preferred
shared DOM structures for:
- Modals (including types/variants)
- Review-stage panels
- Thumbnails / Tiles (including types/variants)
- Action bars
- Image viewers

This is migration-aware guidance for the current app state. It is not a claim
that every legacy surface has already been converted.

## 0. Unified Image UX (binding, app-wide)

Every place in the app that shows a bird image — Stream, Gallery, Subgallery,
Species, Species Overview, Review Desk (event panel + queue panel), Trash,
Orphans, Inbox, Restore, detail modals, and any future surface — must present
the **same image UX**. This is a hard rule, not a guideline. It exists so that
shared concerns (rating stars, favorite, change-species, view-details, hover
tooltips, keyboard navigation, bbox overlay behavior, zoom modal) can be added
or changed in **one place** and instantly apply everywhere.

**The contract:**

1. **One viewer component.** Every image is rendered through
   `templates/components/modal_image_viewer.html` (`render_image_viewer`)
   inside the canonical `wm-image-viewer` / `wm-modal__image` /
   `review-stage-panel__image-frame` shells. No surface builds its own
   `<img>` + bbox overlay stack.
2. **One toolbox.** Every image-bearing surface hosts the shared
   `tile_toolbox` macro (§5) inside a `wm-toolbox-host` container. The toolbox
   is the only place per-image actions live (Favorite, View Details,
   Change Species, Move to Trash, Restore, Deep Scan, Mark No Bird, future
   star ratings, …). Surfaces may *omit* an action when the route does not
   support it; they may **not** rename it, reorder it, or add a parallel
   surface-local button for the same action.
3. **One action vocabulary.** The canonical verbs in §"Detection Action Frame
   Contract" are the only allowed labels. Adding a new image action means
   adding it to the toolbox macro and to this section, never as a one-off in a
   single template.
4. **One zoom path.** Clicking any image opens the same `wm-modal` viewer
   (detection-backed → `detection_modal.html`, image-only fallback →
   `orphan_modal.html`). No surface ships its own lightbox.
5. **Action position is fixed.** The toolbox sits at the bottom of the image
   on hover/focus (`wm-toolbox`) or as a `wm-toolbox--bar` strip in modal
   footers. Surfaces do not relocate it to the side, top, or outside the
   image frame.
6. **Per-image state surfaces are uniform.** Things like favorite stars,
   future rating stars, eligibility badges, and selection checkboxes use
   shared classes (`wm-tile__badge`, `wm-toolbox__fav`, …) and the same DOM
   position across surfaces.
7. **No new image-bearing surface ships without reusing this contract.**
   PRs that introduce a new image view must reuse `render_image_viewer` +
   `tile_toolbox`. If a real exception is needed, this section must be
   updated in the same change.

**Why this matters:** the explicit goal is that adding a new per-image
affordance — for example star ratings — becomes a single edit to the toolbox
macro plus the design-system CSS, and lights up Stream, Gallery, Review,
Trash, modals, and everything else at once. Any drift defeats that goal.

**Audit:** check this file against the codebase after touching
image-bearing surfaces.

## 0a. Hover Tooltip Convention (binding)

Every interactive control on an image-bearing surface — toolbox buttons,
review decision buttons, bbox toggles, navigation arrows, modal action-bar
buttons — must carry a **short, informative hover tooltip**. Tooltips are
discoverability, not noise.

**Rules:**

1. **Mechanism:** native `title="…"` attribute. No custom JS popovers, no
   auto-opening tooltips, no tooltip libraries. The browser's built-in delay
   keeps them unobtrusive.
2. **Content:** one short phrase (≤ ~60 chars) describing **what the button
   does**, in the same imperative voice as the action vocabulary
   (`Approve this event`, `Mark bounding box as wrong`, `Open in per-frame
   queue`). Not a restatement of the visible label.
3. **State-aware:** when a button is disabled, the tooltip explains *why*
   (`Login required`, `Event not yet eligible — needs ≥ 3 frames`).
4. **Accessibility parity:** every button also carries an `aria-label`
   (already required by the action frame contract). The `title` and
   `aria-label` may differ — `aria-label` is the canonical name of the
   action; `title` is the helpful one-liner.
5. **No emoji-only buttons without a tooltip.** If the visible glyph is
   ✓, ⋮, ↗, ‹, ›, ⌕, ☆, etc., a `title` is mandatory.
6. **One source per action.** When the same action appears in the toolbox
   macro and in a review decision rail, both surfaces use the same tooltip
   wording. Tooltip copy lives next to the action definition, not in
   per-template forks.

Variants follow the BEM modifier `--` and are always set in addition to the base
class (for example `wm-modal wm-modal--form`, `wm-tile wm-tile--review`).

## Current Conventions

- `assets/design-system.css` is the authoritative source for shared UI
  primitives such as buttons, badges, review-stage controls, tiles, and modal
  subcomponents.
- Detection-bearing surfaces currently belong to one of two shells:
  `public shell` for Stream, Gallery, Species, and Species Overview, and
  `workbench shell` for Review and Trash. Layout density may differ, but shared
  detection components must not fork semantics or wording between shells.
- New shared buttons must use `btn` plus the design-system modifiers such as
  `btn--primary`, `btn--secondary`, `btn--danger`, `btn--accent`, `btn--success`,
  `btn--info`, `btn--outline-primary`, `btn--outline-danger`, `btn--sm`,
  `btn--lg`, and `btn--block`.
- Legacy Bootstrap-era button classes still exist in older surfaces such as
  `settings`, `edit`, `login`, and `partials/taskbar`. They are tolerated only
  as migration debt. Do not introduce new `btn btn-primary`,
  `btn btn-outline-*`, `btn-light`, or similar legacy-only patterns.
- Detection modal composition should continue to flow through
  `templates/components/detection_modal.html`, which already composes
  `modal_image_viewer.html`, `modal_detection_info.html`, and
  `modal_action_bar.html`.
- Review-stage composition should continue to flow through
  `templates/components/review_stage_panel.html` and
  `templates/components/orphan_modal.html`.
- Review-stage panels should read as one operator workbench:
  queue rail on the left, image viewer in the center, inspector/action rail on
  the right. Keep utility copy short and prefer compact section labels such as
  `Actions`, `BBox`, `Species`, and `Approve`.
- The inline Review viewer should sit inside one stable stage frame with a
  consistent aspect ratio and `contain` behavior so portrait vs landscape images
  do not reflow the workbench or misalign the control strip.
- Review bbox overlays must stay bound to the real rendered image frame inside
  that stage, not to the outer stage container, so inline bbox geometry matches
  the modal viewer.
- The Review facts row, viewer stage, and under-image control strip should share
  the same content width so the workbench reads as one aligned column rather
  than separate floating blocks.
- Review previous/next controls should read as compact centered stage buttons,
  not as stretched side rails that change the perceived height of the viewer.
- Review metrics/facts should not live as a permanent full-width badge row above
  the stage. When shown, prefer a compact toggle-revealed metadata panel inside
  the Review viewer shell so the image remains primary.
- The Review workbench viewer may stay inline for fast triage, but clicking the
  image should open the same larger `wm-modal` viewer style used elsewhere in
  the app when closer inspection is needed.
- When practical, the prominent inline Review image should reuse the same
  shared image-viewer composition (`render_image_viewer` plus toolbox/zoom
  affordances) as the detail modal instead of maintaining a separate Review-only
  image engine.
- Detection-backed Review zoom must reuse
  `templates/components/detection_modal.html` instead of forking a review-only
  modal shell. Only true no-detection review items may fall back to a simpler
  image-backed modal.
- Trash should follow the same `workbench shell` logic:
  summary bar above, left-side ops rail for batch/range/import controls, and
  the item surface on the right.
- Review and modal status text mapping must come from one shared helper or
  macro, not from duplicated inline label logic.
- If markup or decision logic repeats in 2 or more surfaces, extract a shared
  partial, macro, or Python helper instead of duplicating it again.
- `assets/js/gallery_utils.js` is still an oversized compatibility module.
  New unrelated behavior should go into a dedicated JS file rather than growing
  that file further.

## Detection Action Frame Contract

The shared detection action frame is the canonical control surface for
detection-bearing tiles, filmstrips, and modal/detail surfaces.

- Canonical action vocabulary:
  `View Details`, `Favorite`, `Change Species`, `Move to Trash`, `Restore`,
  `Correct`, `Wrong`, `Approve`, `Deep Scan`, `Mark No Bird`
- Surfaces may omit actions only when the subject identity or route does not
  support them. They must not rename the same underlying action on another
  surface.
- New detection controls must use delegated `data-action` handlers. Do not add
  new inline `onclick` handlers for detection actions.
- Public surfaces must render the same frame for guests and authenticated users.
  Protected actions must stay visible in a disabled/login-required state rather
  than disappearing.
- Workbench surfaces may stay authenticated-only, but when they reuse the frame
  they should keep the same wording and ordering.
- Review-side utility panels may keep short section headings, but action labels
  themselves should stay canonical, for example `Change Species`,
  `BBox Confirm`, and `BBox Reject`.
- Review quick-species strips may stay on the local select/confirm path, but
  the Review species section must still expose one explicit route into the full
  species picker such as `Choose another species`.
- Review quick-species state should be legible at a glance: the default
  suggestion and the current selection should use distinct visual markers
  instead of relying on helper copy alone.
- Viewer/navigation controls such as zoom, close, next/previous, and download
  are not canonical detection actions. They may sit next to the frame, but they
  must not replace it.
- In detail modals, object actions such as `Favorite` and `Change Species`
  should prefer the image hover toolbox itself. The modal footer should stay a
  calmer viewer/navigation strip instead of duplicating object actions.

## Detection Presentation Anti-Drift Rules

`docs/UI_STANDARD.md` defines shared frontend composition rules and the
semantic source of truth for detection presentation.

To prevent future drift between surfaces:

- Detection badge meaning, species/title trust semantics, and review approval
  semantics must come from one shared source per concern.
- Templates may compose shared values, but they must not re-derive badge labels,
  manual-vs-AI meaning, or species/title trust rules from raw DB fields in new
  local inline logic.
- When a detection surface needs status text, species/title display values, or
  review-state display values, prefer a presenter/helper or shared macro input
  over branching directly in the template.
- New review actions should use `data-*` attributes plus delegated JS handlers.
  Do not add new inline `onclick="..."` handlers with serialized dynamic data.
- If a modal/detail footer exposes detection actions, it must consume the same
  action-frame vocabulary used by tile and filmstrip surfaces instead of
  inventing modal-only wording such as `Relabel` or `Delete`.
- If a semantic rule changes in one detection surface and should apply to other
  detection surfaces, the shared helper/macro contract must be updated first, or
  in the same change.
- Any PR that changes shared detection semantics must update this file if the
  contract, ownership, or allowed patterns changed.

## Review Checklist

- Use `assets/design-system.css` for shared primitives; page-local CSS should be
  limited to surface-specific layout.
- Prefer shared modal and review-stage compositions over building another local
  variant.
- Do not add new legacy Bootstrap button variants to templates.
- Do not duplicate decision-state or badge semantics in templates.
- Do not rename canonical detection actions per surface.
- Do not add new inline event handlers for review/detection interactions; use
  delegated handlers with `data-*` attributes.
- If a JS file is already a mixed-responsibility module, add new work in a
  dedicated file unless it is the same responsibility.
- If a template pattern repeats in 2 or more places, extract it before the
  third copy lands.
- If a shared detection semantic changed, verify whether
  `docs/UI_STANDARD.md` and the active detection-presentation workflow need the
  same update.

---

## 1. Modal Types (mandatory)

**Types**
- `wm-modal` (Detail/Review with Image-Viewer)
- `wm-modal wm-modal--form` (Settings forms, e.g., Add/Edit Camera)

### 1.1 Standard Modal (wm-modal)

```html
<div class="modal fade gallery-modal wm-modal"
     id="modal-{{ group_id }}-{{ detection_id }}"
     tabindex="-1"
     aria-hidden="true"
     data-modal-group="{{ group_id }}"
     data-image-path="{{ image_path }}">

  <div class="modal-dialog modal-xl modal-dialog-centered modal-dialog-scrollable modal-fullscreen-md-down wm-modal__dialog">
    <div class="modal-content wm-modal__content">

      <div class="modal-header wm-modal__header">
        <div class="wm-modal__title">
          <span class="wm-modal__title-text">{{ title }}</span>
          <span class="wm-modal__title-sub">{{ subtitle }}</span>
        </div>
        <button type="button" class="btn-close wm-modal__close" data-bs-dismiss="modal"></button>
      </div>

      <div class="wm-modal__body">
        <div class="wm-modal__image">
          <!-- Standard Image Viewer -->
        </div>
        <div class="wm-modal__info">
          <!-- Info Block -->
        </div>
      </div>

      <div class="wm-modal__action">
        <!-- Standard Action Bar -->
      </div>

    </div>
  </div>
</div>
```

---

**Review-specific modifiers:**
- `wm-modal__body--review` — applied to the modal body in the Review workbench
  context (orphan_modal.html). Adjusts layout for the inline review stage.
- `wm-modal__image--solo` — applied to the modal image container when displaying
  a non-detection image without bbox overlays or sibling panels.

---

### 1.2 Form Modal (wm-modal wm-modal--form)

```html
<div class="modal fade wm-modal wm-modal--form"
     id="modal-settings-{{ form_id }}"
     tabindex="-1"
     aria-hidden="true">

  <div class="modal-dialog modal-lg modal-dialog-centered wm-modal__dialog">
    <div class="modal-content wm-modal__content">

      <div class="modal-header wm-modal__header">
        <div class="wm-modal__title">
          <span class="wm-modal__title-text">{{ title }}</span>
          <span class="wm-modal__title-sub">{{ subtitle }}</span>
        </div>
        <button type="button" class="btn-close wm-modal__close" data-bs-dismiss="modal"></button>
      </div>

      <form class="wm-modal__form" method="post" action="{{ form_action }}">
        <div class="wm-modal__body">
          <div class="wm-modal__fields">
            <!-- Form fields -->
          </div>
        </div>

        <div class="wm-modal__action wm-modal__action--form">
          <div class="wm-modal__actions">
            <button type="button" class="btn btn--secondary" data-bs-dismiss="modal">Cancel</button>
            <button type="submit" class="btn btn--primary">Save</button>
          </div>
        </div>
      </form>

    </div>
  </div>
</div>
```

---

## 2. Tile Types (mandatory)

**Types**
- `wm-tile` (Standard, Gallery/Species/Stream)
- `wm-tile wm-tile--review` (Review/Orphans)
- `wm-tile wm-tile--bbox` (Thumbnail macro with bounding box)

### 2.1 Standard Tile (wm-tile)

```html
<div class="wm-tile" data-detection-id="{{ detection_id }}">
  <button type="button"
          class="wm-tile__button"
          data-bs-toggle="modal"
          data-bs-target="#modal-{{ group_id }}-{{ detection_id }}">

    <div class="wm-tile__media">
      <img class="wm-tile__image"
           src="{{ thumb_url }}"
           alt="{{ common_name }}">

      <span class="wm-tile__badge">{{ count }}</span>
    </div>
  </button>

  <div class="wm-tile__body">
    <span class="wm-tile__name">{{ common_name }}</span>
    <span class="wm-tile__latin">{{ latin_name }}</span>
  </div>
</div>
```

---

### 2.2 Review Tile (wm-tile wm-tile--review) — Legacy

> **Note:** This tile type has been replaced by the `review-stage-panel`
> composition (see §6). The CSS class `wm-tile--review` still exists in
> `design-system.css` but the HTML structure below is no longer produced by any
> template. Retained here for reference only.

```html
<div class="wm-tile wm-tile--review" data-filename="{{ filename }}">
  <div class="wm-tile__select">
    <input class="form-check-input wm-tile__checkbox" type="checkbox" value="{{ filename }}">
  </div>

  <span class="wm-tile__badge wm-tile__badge--reason">{{ reason_label }}</span>

  <button type="button"
          class="wm-tile__button"
          data-bs-toggle="modal"
          data-bs-target="#modal-review-{{ filename|replace('.', '_') }}">
    <div class="wm-tile__media">
      <img class="wm-tile__image"
           src="{{ thumb_url }}"
           alt="{{ filename }}">
    </div>
  </button>

  <div class="wm-tile__body">
    <span class="wm-tile__meta">{{ formatted_date }}</span>
    <span class="wm-tile__name">{{ filename }}</span>
    <span class="wm-tile__size">{{ file_size_str }}</span>
  </div>

  <div class="wm-tile__actions">
    <!-- review actions -->
  </div>
</div>
```

---

### 2.3 BBox Tile (wm-tile wm-tile--bbox)

`data-bbox-*` values are percentages (0-100) from the detection record.

```html
<div class="wm-tile wm-tile--bbox"
     data-bbox-x="{{ bbox_x }}"
     data-bbox-y="{{ bbox_y }}"
     data-bbox-w="{{ bbox_w }}"
     data-bbox-h="{{ bbox_h }}">
  <button type="button"
          class="wm-tile__button"
          data-bs-toggle="modal"
          data-bs-target="#modal-{{ group_id }}-{{ detection_id }}">

    <div class="wm-tile__media wm-tile__media--bbox">
      <img class="wm-tile__image wm-tile__image--bbox"
           src="{{ thumb_url }}"
           alt="{{ common_name }}">
    </div>
  </button>

  <div class="wm-tile__body">
    <span class="wm-tile__name">{{ common_name }}</span>
    <span class="wm-tile__latin">{{ latin_name }}</span>
  </div>
</div>
```

---

## 3. Standard Action Bar

```html
<div class="modal-action-bar">
  <div class="modal-action-bar__group">
    <!-- left buttons -->
  </div>

  <div class="modal-action-bar__group">
    <!-- navigation + close -->
  </div>
</div>
```

---

## 4. Standard Image Viewer

```html
<div class="modal-image-viewer wm-image-viewer">
  <img class="wm-image-viewer__img bbox-base-image"
       src="{{ image_url }}"
       data-detection-id="{{ detection_id }}"
       role="button"
       data-bs-dismiss="modal">

  <canvas class="wm-image-viewer__overlay bbox-overlay"></canvas>
</div>
```

---

## 5. Tile Toolbox (wm-toolbox)

The tile toolbox is the shared action overlay for detection-bearing tiles,
filmstrip items, and modal image viewers. It is rendered by the
`tile_toolbox` macro in `templates/partials/tile_toolbox.html`.

**Host pattern:** Any container that hosts a toolbox adds the class
`wm-toolbox-host`. This enables hover/focus reveal behavior. Used on
`wm-tile`, `obs-filmstrip__item`, and `wm-modal__image`.

**Class vocabulary:**

| Class | Role |
|---|---|
| `wm-toolbox-host` | Container that reveals the toolbox on hover/focus |
| `wm-toolbox` | Toolbox root — positioned overlay inside the host |
| `wm-toolbox--bar` | Bar variant — horizontal strip layout |
| `wm-toolbox__btn` | Individual action button |
| `wm-toolbox__btn--toggled` | Pressed-state modifier for stateful buttons (zoom toggle, future star ratings, …) |
| `wm-toolbox__btn--locked` | Disabled / login-required state |
| `wm-toolbox__fav` | Favorite toggle button (special styling) |
| `wm-toolbox__menu` | Dropdown trigger (three-dot / more button) |
| `wm-toolbox__more` | Alias for the menu trigger |
| `wm-toolbox__dropdown` | Dropdown panel |
| `wm-toolbox__item` | Individual dropdown menu item |

**Stateful toolbox buttons:** any
toolbox button that carries an on/off state — the smart-zoom toggle,
the bbox-overlay toggle, future star ratings — must mirror its state
on **all three** of the following at the same time:

1. `aria-pressed="true|false"` — for assistive tech.
2. `wm-toolbox__btn--toggled` class — for the visual pressed style.
3. State-aware `title=` — short imperative copy that describes what
   the **next** click will do (e.g. `Show full image` when zoomed in,
   `Zoom into bird` when full). When the action degrades (e.g. zoom
   intent persisted but no bbox on the current frame), the title
   becomes a state explanation (`No bounding box — zoom unavailable
   for this frame`).

The smart-zoom toggle is the canonical reference implementation
(`assets/js/gallery_utils.js:applySmartZoomToggleState`). It also
keeps the existing emoji-label swap (`🔍 Zoom` ↔ `🖼 Full`) for
cross-surface consistency with the Gallery / Stream viewers — the
binding pressed-state contract above is the new layer, not a
replacement of the label swap.

**Workspace-scoped persistence:** stateful
toolbox toggles that persist their state across rail navigation must
share **one** persistence scope per workbench, not per individual
viewer element. The Review workbench uses a single `.wm-viewer-scope`
on the `.review-event-panel__grid` plus an explicit
`data-zoom-pref-key="wmb_review_zoom_pref"` so every cell in the grid
shares one zoom intent and the operator's "I want zoomed-in" choice
survives stepping between sibling events.

```html
<div class="wm-toolbox-host">
  <img class="wm-tile__image" src="..." alt="...">

  <div class="wm-toolbox">
    <button class="wm-toolbox__fav" data-action="favorite">...</button>
    <button class="wm-toolbox__btn" data-action="view-details">...</button>
    <div class="wm-toolbox__menu">
      <button class="wm-toolbox__more">...</button>
      <div class="wm-toolbox__dropdown">
        <button class="wm-toolbox__item" data-action="change-species">Change Species</button>
        <button class="wm-toolbox__item" data-action="move-trash">Move to Trash</button>
      </div>
    </div>
  </div>
</div>
```

---

## 6. Review Stage Panel (review-stage-panel)

The Review workbench uses a dedicated composition that replaces the earlier
`wm-tile--review` tile pattern (see §2.2 Legacy). The stage is rendered by
`templates/components/review_stage_panel.html`,
`templates/components/review_event_panel.html`, and
`templates/components/orphan_modal.html`.

**Layout:** Event rail on the left, image viewer/stage in the center,
decision/inspector rail on the right.

**Key class families:**

| Class prefix | Role |
|---|---|
| `review-stage-panel__content` | Outer content wrapper |
| `review-stage-panel__workbench` | Main workbench grid |
| `review-stage-panel__canvas` | Center canvas area |
| `review-stage-panel__viewer-shell` | Viewer container with stable aspect ratio |
| `review-stage-panel__viewer` | Inner viewer |
| `review-stage-panel__viewer-media` | Media container |
| `review-stage-panel__image-frame` | Stable image frame for bbox alignment |
| `review-stage-panel__facts-toggle` | Toggle control for metadata reveal |
| `review-stage-panel__facts-panel` | Collapsible metadata panel |
| `review-stage-panel__facts-grid` | Grid layout for fact items |
| `review-stage-panel__facts-item` | Individual metadata fact |
| `review-stage-panel__decision-rail` | Right-side decision/action rail |
| `review-stage-panel__section` | Grouped section in the decision rail |
| `review-stage-panel__section-label` | Section heading |
| `review-stage-panel__bbox-actions` | BBox action group |
| `review-stage-panel__species-strip` | Quick species selection strip |
| `review-stage-panel__species-btn` | Individual species button |
| `review-stage-panel__nav` | Navigation controls |
| `review-stage-panel__controls` | General control group |

**Section labels in the decision rail:**
`BBox`, `Species`, `Decision`, `Utilities`

---

## 6b. Review Event Panel (review-event-*)

The Review workbench operates on a single biological unit — the `BirdEvent`
(same species, gap ≤ 30 min) — instead of two parallel paths (per-detection
queue + bulk cluster). There is no `Queue | Bulk Review` mode switch. The
Review rail renders event cards; the stage panel composes the event detail
via the `review-event-*` class family, still inside the shared
`review-stage-panel` shell from §6.

The per-detection queue panel remains reachable as a single-shot escape
hatch from the right control rail via `Review in Queue`, which opens the
event cover frame in the queue. Per-cell drill-downs are gone — mixed
events are now resolved inside the event grid itself via per-frame
`Keep this frame` / `Trash this frame` toggles (see **Mixed-event
resolution** below).

### Mixed-event resolution

Every actionable cell in the event grid carries a local decision toggle
(`data-review-frame-decision`) with two states: `keep` (default) and
`trash`. State is held in the DOM dataset and in cell classes
(`.is-frame-keep` / `.is-frame-trash`); no API call fires on click.

`Approve Event` inspects the toggle state when pressed:

- **All frames `keep`** → fast path, POST to `/api/review/event-approve`
  (unchanged contract).
- **Any frame `trash`** → mixed path, POST to `/api/review/event-resolve`
  with `keep_detection_ids` + `trash_detection_ids`. The server enforces
  disjoint sets, full event coverage, and a non-empty `keep` list.
  `species` + `bbox_review` apply only to the `keep` frames; `trash`
  frames are rejected in the same transaction via the same code path as
  `/api/review/event-trash`, so image visibility recomputes consistently.

`Move Event to Trash` stays as the homogeneous all-wrong shortcut and
still targets `/api/review/event-trash` directly. The operator never has
to drill into the per-detection queue to split a mixed event.

### Event rail — text-only

The side rail renders every event as a compact text row — **no thumbnail,
no per-card trail preview**. The detail view owns the trail via its own
mini-map. The former cover + trail layout is retired.

**Rail class families:**

| Class | Role |
|---|---|
| `review-event-browser` | Side rail container holding the event cards |
| `review-event-browser__grid` | Vertical stack of event cards |
| `review-event-card` | Individual event card — compact count-first row |
| `review-event-card__accent` | Left accent bar, coloured only when `.is-active` |
| `review-event-card__count` | Numeric summary block (`N` + `frame(s)`) |
| `review-event-card__count-value` | Large frame count, tabular-nums |
| `review-event-card__count-label` | Small uppercase `frame(s)` label |
| `review-event-card__body` | Secondary stack: time first, species second |
| `review-event-card__time` | Primary scan target after the count block |
| `review-event-card__species` | Secondary common-name line; legible but not required for scanning |
| `review-event-card__badge` | Eligibility badge (`--ready` / `--fallback`) |

**Rail scan rule:** the rail is now
**count-first**. Operators should be able to scan the left column by
frame count + time window alone; the species line is secondary support,
not the primary wayfinding cue.

**Active-state contract:** the active
card is unmistakable from the rest of the rail. All of the following
ride on the same `--color-primary` token family so the dark-mode scope
picks the highlight up automatically by redefining the token alone,
without any surface-rule edit:

1. `.is-active` adds an accent-fill background (linear gradient on
   the warm-green primary family) plus a 2 px accent border, so the
   card visually pops out of the rail.
2. The left accent bar (`.review-event-card__accent`) lights up in
   the same primary token.
3. The active card mirrors its state as `aria-current="true"` for
   assistive tech. `review_workspace.js` rotates both `is-active`
   **and** `aria-current` atomically inside the same DOM-write block
   on rail navigation, so there is no flicker when the operator
   presses `‹ / ›`.
4. **Stage-connecting cue:** `.review-stage-panel.is-active` carries a
   3 px left border in the same `--color-primary` token, so the rail
   card and the stage panel read as one bound unit at a glance.
5. Non-active cards drop to ~92% opacity and a much lighter border so
   the contrast between the active card and the rest is unmistakable
   without the rail looking "broken" when nothing is selected.
6. The card padding is compensated by 1 px on every side when active
   so the geometry does not jitter as the rail rotates between cards.
7. Token-only — no hardcoded hex anywhere in the active-state rules.
   The dark-mode scope (see §6d) redefines
   `--color-primary*` inside an `@media (prefers-color-scheme: dark)`
   block and the entire highlight chain follows for free.

### Event stage — equal-size detection grid

The event-mode stage renders every `event.members` entry as a first-class
cell in a responsive grid. **There is no cover frame, no filmstrip strip,
no special per-event "primary image".** Every frame is comparably sized,
comparably actionable, and carries its own bbox overlay + zoom toggle +
toolbox — which makes visual species decisions a direct comparison task
instead of a filmstrip toggling task.

**Grid class families:**

| Class | Role |
|---|---|
| `review-event-panel__grid` | Responsive CSS grid: `repeat(auto-fill, minmax(240px, 1fr))` |
| `review-event-panel__cell` | Individual detection cell — a `wm-toolbox-host` with its own `render_image_viewer` + `tile_toolbox` |
| `review-event-panel__cell-media` | Image frame; positioning parent for `.review-species-ref` and the bbox canvas |
| `review-event-panel__cell-time` | Timestamp badge overlaid on the media frame |
| `review-event-panel__cell-body` | Species + reason line |
| `review-event-panel__cell-species` | Common name, tinted with `var(--cell-species-colour)` |
| `review-event-panel__cell-reason` | `reason_label` for the frame |
| `review-event-panel__cell-action` | Per-frame `Keep this frame` / `Trash this frame` decision toggle (`data-review-frame-decision`); carries `is-keep` / `is-trash` state classes |
| `review-event-panel__grid-caption` | One-line caption under the grid: `N frame(s) of <species> across <duration>. Approve once the pattern is consistent.` |
| `review-event-panel__trail-section--aside` | Mini-map variant inside the right control rail; replaces the retired `--compact` under-grid form |
| `review-event-panel__trail-map` | Full trail visualisation inside the aside section |
| `review-event-panel__trail-box` | Individual bbox marker inside the trail map |
| `review-event-panel__controls` | Right-side event control rail |
| `review-event-panel__species-pill` | Event species summary pill |
| `review-event-panel__badge` | Eligibility badge (`--ready` / `--fallback`) |

**Per-cell contract:**
1. Every cell is a `wm-toolbox-host` — the delegated toolbox handlers
   (`data-action="favorite" | "view-details" | "change-species" | "move-trash"`)
   reach the new DOM without any new event listeners.
2. Every cell mounts `tile_toolbox` with `show_viewer_tools=true` so the
   zoom toggle and bbox-overlay toggle are available on every frame, not
   just on the former cover image. This is the binding form of the
   on-image toolbox rule plus the forward-compatibility hook for the
   shared zoom-toggle pressed state.
3. Every cell uses `render_image_viewer` with its own per-frame `bbox`
   payload — each cell has its own bbox canvas, scoped to its own
   detection.
4. The first cell in DOM order carries `data-cover-detection="true"` as a
   reserved hook for a future "jump to cover" affordance. It carries **no**
   visual difference from the other cells — equal-size grid is mandatory
   for every frame.
5. Cells marked `data-context-only="1"` (rare outside the continuity-batch
   path) do **not** mount a toolbox, carry a dashed border, and are muted
   to 65% opacity — the same visual treatment as the batch anchor cells
   from §6c.
6. Grid-cell species-colour wiring rides on the same
   `data-species-colour` + inline `--cell-species-colour` style pattern
   as the continuity-batch cells (see §6d). The species colour token drives the cell border, the
   cell-species name tint, and an inset left accent on the media frame.
   The `tile_toolbox` `current_bbox_json` payload carries a
   `speciesColour` field so a future canvas-draw fix can colour the bbox
   stroke without touching the template.

**Mini-map placement:** the trail section lives
as the **first element of the right control rail**, via the
`--aside` modifier. Always visible. Never behind an accordion, a
modal, or a hover trigger. The retired `--compact` under-grid
variant is retired. The map carries an `aspect-ratio: 16/9` so it
keeps a sensible shape inside the narrow aside without dominating
the rail's vertical budget.

**Species change-receipt strip (shared component):** the receipt is rendered by the shared partial
`templates/partials/review_species_receipt.html` and mounted on both
review surfaces (orphan modal Species section, event panel Species
section). The contract:

1. The server always renders an empty slot wrapper
   (`[data-review-species-receipt]`). The slot is hidden via CSS
   when empty (`.review-species-receipt-slot:not(.is-active)`).
2. JS rebuilds the slot contents on every quick-pick species click
   inside `applyReviewSpeciesUi(controls, species, options)`. The
   write happens in the same DOM mutation as the `is-selected` toggle,
   so prev/new + Undo land atomically with the quick-pick state.
3. JS uses `createElement` + `textContent` exclusively — never
   `innerHTML`. XSS-safe by construction. Pinned by the
   `test_review_workspace_js_handles_species_change_receipt` template
   test.
4. The "previous" side is anchored to `data-original-species` /
   `data-original-species-common` on the controls root. These attributes
   are set once at initial render and **never** mutated by JS.
5. The Undo button (`data-review-panel-action="undo_species_change"`)
   resolves the controls root via
   `closest('[data-review-controls], [data-review-event-controls]')`
   and reverts to the original via `applyReviewSpeciesUi(controls,
   originalKey, ...)` — the same code path as a regular species click.
6. `stepReviewItem` calls `clearReviewSpeciesReceipts()` synchronously
   **before** loading the next panel so a receipt from the previous
   item never leaks into the next one.
7. The event-panel surface uses the same receipt strip for its new
   event-local species change path. Without multi-select, quick picks +
   the picker button update the pill + receipt immediately, but the
   choice stays local until `Approve Event` commits it.
8. When event-grid multi-select is active and at least one frame is
   checked, clicking a quick-pick species in the right rail no longer
   changes the event pill. Instead it relabels the selected frames
   immediately via `/api/moderation/bulk/relabel`, keeps the operator in
   multi-select mode, clears the selection, and updates the selected
   grid-cell species labels in place so the change is visible "oben bei
   den Bildern" before any panel reload.

**Data attributes (delegated handlers):**

| Attribute | Role |
|---|---|
| `data-review-event` | Side-rail event card |
| `data-event-key` | Stable event identifier (DOM + API) |
| `data-review-event-controls` | Event control rail root |
| `data-review-event-bbox-toggle` | Event-scope bbox correct/wrong toggle |
| `data-review-event-bbox-copy` | Copy element inside the bbox toggle |
| `data-review-panel-action="approve_event"` | Single-button event approve action |
| `data-review-panel-action="trash_event"` | Reject every detection in the event |
| `data-review-panel-action="select_event_species"` | Event-local quick-pick species selection, or selected-frame relabel shortcut while multi-select is active |
| `data-review-panel-action="open_event_species_picker"` | Event-local full species picker trigger |
| `data-review-open-item` | Drill-down into the per-detection queue panel |
| `data-cover-detection="true"` | Reserved hook on the first grid cell (no visual effect today) |
| `data-context-only="1"` | Read-only Gallery anchor cell (no toolbox, dashed border, muted) |

**Section labels in the decision rail:** `Species`, `BBox`, `Decision`, `Utilities`

**Species hint copy:** `Event-wide species choice stays local until you approve the event. With selected frames, a quick-pick relabels them immediately.`

**Action verbs:** `Approve Event` (primary), `Move Event to Trash`,
`Choose another species`, `Review in Queue` / `Open in Queue`
(drill-down). The canonical event-wide destructive verb is
`Move Event to Trash`, which rejects detections only; image visibility
is then recomputed from the remaining active detections. If an image has
no active detections left, it becomes `no_bird` and surfaces in Trash.

**Legacy note:** the former cover + filmstrip layout
(`__filmstrip-shell`, `__filmstrip-head`, `__member`,
`__member-media`, `__member-image`, …) is retired. If you find
references to those classes, do not revive them; migrate them to the
`__grid` / `__cell` vocabulary.

---

## 6c. Continuity Batch Stage

When the continuity-batch helper attaches a continuity batch
to a Review event, the event panel **prepends** a batch-scoped stage
above the single-event sections. In the current event-grid layout,
the "focused split event" detail under the batch panel
is the same `review-event-panel__grid` that the normal event-mode path
uses (§6b), visually de-emphasised via
`.review-stage-panel__content--batch` (opacity dim + a `::before`
"Focused split event" label). The retired `__filmstrip-shell` variant
is no longer rendered in either mode.

**Hard rules:**
- The continuity batch stage only renders when `event.continuity_batch`
  is set on the payload. The backend wires this in
  `_load_event_with_continuity_batch` (`web/blueprints/review.py`)
  whenever the event participates in a batch from
  `build_review_continuity_batches`.
- `Already in Gallery` (read-only context anchors) and `Review now`
  (every actionable frame across sibling events) render as two
  distinct grids inside `.review-batch-panel`.
- Context anchors carry `data-context-only="1"`, never get a
  `tile_toolbox`, never accept clicks (`pointer-events: none`), and
  never appear in any `Approve Batch` payload.
- A combined continuity mini-map (`.review-batch-panel__minimap`) is
  always visible, driven by `batch_bbox_map`. Markers use the
  `--context` and `--review` modifier classes plus an `is-active-event`
  ring on whichever frames belong to the rail-selected split event.
- The `Apply <species> to all N review frames` CTA is hard-gated on
  `continuity_batch.recommended_species` being truthy. When mixed
  anchors disagree the panel shows a neutral hint, **never** an
  aggressive default CTA.
- `Approve Batch` is rendered disabled and only enables once every
  actionable cell has converged on the same species (handled by
  `updateBatchApproveGate` in `assets/js/review_workspace.js`).
- The batch approval submits to `/api/review/event-approve` **without**
  an `event_key` so the backend skips the strict
  `event_key`/`detection_ids` parity check. The server-side guard
  refuses any submitted detection whose image is already
  `review_status='confirmed_bird'` (i.e. a Gallery anchor); the JS-side
  guard mirrors this by filtering against
  `data-batch-context-detection-ids` before posting.

**Class vocabulary:**

| Class | Role |
|---|---|
| `review-stage-panel__content--batch` | Outer wrapper modifier; de-emphasises the single-event sections below |
| `review-batch-panel` | Continuity batch stage root |
| `review-batch-panel__head` | Title block + anchor chips |
| `review-batch-panel__title` | "Anchored by …" / "Mixed Gallery anchors" |
| `review-batch-panel__anchor-chips` | Confirmed-anchor species chips |
| `review-batch-panel__minimap` | Always-visible combined continuity mini-map |
| `review-batch-panel__minimap-box--context` | Context-anchor bbox marker (dashed) |
| `review-batch-panel__minimap-box--review` | Actionable review-frame bbox marker |
| `.is-active-event` | Marker ring for the rail-selected split event |
| `review-batch-panel__decision` | Decision strip wrapper |
| `review-batch-panel__cta` | Apply-to-all primary CTA |
| `review-batch-panel__approve` | Approve Batch button (disabled until convergence) |
| `review-batch-panel__receipt` | Apply receipt slot with Undo |
| `review-batch-panel__section--anchors` | Already in Gallery section |
| `review-batch-panel__section--review` | Review now section |
| `review-batch-cell--context` | Read-only context anchor cell |
| `review-batch-cell--review` | Actionable review-frame cell |
| `is-active-event-member` | Cell modifier highlighting the rail-selected event |

**Data attributes (delegated handlers):**

| Attribute | Role |
|---|---|
| `data-review-batch-panel` | Stage root marker |
| `data-batch-key` | Continuity batch identifier |
| `data-batch-recommended-species` | Pre-computed batch recommendation |
| `data-batch-review-detection-ids` | Comma-separated actionable detection ids |
| `data-batch-context-detection-ids` | Comma-separated read-only anchor detection ids — used as the JS-side refusal guard |
| `data-active-event-key` | The currently rail-selected split event |
| `data-context-only` | `1` for anchors, `0` for actionable frames |
| `data-batch-event-key` | Source event_key for sibling-event highlighting |
| `data-review-batch-receipt` | Receipt slot for the apply-to-all summary |
| `data-review-panel-action="apply_batch_species"` | Apply CTA |
| `data-review-panel-action="undo_batch_species_change"` | Undo CTA inside the receipt |
| `data-review-panel-action="approve_batch"` | Approve Batch action |

---

## 6d. Species Colour Coding + Reference Images

Every Review surface that displays a detection or species name carries
a deterministic colour token and an inline species reference image.
This is the fastest possible visual sanity check the operator gets:
"everything blue is Kohlmeise, everything orange is Blaumeise" without
reading a label.

**Wong (2011) palette tokens** live in `:root` of `design-system.css`:

| Token | Hex | Notes |
|---|---|---|
| `--species-colour-0` | `#0072B2` | blue |
| `--species-colour-1` | `#E69F00` | orange |
| `--species-colour-2` | `#009E73` | green |
| `--species-colour-3` | `#CC79A7` | pink |
| `--species-colour-4` | `#56B4E9` | sky |
| `--species-colour-5` | `#D55E00` | vermilion |
| `--species-colour-6` | `#F0E442` | yellow — non-text only, insufficient contrast on white |
| `--species-colour-7` | `#000000` | black — fallback |

The JS canvas-bbox draw code in `assets/js/gallery_utils.js` mirrors
the same palette as `SPECIES_COLOURS` because a detached canvas can't
read the CSS custom property from a DOM ancestor. Both must stay in
sync — if a slot is added, update both files in the same commit.

**Server-side contract:**
- `assign_species_colours(scientific_names)` returns
  `dict[str, int]` mapping each name to a slot in `[0..7]`. Sort is
  alphabetical, slot wraps at 8, blanks ignored. Deterministic across
  identical inputs.
- The colour map is computed **once per workspace load** from the full
  `raw_events` set (including pure-context anchors) so the same
  scientific name keeps the same slot across the rail, the filmstrip,
  the orphan modal, and the continuity batch stage.
- The fragment endpoint (`/api/review/event-panel/<event_key>`) builds
  its own workspace-scoped map from the same raw events and stamps
  even when no batch is present. The "no-batch early return without
  stamping" regression is locked
  down by `tests/test_species_colour.py::test_stamp_species_display_on_event_without_members_or_batch`.
- `species_ref_image_url` is resolved from a process-scoped cache of
  `assets/review_species/`. `.webp` wins over `.png`. Filenames must
  match the scientific name (`Parus_major.webp` etc.) — the same
  shape used by `species_key` in the database.

**Template wiring:**
- Every coloured surface carries `data-species-colour="N"` plus an
  inline `style="--cell-species-colour: var(--species-colour-N)"` so
  one CSS variable drives bbox border, accent bar, badge tint, and
  the reference image overlay border.
- The reference image renders as `<img class="review-species-ref">`
  inside the same `.wm-toolbox-host` frame, anchored **top-right** with
  `pointer-events: none` so it never blocks toolbox hover. Top-right
  keeps it clear of the tile toolbox strip that anchors to the bottom
  edge of each frame. The image must keep an explicit square box
  (`width == height`, `aspect-ratio: 1 / 1`) so global
  `img { max-width: 100%; height: auto; }` rules do not stretch it
  into an oval.
- Fallback when no reference image exists for a species:
  `<span class="review-species-ref review-species-ref--initial">` with
  the first letter of the common name. **Never** a broken-image icon.

**Responsive bbox label:** `truncateBboxLabel(text, maxPx, ctx)`
in `gallery_utils.js` shrinks labels by **dropping a level**, never
by scaling the font. Order: full → first 4 chars + `.` → first letter
→ empty. On mobile (`window.innerWidth < 640`) it starts at the
abbreviated level even for wide boxes. The container border + species
colour still carry species identity when the text is dropped entirely.

**Surfaces currently wired:**
- Event-mode detection grid cells (`.review-event-panel__cell`) —
  replaces the retired cover image + filmstrip member bindings
- Continuity batch cells, both `--context` and `--review`
  (`.review-batch-cell`)
- Orphan modal cover frame
- Quick-pick species buttons (`.review-stage-panel__species-btn`)

**Surfaces still needing wiring:**
- Stream / Gallery / Trash surfaces

**Dark-mode palette:** the
Wong (2011) palette has a dark-variant scope inside an
`@media (prefers-color-scheme: dark)` block at the top of
`assets/design-system.css` that redefines every load-bearing
`:root` token. The scope follows the OS preference via
`prefers-color-scheme` without any JS toggle infrastructure.

- **Wong dark variants.** Same hues as the light palette but lifted
  toward the brighter Wong family so they stay readable on dark
  backgrounds without breaking the colour-blind-friendly contrast
  relationships between slots:
  | Slot | Light | Dark | Note |
  |---|---|---|---|
  | 0 | `#0072B2` | `#4aa3d9` | blue — lifted |
  | 1 | `#E69F00` | `#f3b340` | orange — lifted |
  | 2 | `#009E73` | `#3fbe93` | green — lifted |
  | 3 | `#CC79A7` | `#e09bc2` | pink — lifted |
  | 4 | `#56B4E9` | `#7bc4ee` | sky — lifted |
  | 5 | `#D55E00` | `#e8793a` | vermilion — lifted |
  | 6 | `#F0E442` | `#f5ea63` | yellow — **text-safe on dark**, the light-scope "non-text only" constraint no longer applies |
  | 7 | `#000000` | `#ececec` | fallback — flipped from black to near-white so the fallback is actually visible on dark surfaces |

- **Primary highlights ride on `--color-primary*`, so they pick up
  the dark scope automatically.** The active-event accent fill, the
  left accent bar, the stage-connecting cue, the grid border accent,
  and the species change-receipt `border-left` are all
  `var(--color-primary, ...)` today. The dark scope redefines
  `--color-primary` from `#53676b` to `#92a7ab` and the entire
  highlight chain tracks the change without a single surface-rule
  edit.

- **Review-surface overrides block.** A secondary
  `@media (prefers-color-scheme: dark)` block near
  `.review-event-panel__species-meta` in `design-system.css`
  explicitly overrides the handful of Review-scope rules that paint
  **hardcoded** rgba() light backgrounds (the rail aside linear
  gradient, the rail card background, the active rail card gradient,
  the trail-section aside background, the detection-grid gradient,
  the grid-cell background, the species pill, the species change-
  receipt background, the eligibility badges, the species quick-pick
  media fallback, and the continuity-batch panel fallback).
  Token-only overrides cannot reach those surfaces, so they are
  written out explicitly. Stream / Gallery / Trash rules are **not**
  touched here — wire them separately when those surfaces adopt the
  same contract.

- **JS `SPECIES_COLOURS` mirror.** The canvas-side palette in
  `gallery_utils.js` now reads `--species-colour-0..7` live from
  `:root` via `getComputedStyle(document.documentElement)`, so the
  bbox stroke automatically tracks the dark scope. CSS stays the
  single source of truth. A `window.matchMedia('(prefers-color-scheme: dark)')`
  listener invalidates the cache on OS theme change so the next draw
  picks up the new tokens. A `SPECIES_COLOURS_FALLBACK` array
  remains as a SSR / detached-node safety net with the light
  variants. `const SPECIES_COLOURS` stays as a `Proxy` that always
  returns the currently resolved palette, so legacy index access
  (`SPECIES_COLOURS[slot]`) keeps working without any call-site
  edits.

- **Slot 6 (yellow) text-contrast:** the light-scope constraint
  "non-text only" can be relaxed inside the dark scope because
  `#f5ea63` on a dark surface has comfortable contrast. If any
  future surface paints a text label inside a slot-6-coloured
  element, remember to gate it on the OS preference — the light
  fallback is still not text-safe.

---

## 6e. Subgallery Concurrent Visits

Two species in the same 30-minute window stay **two `BirdEvent`s**
in the data layer — the same-species split rule at `core/events.py`
is not weakened. The Subgallery groups them visually so the operator
can see at a glance that the visits happened at the same time.

**Binding architectural decision:** the grouping is UI-only. No new
database tables. No changes to `core/events.py`.

**Helper:** `core.gallery_core.group_concurrent_observations(observations, *, window_minutes=5.0)`
- Pure function, no DB calls, no mutation.
- Takes the observation list returned by `group_detections_into_observations`.
- Returns `list[list[dict]]` — each inner list is one visit window.
- Tolerance: `window_minutes=5` by default. An observation joins the
  current window when its `start_time` is within tolerance of the
  running max `end_time` of that window.
- Deterministic given the same input set, independent of input order.

**Template contract (`templates/subgallery.html`):**
- The Subgallery iterates `visit_windows`, not the flat `observations`
  list. `observations` stays in the template context for modal
  rendering (backwards compatibility).
- A visit window of **one** observation renders with **no** wrapper —
  the common case must not gain visual noise.
- A visit window of **two or more** observations is wrapped in a
  `.concurrent-visit` shell with a single header line
  (`HH:MM:SS – HH:MM:SS · N species`).
- Each observation inside a visit window remains its own `.wm-tile`,
  clickable, with its own filmstrip, its own toolbox, its own modal.
- Species colouring (§6d) inside a concurrent visit uses the same
  palette assignments as elsewhere — a window with Elster + Kohlmeise
  shows two different colours inside one shell, which is exactly the
  point.

**BEM block (`assets/design-system.css`):**
- `.concurrent-visit` — the outer container. Full-row child of
  `#subgallery-grid` via `grid-column: 1 / -1`.
- `.concurrent-visit__header` — single-line header with time range +
  species count.
- `.concurrent-visit__time` — tabular-nums time range.
- `.concurrent-visit__meta` — species count, separated by a middle dot.
- `.concurrent-visit__shell` — nested grid that mirrors
  `.gallery-grid`'s `minmax(240px, 1fr)` so tiles inside the shell
  line up visually with tiles outside.
- Background uses `--color-surface-2`, clearly lighter than the tile
  background, so tiles inside still read as individual cards.

**Sort behaviour:**
- The server-side route re-sorts visit windows using the active
  `sort_by` parameter applied to the first observation of each window
  (`time_desc` → `end_time` desc, `time_asc` → `start_time` asc,
  `score` → `max(best_score)` desc, `species` → common name asc).
- Ties resolve deterministically via the internal
  `(start_time, end_time, observation_id)` sort inside
  `group_concurrent_observations`.

**Pagination note (follow-up, not implemented yet):**
- Today the helper runs on the already-paged slice, so a visit window
  that would straddle a page boundary splits across pages. When the
  Subgallery gains pagination that cares about this, the rule becomes
  "paginate by visit window, not by observation".

---

## Rules

1. Every modal structure uses a defined type: `wm-modal` or `wm-modal wm-modal--form`.
2. Every tile structure uses a defined type: `wm-tile`, `wm-tile wm-tile--review` (legacy), or `wm-tile wm-tile--bbox`.
3. The Review workbench uses the `review-stage-panel` composition (§6), not standalone tiles.
4. Every detection-bearing surface uses `tile_toolbox` (§5) for action overlays.
5. No template may build its own modal, tile, or toolbox structures.
6. Only these classes may be used.
7. CSS refers exclusively to these classes.
8. Continuity batch approvals (§6c) **must** post to `/api/review/event-approve` without an `event_key` and **must** filter `data-batch-context-detection-ids` out of the payload before posting. The server-side guard (refusal on `images.review_status='confirmed_bird'`) is the second line of defence, not the first.
9. Species colour slots (§6d) are workspace-scoped, deterministic, and assigned across the union of (actionable events ∪ context anchors ∪ orphans). Per-event recomputation is forbidden — the same scientific name must keep one slot across the rail, the event-mode detection grid, the orphan modal, the batch stage, and the canvas bbox stroke.
10. New surfaces that ship a bbox overlay must read the species slot from the host element's `data-species-colour` (or `box.speciesColour` on the canvas side), not from the legacy `BBOX_COLORS` rotation.
