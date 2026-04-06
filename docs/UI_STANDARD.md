# WatchMyBirds UI Standard

This file defines the current branch frontend conventions and the preferred
shared DOM structures for:
- Modals (including types/variants)
- Review-stage panels
- Thumbnails / Tiles (including types/variants)
- Action bars
- Image viewers

This is migration-aware guidance for the current branch state. It is not a claim
that every legacy surface has already been converted.

Variants follow the BEM modifier `--` and are always set in addition to the base
class (for example `wm-modal wm-modal--form`, `wm-tile wm-tile--review`).

## Current Branch Conventions

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

`docs/UI_STANDARD.md` defines shared frontend composition rules. The architecture
lane in
`agent_handoff/workflow/active/main/2026-03-28_ARCHITECTURE_detection-presentation-source-of-truth_ACTIVE_ONHOLD.md`
defines the semantic migration target for detection presentation.

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
| `wm-toolbox__fav` | Favorite toggle button (special styling) |
| `wm-toolbox__menu` | Dropdown trigger (three-dot / more button) |
| `wm-toolbox__more` | Alias for the menu trigger |
| `wm-toolbox__dropdown` | Dropdown panel |
| `wm-toolbox__item` | Individual dropdown menu item |

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
`templates/components/review_stage_panel.html` and
`templates/components/orphan_modal.html`.

**Layout:** Queue rail on the left, image viewer/stage in the center,
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

## Rules

1. Every modal structure uses a defined type: `wm-modal` or `wm-modal wm-modal--form`.
2. Every tile structure uses a defined type: `wm-tile`, `wm-tile wm-tile--review` (legacy), or `wm-tile wm-tile--bbox`.
3. The Review workbench uses the `review-stage-panel` composition (§6), not standalone tiles.
4. Every detection-bearing surface uses `tile_toolbox` (§5) for action overlays.
5. No template may build its own modal, tile, or toolbox structures.
6. Only these classes may be used.
7. CSS refers exclusively to these classes.
