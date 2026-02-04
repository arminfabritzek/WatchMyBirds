# WatchMyBirds UI Standard (mandatory)

This file defines the mandatory DOM structure for:
- Modals (including types/variants)
- Thumbnails / Tiles (including types/variants)
- Action-Bar
- Image-Viewer

From Phase A Task A.5 onwards, templates must exclusively use these structures.
No deviations. No custom interpretations.

Variants follow the BEM modifier `--` and are always set in addition to the base class
(e.g., `wm-modal wm-modal--form`, `wm-tile wm-tile--review`).

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

### 2.2 Review Tile (wm-tile wm-tile--review)

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

## Rules

1. Every modal structure uses a defined type: `wm-modal` or `wm-modal wm-modal--form`.
2. Every tile structure uses a defined type: `wm-tile`, `wm-tile wm-tile--review`, or `wm-tile wm-tile--bbox`.
3. No template may build its own modal or tile structures.
4. Only these classes may be used.
5. CSS refers exclusively to these classes.
