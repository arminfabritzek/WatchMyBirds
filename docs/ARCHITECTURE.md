# Architecture Documentation

## 1. Overview
WatchMyBirds is an AI-powered bird detection system that provides real-time video analysis and a server-side rendered web interface. It relies on a high-performance Flask + Jinja2 architecture for the core UI and administration. Client-side components are used strictly for specific complex interactions and are not the primary rendering model. The system emphasizes low-latency MJPEG streaming, authority-based metadata, and immutable file management.

## 2. Core Architectural Invariants
*   **Originals are Immutable:** The original captured images (`originals/`) are the primary source of truth and must never be modified in place.
*   **Derivatives are Disposable:** Optimized images and thumbnails (`derivatives/`) are caches generated from originals and can be regenerated or deleted at any time.
*   **Database as Metadata Authority:** The SQLite database is the sole authority for metadata. It contains *NO* absolute filesystem paths, only filenames and relative references resolved dynamically.
*   **PathManager Authority:** All filesystem operations (read/write/delete) MUST resolve paths via `utils.path_manager.PathManager`. No module is permitted to construct filesystem paths manually.
*   **Single Image Ops Source:** `utils.image_ops` is the sole canonical module for image processing logic (e.g., cropping).
*   **No Dash Dependency:** The legacy Dash application is deprecated. All new UI features must use Flask/Jinja2.
*   **Deletion Integrity:** Hard deletions MUST attempt removal of files from disk *before* removing database records. If a file is missing from disk, the operation MUST NOT abort; it must proceed to ensure the database record is removed.

## 3. Data & Storage Model
*   **Originals:** `OUTPUT_DIR/originals/YYYY-MM-DD/filename.jpg`
    *   Primary asset. Exists once per capture.
*   **Derivatives:** `OUTPUT_DIR/derivatives/[optimized|thumbs]/YYYY-MM-DD/[filename]`
    *   Generated on demand or at ingest.
    *   Originals and derivatives MUST share the same identifying base filename structure; differentiation is handled via directory location and file extension.
*   **Database (`images.db`):**
    *   `images` table: Stores filename and global metadata.
    *   `detections` table: Stores bounding boxes, scores, and classifications. Links to `images`.
    *   `classifications` table: Stores species predictions.

## 4. Key Modules and Responsibilities
### `utils/path_manager.py`
*   **MUST:** Be the single source of truth for all path resolution.
*   **MUST:** Handle date-based directory structures (`YYYY-MM-DD`).
*   **MUST NOT:** Perform actual file I/O (only path string manipulation).

### `utils/image_ops.py`
*   **MUST:** Contain all shared image manipulation logic (cropping, padding).
*   **MUST:** Be stateless and pure (functional).

### `detectors/detection_manager.py`
*   **MUST:** Be a thin orchestrator for the AI pipeline (Frame → Detect → Crop → Classify → Save → Notify).
*   **MUST:** Delegate all work to specialized Services:
    - `DetectionService` - Object detection (lazy-loaded Detector)
    - `CropService` - Image cropping for classification
    - `ClassificationService` - Species classification
    - `PersistenceService` - Image/detection saving, EXIF, database
    - `NotificationService` - Telegram notifications with cooldown
*   **MUST NOT:** Import or use directly:
    - `utils.image_ops` (use CropService)
    - `detectors.detector.Detector` (use DetectionService)
    - `detectors.classifier.ImageClassifier` (use ClassificationService)
    - `utils.telegram_notifier` (use NotificationService)
    - `piexif` (use PersistenceService)
*   **MUST NOT:** Implement custom image processing, path logic, or notification logic.

### `utils/file_gc.py`
*   **MUST:** Handle safe deletion of files and database records.
*   **MUST:** Operate exclusively on ABSOLUTE paths resolved via `PathManager`.
*   **MUST:** Ensure referential integrity (don't delete shared files if used elsewhere).

### `web/web_interface.py`
*   **MUST:** Serve the web UI via Flask routes.
*   **MUST:** Use `path_manager` to resolve files for serving (`send_from_directory`).
*   **MUST NOT:** Contain legacy Dash callbacks or layout logic.

## 5. Change Rules
*   **Storage Path Changes:** If the filesystem structure changes, `PathManager` MUST be updated. All call sites relying on it will automatically inherit the change.
*   **New Routes:** All new web routes MUST be implemented in Flask (`server.route`).
*   **Path Construction:** No new code may use `os.path.join` to build storage paths manually. Use `path_manager`.
*   **Cross-Cutting Impact:** Any change affecting storage layout, deletion logic, or image processing MUST trigger a simultaneous review of `PathManager`, `detection_manager`, `file_gc`, and `web_interface` to ensure invariant consistency.

## 6. Non-Goals
*   **Client-Side Rendering (CSR):** The core gallery is Server-Side Rendered (SSR). We do not aim to move the app to a SPA framework.
*   **Cloud Storage:** The system is designed for local filesystem storage (NAS/Disk). Cloud sync is an external concern.
*   **Dash UI:** Dash-based UI components are intentionally deprecated. Flask/Jinja2 is the only supported UI layer.

---

## 7. Service Layer Architecture

The web layer follows a strict three-tier architecture:
- **Web Layer** (`web/`): Flask Routes, Request-Handling, Template-Rendering
- **Service Layer** (`web/services/`): Thin wrappers, web-specific logic
- **Core Layer** (`core/`): Business logic, database access

### Layer Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         WEB LAYER                            │
│   web/web_interface.py (Flask Routes)                        │
│   web/blueprints/                                            │
│   ├── analytics.py     (Analytics UI + API)                 │
│   ├── api_v1.py        (REST API v1)                        │
│   ├── auth.py          (Login/Logout)                       │
│   ├── backup.py        (Backup/Restore UI)                  │
│   ├── inbox.py         (Inbox/Ingest UI)                    │
│   ├── review.py        (Orphan Review UI)                   │
│   └── trash.py         (Trash Management)                   │
│                             │                                │
│                             ▼                                │
│   ┌─────────────────────────────────────────────────┐       │
│   │              web/services/                       │       │
│   │   analytics_service.py                           │       │
│   │   backup_restore_service.py                      │       │
│   │   db_service.py                                  │       │
│   │   detections_service.py                          │       │
│   │   gallery_service.py                             │       │
│   │   ingest_service.py                              │       │
│   │   onvif_service.py                               │       │
│   │   path_service.py                                │       │
│   │   settings_service.py                            │       │
│   └─────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        CORE LAYER                            │
│   core/                                                      │
│   ├── analytics_core.py    (Analytics aggregations)         │
│   ├── backup_restore_core.py (Backup/Restore logic)         │
│   ├── db_core.py           (Database connection mgmt)       │
│   ├── detections_core.py   (Detection CRUD operations)      │
│   ├── gallery_core.py      (Gallery business logic)         │
│   ├── ingest_core.py       (Image ingest logic)             │
│   ├── onvif_core.py        (ONVIF camera operations)        │
│   ├── path_core.py         (Path resolution)                │
│   └── settings_core.py     (Settings management)            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      INFRASTRUCTURE                          │
│   utils/db/          (Low-level DB queries)                  │
│   utils/path_manager (PathManager singleton)                 │
│   utils/image_ops    (Image processing)                      │
│   camera/            (Camera access, ONVIF protocol)         │
│   detectors/         (AI Detection pipeline)                 │
│   config.py          (Global configuration)                  │
└─────────────────────────────────────────────────────────────┘
```

### Dependency Rules

#### ALLOWED Imports

| Module           | May Import From                                        |
|------------------|--------------------------------------------------------|
| `web/*`          | `web/services/*` ONLY (never utils/, camera/, core/)  |
| `web/services/*` | `core/*` ONLY                                          |
| `core/*`         | `utils/`, `camera/`, `detectors/`, `config`            |

#### FORBIDDEN Imports

| From             | To                                                     |
|------------------|--------------------------------------------------------|
| `web/*`          | `utils/db/`, `utils/image_ops`, `camera/`, `detectors/`|
| `web/services/*` | `utils/db/`, `camera/` (must go through core)          |
| `core/*`         | `web/`, `flask`, `werkzeug`                            |

### Example Flows

#### Gallery Route Flow
```
GET /gallery/<date>
    │
    ▼
web/web_interface.py::gallery_page()
    │
    ├── gallery_service.get_detections_for_date(date)
    │       │
    │       └── gallery_core.get_detections_for_date(date)
    │               │
    │               └── utils.db.fetch_detections_for_gallery(conn, date)
    │
    └── render_template("subgallery.html", detections=...)
```

#### Detection Reject Flow
```
POST /api/detections/reject
    │
    ▼
web/blueprints/trash.py::reject_detections()
    │
    ├── detections_service.reject_detections(ids)
    │       │
    │       └── detections_core.reject_detections(ids)
    │               │
    │               ├── utils.db.update_detection_status(conn, id, "rejected")
    │               └── utils.file_gc.move_to_trash(path)
    │
    └── jsonify({"status": "success"})
```

#### ONVIF Discovery Flow
```
GET /api/onvif/discover
    │
    ▼
web/web_interface.py::onvif_discover_route()
    │
    ├── onvif_service.discover_cameras()
    │       │
    │       └── onvif_core.discover_cameras()
    │               │
    │               └── camera.network_scanner.scan_for_onvif()
    │
    └── jsonify({"cameras": [...]})
```

#### Species Summary Flow
```
GET /api/daily-species-summary?date=2026-02-04
    │
    ▼
web/web_interface.py::daily_species_summary_route()
    │
    ├── gallery_service.get_daily_species_summary(date, common_names)
    │       │
    │       └── gallery_core.get_daily_species_summary(date, common_names)
    │               │
    │               └── utils.db.fetch_detection_species_summary(conn, date)
    │
    └── jsonify(summary)
```

### Validation

Run `pytest tests/test_import_boundaries.py` to verify these rules are not violated.

```bash
$ pytest tests/test_import_boundaries.py -v
tests/test_import_boundaries.py::TestWebLayerBoundaries::test_services_only_import_from_core PASSED
tests/test_import_boundaries.py::TestWebLayerBoundaries::test_core_does_not_import_web PASSED
tests/test_import_boundaries.py::TestWebLayerBoundaries::test_count_web_interface_violations PASSED
tests/test_import_boundaries.py::TestModuleStructure::test_core_modules_exist PASSED
tests/test_import_boundaries.py::TestModuleStructure::test_service_modules_exist PASSED
```

### Migration Status (as of 2026-02-04)

| Module | Status |
|--------|--------|
| Gallery routes | ✅ DONE |
| Species routes | ✅ DONE |
| Settings routes | ✅ DONE |
| ONVIF routes | ✅ DONE |
| Analytics routes | ✅ DONE |
| Detection routes | ✅ DONE |
| Backup/Restore routes | ✅ DONE |
| Inbox/Ingest routes | ✅ DONE |
| Review routes | ✅ DONE |
| Trash routes | ✅ DONE |

---

## 8. UI Architecture Contract

The UI components follow a mandatory standard defined in `docs/UI_STANDARD.md`.
This standard is architecturally equivalent to `PathManager`, `image_ops`, and the service layer.

### UI Component Types

#### Modal Types
| Class | Usage | Structure |
|-------|-------|-----------|
| `wm-modal` | Detail/Review modals with image viewer | Header, Body (Image + Info), Action Bar |
| `wm-modal--form` | Settings forms (Add/Edit Camera, ONVIF) | Header, Body (Fields), Action Bar |

#### Tile Types
| Class | Usage | Structure |
|-------|-------|-----------|
| `wm-tile` | Standard tile (Stream, Species, Subgallery) | Button, Media, Image, Badge, Body |
| `wm-tile--review` | Review/Orphans Queue | Select, Badge, Button, Media, Body, Actions |
| `wm-tile--bbox` | Thumbnail with BBox clipping | Select, Media, Image, Body |

#### Viewer & Action Bar
| Class | Usage |
|-------|-------|
| `wm-image-viewer` | Zoomable image viewer in modals |
| `modal-action-bar` | Standardized modal footer with actions |

### Invariants

1. **No Custom Structures:** Templates must **not build** their own modal, tile, or viewer structures.
   Only the classes and DOM structures defined above are allowed.

2. **Single Source of Truth:** `docs/UI_STANDARD.md` is the sole authority for UI component structures.
   Changes to UI patterns require an update to `UI_STANDARD.md` first.

3. **CSS Coupling:** `assets/design-system.css` works **exclusively** with the `wm-*` classes.
   No template-specific CSS rules outside the design system.

4. **Template Compliance:** All templates in `templates/` must use the standard structures.
   Deviations are prohibited and will be rejected in code reviews.

### Architecture Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    TEMPLATE LAYER                            │
│   templates/*.html                                           │
│   templates/partials/*.html                                  │
│                             │                                │
│                             │ MUST USE                       │
│                             ▼                                │
│   ┌─────────────────────────────────────────────────┐       │
│   │            docs/UI_STANDARD.md                   │       │
│   │   ├── wm-modal, wm-modal--form                   │       │
│   │   ├── wm-tile, wm-tile--review, wm-tile--bbox    │       │
│   │   ├── wm-image-viewer                            │       │
│   │   └── modal-action-bar                           │       │
│   └─────────────────────────────────────────────────┘       │
│                             │                                │
│                             │ STYLES                         │
│                             ▼                                │
│   ┌─────────────────────────────────────────────────┐       │
│   │          assets/design-system.css                │       │
│   │   └── .wm-modal*, .wm-tile*, .wm-image-viewer*   │       │
│   └─────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### Change Rules

1. **New UI Pattern:** New UI patterns require:
   - Definition in `docs/UI_STANDARD.md`
   - CSS in `assets/design-system.css`
   - Only then usage in templates

2. **Pattern Modification:** Changes to existing patterns require:
   - Update in `docs/UI_STANDARD.md`
   - Corresponding CSS adjustment
   - Migration of all affected templates

3. **Deprecation:** Old classes must no longer be used:
   - ❌ `species-card`, `gallery-item`, `orphan-tile`, `gallery-tile`
   - ❌ `edit-tile`, `edit-checkbox`, `orphan-checkbox`
   - ❌ `thumbnail-button`, `thumbnail-wrapper`

### Validation

Grep check for forbidden classes:
```bash
grep -rn "species-card\|gallery-item\|orphan-tile\|gallery-tile" templates/
# Expected result: No matches
```

### Migration Status (as of 2026-02-04)

| Component | Old → New | Status |
|-----------|-----------|--------|
| Detection Modal | `modal` → `wm-modal` | ✅ DONE |
| Orphan Modal | `modal` → `wm-modal` | ✅ DONE |
| Settings Modals | `modal` → `wm-modal--form` | ✅ DONE |
| Stream Tiles | `species-card` → `wm-tile` | ✅ DONE |
| Species Tiles | `species-card` → `wm-tile` | ✅ DONE |
| Subgallery Tiles | `gallery-item` → `wm-tile` | ✅ DONE |
| Orphan Tiles | `orphan-tile` → `wm-tile--review` | ✅ DONE |
| Thumbnail Macro | `gallery-tile` → `wm-tile--bbox` | ✅ DONE |
| Edit Tiles | `edit-tile` → `wm-tile--bbox` | ✅ DONE |
| Image Viewer | `modal-image-viewer` → `wm-image-viewer` | ✅ DONE |
