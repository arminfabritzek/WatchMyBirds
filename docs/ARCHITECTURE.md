# Architecture Documentation

## 1. Overview
WatchMyBirds is an AI-powered bird detection system that provides real-time video analysis and a server-side rendered web interface. It relies on a high-performance Flask + Jinja2 architecture for the core UI and administration. Client-side components are used strictly for specific complex interactions and are not the primary rendering model. The system emphasizes low-latency MJPEG streaming, authority-based metadata, and immutable file management.

## 2. Core Architectural Invariants
*   **Originals are Immutable:** After initial capture/import finalization (including initial EXIF), stored originals must not be overwritten or edited for review corrections. Initial ingestion may encode the incoming image; this is not a promise of byte identity with an uploaded source. Retention and explicit hard deletion are separate removal workflows.
*   **Derivatives and Retention:** Optimized images and thumbnails can be regenerated only while their source remains available. `core/retention_core.py` can retire originals while preserving derivatives and metadata. Do not treat those surviving derivatives as disposable caches.
*   **Database as Metadata Authority:** The SQLite database is the sole authority for metadata. It contains *NO* absolute filesystem paths, only filenames and relative references resolved dynamically.
*   **PathManager as the storage-path API:** New storage-path construction SHOULD go through `utils.path_manager.PathManager`. This is specified as a SOFT invariant (`S-05` in `INVARIANTS.md`), not as a fact about the current codebase — some legacy call sites still build paths manually. New code must not add to that pile; refactors that remove a manual path are welcome.
*   **`utils.image_ops` as the image-op API:** Shared image transformations (crop, pad) live in `utils.image_ops`. Same status as PathManager: required for new code, partial in legacy (`O-04` in `INVARIANTS.md` retired the "already-enforced" claim).
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
    *   `label_subjects` / `human_label_facts`: Explicit human assertions, separate from model outputs.
    *   `manual_objects` / `manual_object_revisions`: Manually added birds and their revision history.
    *   `station_event_reviews`: Explicit biological-event review decisions.

## 4. Key Modules and Responsibilities
### `utils/path_manager.py`
*   **MUST:** Be the canonical API for new storage-path resolution (`S-05` SOFT in `INVARIANTS.md`).
*   **MUST:** Handle date-based directory structures (`YYYY-MM-DD`).
*   **Responsibility:** Resolve storage paths and ensure directories where the API promises it. Image decoding, transformation, and persistence belong to image utilities and use-case services.
*   **Containment:** Getters such as `get_original_path` construct paths without guaranteeing containment. For untrusted filenames, pass the candidate to `contained_path(candidate, root)`, reject `None`, and use the returned resolved path for IO. Calling a getter alone is not a security boundary.
*   **Contract correction:** The former blanket ban on filesystem I/O did not describe this API: `ensure_date_structure` and several path getters create directories. Directory creation remains part of these methods; this does not authorize arbitrary persistence logic in `PathManager`.
*   **NOTE:** "All storage paths are already exclusively resolved via PathManager" is no longer a factual invariant — see `O-03 OBSOLETE` in `INVARIANTS.md`. Legacy call sites still construct paths manually; do not extend that pattern.

### `utils/image_ops.py`
*   **MUST:** Be the canonical home for shared image transformations (cropping, padding) in new code.
*   **Responsibility:** Keep in-memory transformations functional. Explicit file helpers such as `generate_preview_thumbnail` also read/write images and create their output directory; the module as a whole is not pure.
*   **NOTE:** "All image manipulation already flows through `utils.image_ops`" is no longer a factual invariant — see `O-04 OBSOLETE` in `INVARIANTS.md`. Add new transforms here; don't duplicate inline.

### `detectors/detection_manager.py`
*   **Target:** Orchestrate capture, detection, classification, scoring, persistence, and notifications through specialized services.
*   **Current composition:** The manager constructs `ImageClassifier`, wraps the selected backend in `ClassificationService`, initializes capture and database context, and coordinates the scoring pipeline. Classifier construction is not a forbidden inference path; classification work belongs in the service.
*   **Guarded boundary:** Direct imports of `utils.image_ops`, `utils.telegram_notifier`, and `piexif` are rejected by `test_detection_manager_delegates_image_and_notification_helpers`.
*   **Implementation debt:** The manager still contains direct DB and image-processing work. Do not describe it as an already-thin orchestrator or extend those responsibilities for new features.

### `utils/file_gc.py`
*   **MUST:** Handle safe deletion of files and database records.
*   **MUST:** Operate exclusively on ABSOLUTE paths resolved via `PathManager`.
*   **MUST:** Ensure referential integrity (don't delete shared files if used elsewhere).

### `core/recovery_core.py`
*   **MUST:** Be the single recovery engine for the CLI and guided Pi runner.
*   **MUST:** Validate and stage complete output state before atomic publication.
*   **MUST:** Flush staged files, directories, journal updates, and rename parents before advancing recovery phases.
*   **MUST:** Retain a complete destination checkpoint and reconcile interrupted swaps before the app starts.
*   **MUST NOT:** Manage services or accept web-framework objects.

### `scripts/recovery_runner.py`
*   **MUST:** Accept only a discovered snapshot identifier and the fixed Pi destination.
*   **MUST:** Serialize maintenance, stop database users, restart the app, and verify health.
*   **MUST:** Keep independent token-authenticated progress, retry, and rollback available while Flask is stopped.
*   **MUST:** Refuse rollback if `app.service` cannot be stopped.
*   **MUST NOT:** Restore application binaries or accept arbitrary commands, paths, ports, or unit names.

### `web/web_interface.py`
*   **MUST:** Serve the web UI via Flask routes.
*   **Target:** Resolve stored media through `PathManager` or its service wrappers. Existing route-level IO is legacy debt under S-04/S-05, not proof of completed migration.
*   **MUST NOT:** Contain legacy Dash callbacks or layout logic.

## 5. Change Rules
*   **Storage Path Changes:** If the filesystem structure changes, `PathManager` MUST be updated. Call sites that go through it inherit the change; call sites that still build paths manually must be migrated as part of the same change.
*   **New Routes:** All new web routes MUST be implemented in Flask (`server.route` or a blueprint).
*   **Path Construction in new code:** Do not use `os.path.join` to build storage paths in new code. Use `path_manager`. (Legacy manual-path call sites exist; do not extend them.)
*   **Cross-Cutting Impact:** Any change affecting storage layout, deletion logic, or image processing MUST trigger a simultaneous review of `PathManager`, `detection_manager`, `file_gc`, and `web_interface` to ensure consistency.
*   **Authority hierarchy:** When this document and `INVARIANTS.md` disagree, `INVARIANTS.md` wins. It is schema-versioned (see its title for the current version) and tracks what is actually enforced; this document is the narrative companion.

## 6. Non-Goals
*   **Client-Side Rendering (CSR):** The core gallery is Server-Side Rendered (SSR). We do not aim to move the app to a SPA framework.
*   **Cloud Storage:** The system is designed for local filesystem storage (NAS/Disk). Cloud sync is an external concern.
*   **Dash UI:** Dash-based UI components are intentionally deprecated. Flask/Jinja2 is the only supported UI layer.

---

## 7. Service Layer Architecture

The preferred dependency direction is routes → web services → core →
infrastructure. H-01–H-05 in `INVARIANTS.md` define the blocking boundaries;
S-01–S-05 define the preferred patterns for new work.

| Layer | Responsibility and dependency policy |
|-------|--------------------------------------|
| `web/blueprints/`, `web/web_interface.py` | HTTP, authentication, response mapping and bootstrap. New use cases call services; existing direct SQL/IO is monitored debt. |
| `web/services/` | Web use cases, adapters and background jobs; H-01 permits core/infrastructure imports plus explicitly listed file-scoped exceptions. |
| `core/` | Business rules, DB and infrastructure coordination; no web-framework imports (H-02). Core is not a pure no-IO domain layer. |
| `detectors/services/` | Detection pipeline services; no web imports (H-03), only documented internal edges (H-04). |

The former strict `web → services ONLY` / `services → core ONLY` table and
completed migration checklist overstated the implemented boundaries. They
must not be used as evidence that legacy route logic has been removed.

### Representative flows

- Gallery data: `web.services.gallery_service.get_detections_for_date` →
  `core.gallery_core.get_detections_for_date` → SQLite queries.
- Trash: `web.blueprints.trash.reject_detection` currently calls
  `db_service.reject_detections` → `utils.db.detections.reject_detections`.
  This changes detection/classification status; it does not move image files
  or assert a training label. The service pass-through is S-02 debt.
- Hard deletion: core/service wrappers → `utils.file_gc`. Referenced files
  are attempted first, then DB rows; missing files do not stop DB cleanup.
- Corrections: `web.blueprints.human_labels` → `human_label_service` →
  core human-label/manual-object logic. Model proposals, manual objects,
  image-wide facts, and biological event review are distinct evidence.

### Validation and limits

Run `python -m pytest tests/test_import_boundaries.py -m arch_hard` for
blocking import/module checks, and
`python -m pytest -s tests/test_architecture_soft_monitoring.py` for current
route-pattern counts. These are non-blocking snapshots, not comparisons with
a baseline; optional report output only appends counts. The monitor never
fails on a metric value. Other tests in
`test_import_boundaries.py` cover template inheritance and specific manager
imports; none establish full UI, persistence, or architectural correctness.

### Persistence gap

`utils.ingest.ingest_file` derives detection-bearing image filenames from
EXIF time with one-second precision, then writes to that original path.
Hash deduplication prevents identical inputs but does not prevent different
images with the same timestamp from sharing a destination. The immutability
contract remains in force; collision-safe publication is an implementation
gap, not permission to overwrite. Live capture uses microsecond filenames,
but its writer also does not enforce exclusive creation.

## 8. UI Architecture Contract

`docs/UI_STANDARD.md` owns the current component and interaction contract.
Reuse the shared viewer, action vocabulary, toolboxes and bird editor rather
than copying their semantics into individual pages.

Canonical implementations:

- `templates/components/modal_image_viewer.html`: `render_image_viewer`.
- `templates/partials/tile_toolbox.html`: grid/image action toolbox.
- `templates/components/bird_editor_toolbar.html`: shared modal editor rail.
- `templates/components/detection_modal.html`: detection detail shell.
- `templates/components/review_grid_card.html`: current Review card composition.
- `assets/design-system.css`: shared styling and tokens, including component
  families beyond `wm-*` (for example `review-grid-*`).

Update `UI_STANDARD.md` alongside a shared pattern change. Its documented
surface-specific compositions and exceptions are part of the contract;
`wm-tile` is not a universal description of every present-day card. Historical
migration checkmarks and substring searches for class names are not UI
acceptance tests. Validate affected flows in a browser as well as with their
relevant template/JS tests.
