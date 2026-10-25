# Architecture Documentation

## 1. Overview
WatchMyBirds is an AI-powered bird detection system that provides real-time video analysis and a server-side rendered web interface. It relies on a high-performance Flask + Jinja2 architecture for the core UI and administration. Client-side components are used strictly for specific complex interactions and are not the primary rendering model. The system emphasizes low-latency MJPEG streaming, authority-based metadata, and immutable file management.

## 2. Core Architectural Invariants
*   **Originals are Immutable:** After initial capture/import finalization (including initial EXIF), stored originals must not be overwritten or edited for review corrections. Initial ingestion may encode the incoming image; this is not a promise of byte identity with an uploaded source. Retention and explicit hard deletion are separate removal workflows.
*   **Derivatives and Retention:** Optimized images and thumbnails can be regenerated only while their source remains available. `core/deletion_plan.py` decides which files and rows expire. Bird previews survive with Keep previews; Delete everything expires them later. Answers and admitted bird rows survive file expiry. Do not treat retained derivatives as disposable caches.
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
        Rank 1 is the downstream selected label, which may be geographically
        weighted; its confidence and the four alternatives retain image-model
        probabilities. `detections.raw_species_name` and optional
        `detections.geography_json` preserve the raw top choice, complete raw
        candidate order, geographic ranking and model provenance separately.
        Geographic scoring never consumes or writes human label facts.
    *   `label_subjects` / `human_label_facts`: Explicit human assertions, separate from model outputs.
    *   `manual_objects` / `manual_object_revisions`: Manually added birds and their revision history.
    *   `bird_add_requests`: Idempotent add-or-correct receipts; geometric matches require an explicit identity choice before writing.
    *   `station_event_reviews` / `station_event_review_members`: Unused legacy event assessments, retained without readers or writers.

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
- Trash: `web.blueprints.trash.move_to_trash` calls
  `detections_service.move_to_trash` → `core.detections_core.move_to_trash`.
  This changes detection/classification status; it does not move image files
  or assert a training label. It stamps `detections.trashed_at` in the same transaction.
- Hard deletion: core/service wrappers → `utils.file_gc`. Referenced files
  are attempted first, then DB rows; missing files do not stop DB cleanup.
- Corrections: `web.blueprints.human_labels` → `human_label_service` →
  core human-label/manual-object logic. Model proposals, manual objects,
  and image-wide facts remain distinct evidence.

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

Live capture uses microsecond filenames, but its writer does not enforce
exclusive creation. Collision-safe publication remains an implementation gap,
not permission to overwrite an existing original.

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

## Training export

`core/canonical_dataset.py` derives the sole training bundle from current human
facts and active manual objects. Stable `detection:<id>` and `manual:<id>` keys
identify birds once per training view. Species crops are independent of unresolved
neighbors; COCO contains only frames whose objects are all resolved, including
manually drawn boxes. Historical facts and manual revisions remain audit evidence.
Manual records never acquire synthetic model predictions. Bundle/rule version 3
uses sample IDs `<labeling_installation_id>:<24-hex local sample hash>` for both
object and image samples. Object absence suppresses an older species answer in
the training object; raw historical facts remain audit evidence. Unknown manual
species can supply a box but cannot supply a species label. Missing originals
exclude training use.

`core/training_export.py` records each export's immutable bundle metadata, creation
time and exact answer revisions in `labeling_exports` / `labeling_export_members`.
The incompatible legacy `training_exports` table stays untouched and unused;
its rows never count as current exports or receipts. Backups preserve it as data,
while merge imports export evidence only from the new tables.
The builder can select detection, manual-object or image-statement keys; selected
photos include all their objects and statements. Current asserted revisions with
an original wait until included in an export. Changed answers wait again.
`core/training_retention.py` protects waiting photos and photos in any unconfirmed
export, including unchanged context siblings. Historical facts and legacy flags
alone grant no training protection. Favorites remain a separate retention rule.
An idempotent receipt stores `received_at` for that export only; it is the reference
time for deletion deadlines. `core/retention_core.py` executes due planner actions,
rechecking each photo under `BEGIN IMMEDIATE` before files and then rows.
Setting changes serialize with that transaction. Export records have no foreign
keys to images, detections or label subjects and survive their deletion. The
unused legacy `training_exports` table is never explicitly written or altered;
its existing detection cascade can remove rows when a detection is fully deleted.

ZIP responses stream media in 64 KiB chunks without a temporary archive. Saved
metadata rebuilds the same bytes after later edits; originals remain immutable.
Metadata memory scales with the selected photo context, not media byte volume.
Missing originals make a repeat download fail explicitly. SQLite backup includes
both tables; USB merge maps membership IDs and media locations while preserving
the original snapshot and sample IDs.

## Storage diagnostics

`core/storage_scan.py` owns one manually started scan per Flask application and
persists the latest report plus at most 32 measurements via PathManager. Status
requests read that bounded state; they never enumerate media or rebuild analytics.
`core/storage_accounting.py` streams known application storage roots and reports
logical file sizes separately from allocated blocks and filesystem capacity.
Missing optional scopes, access failures and live changes remain visible. Inactive
Linux automounts are detected from the kernel mount table and left unmounted.

SQLite diagnostics use a read-only transaction. Connection-local TEMP views limit
the unchanged canonical exporter to batches of 200 complete images; they never
modify source tables. Protection attribution reuses the existing training predicate
and retention decision, with overlapping reasons and a distinct protected union.
Readiness does not release training protection. The filesystem scan is not atomic;
concurrent files/DB changes make reclaim estimates provisional. Training export receipts are recorded separately from temporary-file diagnostics.

`core/storage_pressure.py` gates image persistence using a two-second filesystem
cache and subtracts a conservative per-frame budget between checks. Recovery runs
outside capture through that same planner and executor: open/model non-bird files
and rows first, then unprotected answered/admitted originals, then bird previews
only with Delete everything. Every pass rechecks all objects and current settings
under the write lock. Favorites and pending training files stay protected; normal
trash deadlines still apply. Low space can shorten the normal Review grace period.
The photo switch gates early deletion too. While off, or when reclamation cannot
restore the reserve and frame headroom, new image/detection persistence pauses.
Live frames and inference continue, but these unsaved frames produce no ticker,
Telegram detection alert, MQTT event or history row. Already queued alerts and
reports about stored observations can still send. Saving resumes automatically
after free space recovers. Authenticated pages show the cached saving/cleanup status.

Classifier rejects persist neither photos nor notes. The unused `reject_audit`
table remains for schema compatibility; a transactional one-time migration clears
its former rows. `core/technical_storage.py` bounds runtime/weather samples and
disposable files independently of the photo switch. Application and aesthetic logs
rotate; vital-sign CSV files rotate separately. Human answers, export records and
admitted observations have no technical-data expiry.
