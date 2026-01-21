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
*   **MUST:** Orchestrate the AI pipeline (Frame -> Detect -> Crop -> Classify -> Save).
*   **MUST:** Use `image_ops` for cropping and `path_manager` for saving.
*   **MUST NOT:** Implement custom image processing or path logic.

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
