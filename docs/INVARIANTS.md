# System Invariants

## 1. Storage & Filesystem
*   Original images (`originals/`) MUST BE immutable; they MUST NOT be modified in place after initial write.
*   Derivative images (`derivatives/`) MUST BE treated as disposable caches; they MUST be reproducible from originals at any time.
*   Originals and derivatives MUST share the same base filename structure.
*   All image storage MUST follow the `YYYY-MM-DD` directory structure based on the image capture date.

## 2. Path Resolution
*   `utils.path_manager.PathManager` MUST BE the distinct and exclusive authority for resolving all filesystem paths.
*   Application code MUST NOT manually construct filesystem paths using string concatenation or `os.path.join` for storage locations.
*   All filesystem operations MUST use paths resolved by `PathManager`.

## 3. Database
*   The database tables MUST NOT store absolute or relative filesystem paths.
*   The `filename` column in the `images` table MUST BE the unique identifier for linking database records to files.
*   Database records MUST refer to files strictly by their filename.

## 4. Image Processing
*   `utils.image_ops` MUST BE the single canonical source for all shared image manipulation logic.
*   Detectors, Ingest, and Web modules MUST NOT implement their own image manipulation logic; they MUST import from `utils.image_ops`.

## 5. Deletion & Cleanup
*   Hard deletion operations MUST attempt to remove files from the filesystem *before* removing corresponding database records.
*   Deletion logic MUST operate exclusively on **Absolute Paths** resolved via `PathManager`.
*   Deletion operations MUST BE idempotent; a missing file MUST NOT cause the database deletion to fail.
*   Files MUST NOT be deleted unless they are verified to be within the configured `OUTPUT_DIR`.

## 6. UI Architecture
*   Flask with Jinja2 MUST BE the only supported UI rendering technology for core features.
*   Legacy Dash components and routes MUST remain deprecated and non-functional.
*   New UI features MUST BE implemented using Flask routes and templates.

## 7. Cross-Impact Governance (HARD)
*   If a change affects cross-cutting contracts or operations (for example ports, firewall, service names, env vars, API shape, storage paths, or build/deploy behavior), all required linked updates MUST BE completed in the same task.
*   If immediate completion is not possible, an explicit delegation entry MUST BE added to `docs/IMPACT_LEDGER.md` before merge.
*   A task with unresolved cross-impact follow-ups MUST NOT be marked complete without either complete linked updates or a ledger delegation entry.
