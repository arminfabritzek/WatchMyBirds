# INVARIANTS.md v2

## HARD

### H-01 Web Service Import Boundary
DO:
- Keep imports in `web/services/*.py` limited to:
  - `core/*` (business logic layer)
  - Python stdlib / `typing` modules
  - `config`, `logging_config` (project infrastructure)
  - `utils.*` (shared utilities)
  - `web.services.*` (intra-package imports, e.g. `db_service`)
DO NOT:
- Import `camera/*` or `detectors/*` from `web/services/*.py`.

### H-02 Core Isolation From Web Framework
DO:
- Keep `core/*.py` independent from Flask and web modules.
DO NOT:
- Import `web/*`, `flask`, or `werkzeug` from `core/*.py`.

### H-03 Detector Service Isolation
DO:
- Keep `detectors/services/*.py` independent from web modules.
DO NOT:
- Import `web/*`, `flask`, or `werkzeug` from `detectors/services/*.py`.

### H-04 Detector Service Internal Dependency Rule
DO:
- Allow only `detectors/services/persistence_service.py` to import `detectors/services/crop_service.py`.
DO NOT:
- Add any other direct import between files in `detectors/services/*.py`.

### H-05 Required Module Set
DO:
- Keep these files present in `core/`: `gallery_core.py`, `settings_core.py`, `onvif_core.py`, `analytics_core.py`, `detections_core.py`.
- Keep these files present in `web/services/`: `gallery_service.py`, `settings_service.py`, `onvif_service.py`, `analytics_service.py`, `detections_service.py`.
- Keep these files present in `detectors/services/`: `persistence_service.py`, `crop_service.py`, `classification_service.py`, `detection_service.py`, `notification_service.py`.

## SOFT

### S-01 Route Thinness
DO:
- Keep blueprint handlers and `web/web_interface.py` routes focused on request parsing, service calls, and HTTP response mapping.
DO NOT:
- Add new business rules, SQL statements, or file-processing pipelines directly in route handlers.

### S-02 Service Responsibility
DO:
- Put use-case logic in dedicated service modules.
DO NOT:
- Use `web/services/db_service.py` as a pass-through for route-level SQL orchestration in new code.

### S-03 Runtime State Ownership
DO:
- Prefer explicit, injectable stateful services for background work and progress tracking.
DO NOT:
- Introduce additional module-level mutable globals in web blueprints.

### S-04 IO Placement
DO:
- Route OS commands, filesystem workflows, and system metrics collection through dedicated service modules.
DO NOT:
- Add new direct `subprocess`, large file IO workflows, or hardware metric collection in route handlers.

### S-05 Path and Image Operation Centralization
DO:
- Use `PathManager` and shared image utility modules for new storage/image operations.
DO NOT:
- Add new manual storage path construction or duplicate image transformation logic in web handlers.

## OBSOLETE

### O-01 Pure Domain Core
DO NOT:
- Treat `core/*` as a pure domain/model layer with zero IO.

### O-02 Strict UI Orchestration-Only Rule
DO NOT:
- Use "UI/API layer contains no logic at all" as an acceptance criterion for the current repository state.

### O-03 Exclusive PathManager Authority
DO NOT:
- Use "all storage paths are already exclusively resolved via PathManager" as a factual invariant.

### O-04 Single Image-Ops Authority Already Enforced
DO NOT:
- Use "all image manipulation already flows through `utils.image_ops`" as a factual invariant.
