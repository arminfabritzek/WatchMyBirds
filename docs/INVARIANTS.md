# INVARIANTS.md v5

## Rule lifecycle

Rules live in one of three sections, by enforcement strength:

- **HARD** — blocking contracts checked by `tests/test_import_boundaries.py`
  within the scope stated below. Import checks scan Python files recursively,
  including package initializers, local imports, and relative imports. They
  validate static imports, not dynamic imports, transitive dependencies, or
  runtime behavior. HARD rules constrain dependencies and module placement.
- **SOFT** — preferred patterns for new code. Not enforced as facts;
  legacy violations still exist. Monitored non-blocking by
  `tests/test_architecture_soft_monitoring.py`, which emits non-blocking
  snapshots of selected source patterns. It does not compare previous runs
  or establish conformance to every SOFT rule.
- **OBSOLETE** — former rules retired because they over-constrained
  reality. Kept (not deleted) so the reasoning stays on the record.

**Demoting a rule** (HARD→SOFT, SOFT→OBSOLETE) is a deliberate
architectural decision — never an automatic consequence. In
particular, neither "a roadmap endorses a change that this rule blocks"
nor "the enforcement test is firing" is, on its own, a
sufficient reason: a firing test may mean the test is wrong, not the
rule. Demotion requires all of:

1. a deliberate architectural decision to demote;
2. a written rationale recorded with the rule's new section;
3. the corresponding enforcement/monitoring tests updated to match;
4. a `schema_version` bump (the title line, e.g. `v2` → `v3`).

Never silently delete a rule. Move it to OBSOLETE with its reason.

## HARD

### H-01 Web Service Import Boundary

Python modules under `web/services/` may import stdlib, `core`, `config`,
`logging_config`, and other `web.services` modules. Other imports are allowed
only for the file/module pairs below (paths relative to `web/services/`):

| File | Additional allowed modules or symbols | Reason for retaining this dependency here |
|------|---------------------------------------|------------------------------------------|
| `aesthetic_tag_scheduler.py` | `scripts.aesthetic_tag_nightly`, `open_clip`, `torch` | The scheduler invokes the existing worker and probes optional dependencies before starting. Preserve one worker implementation; this exception does not move model inference into the scheduler. |
| `analysis_service.py` | `cv2`, `web.security.safe_log_value` | Retain image decoding at the existing deep-analysis job boundary and reuse the shared log sanitizer. This is a bounded legacy IO exception, not a general image-processing layer. |
| `companion/llama_cpp_adapter.py` | `llama_cpp` | The inference adapter owns lazy SDK loading and model invocation; a core wrapper would duplicate this adapter boundary. |
| `model_registry_service.py` | `yaml` | The registry service parses model metadata for its existing presentation payloads. YAML decoding belongs with that adapter, not in a separate pass-through module. |
| `nightly_jobs/sharpness_job.py` | `cv2`, `utils.image_ops` | The existing job owns crop decoding and calls shared metrics instead of duplicating transformations. Retain this bounded job exception; shared image algorithms remain in `image_ops`. |
| `report_scheduler.py` | `utils.daily_report` | The scheduler invokes the existing report entrypoint. Retain that composition point without duplicating report generation. |
| `telemetry_service.py` | `requests`, `psutil`, `utils.settings` | This opt-in adapter owns payload transport, host measurements, and installation-ID persistence. Keeping that lifecycle together avoids a core module that merely forwards these calls. |
| `update_service.py` | `web.security.safe_log_value` | Reuse one sanitizer for logging the requested update target; this permits no authentication or route dependency. |
| `usb_format_service.py` | `web.security.safe_log_value` | Reuse the same sanitizer for device-target logging; this permits no authentication or route dependency. |

Do not add imports from `camera`, `detectors`, Flask, or Werkzeug. New external
or infrastructure dependencies require a justified, file-scoped adapter/job
exception in this document and its enforcement test. Record why the owning
service needs the dependency and why a domain/core abstraction would not add
meaningful behavior. Shared business rules and infrastructure use cases still
belong in `core`; do not create a core pass-through solely to satisfy H-01.
Retaining a listed legacy dependency does not authorize expanding its role.

**v5 decision:** retain the implemented service composition and make its
exceptions explicit. v4 claimed a strict allowlist but tested only three
forbidden prefixes, omitted subpackages, and missed `from package import name`.
The two previously documented `utils` exceptions remain; the existing nested
sharpness job adds a third. This is a deliberate contract revision, not a
claim that the v4 implementation satisfied its rule. New exceptions are not
automatically accepted merely because an import exists.

### H-02 Core Isolation From Web Framework
DO:
- Keep Python modules under `core/` independent from Flask and web modules.
DO NOT:
- Import `web/*`, `flask`, or `werkzeug` from Python modules under `core/`.

### H-03 Detector Service Isolation
DO:
- Keep Python modules under `detectors/services/` independent from web modules.
DO NOT:
- Import `web/*`, `flask`, or `werkzeug` from Python modules under `detectors/services/`.

### H-04 Detector Service Internal Dependency Rule

Allow only these directed imports between modules under `detectors/services/`:

| Source | Allowed targets |
|--------|-----------------|
| `persistence_service` | `crop_service` |
| `capability_registry` | `decision_policy_service`, `temporal_decision_service` |
| `scoring_pipeline` | `bbox_quality_service`, `capability_registry`, `decision_policy_service`, `temporal_decision_service` |

The root `detectors/services/__init__.py` may re-export services. Service
implementations import concrete modules rather than consuming package
re-exports. Every other internal edge, including a new module or a reverse
edge, fails the check. New edges require updating this contract and its test.

**v5 decision:** retain the scoring pipeline and capability-version registry
as explicit composition points. v4 allowed only persistence → crop, while its
test checked a fixed list of six service names and missed newer targets.
The new allowed graph is acyclic; the test now examines every service module,
including nested modules and relative imports, without a fixed target list.

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
