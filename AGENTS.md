# CLAUDE.md — WatchMyBirds

Operating notes for AI coding agents working in this repository.
Humans should read [`CONTRIBUTING.md`](CONTRIBUTING.md) instead.

## What this project is

WatchMyBirds is a **single-station bird-detection appliance**:
a multi-stage AI pipeline (object detection → species classification)
behind a Flask + Jinja2 web UI, deployed on Raspberry Pi 5 or via
Docker. Originals are immutable, the SQLite DB is the metadata
authority, new storage-path construction goes through
`utils.path_manager.PathManager`.

The architectural narrative lives in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md);
the enforced invariants live in [`docs/INVARIANTS.md`](docs/INVARIANTS.md).
**Read both before any non-trivial change.**

## Environment

- **Python:** 3.12+ (pinned in `pyproject.toml`, also pinned on the
  RPi build lane — do not bump unilaterally).
- **Virtualenv:** prefer `.venv/` in the repo root when present.
- **Dependencies:** `requirements.txt` is authoritative for runtime;
  `requirements-aesthetic.txt` and `requirements-companion.txt` are
  optional extras.
- **Entry point:** `python main.py` → web UI on `http://localhost:8050`.

## Tooling

All tool configuration is centralized in `pyproject.toml`. There are
no separate `.flake8` / `.isort.cfg` / `pytest.ini` files.

| Task    | Command                  | Notes                                |
|---------|--------------------------|--------------------------------------|
| Lint    | `ruff check .`           | Autofix: `ruff check --fix .`        |
| Format  | `ruff format .`          | Black is deprecated, prefer `ruff`.  |
| Test    | `pytest`                 | `testpaths = ["tests"]`.             |

## Binding documents (read before touching the listed surfaces)

Authority order when they disagree: **`INVARIANTS.md` > `ARCHITECTURE.md` > everything else.** `INVARIANTS.md` is schema-versioned and tracks what is actually enforced; the others are narrative.

- [`docs/INVARIANTS.md`](docs/INVARIANTS.md) — schema-versioned (`v2`)
  list of HARD / SOFT / OBSOLETE invariants. The canonical answer to
  "is this rule actually enforced?" Read first for any cross-cutting
  change.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — narrative architecture:
  data model, module responsibilities, change rules. Context for any
  change in `detectors/`, `core/`, `web/`, `utils/path_manager.py`,
  `utils/image_ops.py`, `utils/file_gc.py`.
- [`docs/UI_STANDARD.md`](docs/UI_STANDARD.md) — image UX,
  hover-tooltip convention, modal/review/stream rules. **Sections 0,
  0a, and 0c are marked `(binding)`.** Read before any template /
  frontend-JS / CSS change.
- [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) — two-layer config
  model: boot ENV vars (read-only at runtime) + `OUTPUT_DIR/settings.yaml`
  (live-editable). Merge order: defaults → env → settings.yaml.
- [`docs/PRIVACY.md`](docs/PRIVACY.md) — privacy posture (local-only
  by default, opt-in heartbeat only).

## Hard rules (HARD invariants from `INVARIANTS.md v2`)

These are the rules whose violations are detected by tests. If you
break one, CI catches you.

- **`H-01` Web service import boundary.** `web/services/*.py` may
  import from `core/*`, stdlib/typing, `config`, `logging_config`,
  `utils.*`, and other `web.services.*` — **but never from `camera/*`
  or `detectors/*`**.
- **`H-02` Core isolation.** `core/*.py` must not import `web/*`,
  `flask`, or `werkzeug`.
- **`H-03` Detector service isolation.** `detectors/services/*.py`
  must not import `web/*`, `flask`, or `werkzeug`.
- **`H-04` Detector service internal coupling.** Only
  `detectors/services/persistence_service.py` may import
  `detectors/services/crop_service.py`. No other inter-file imports
  inside `detectors/services/`.
- **`H-05` Required module set.** Do not delete or rename the
  required files in `core/`, `web/services/`, and `detectors/services/`
  (see `INVARIANTS.md` for the exact list).

Project-level rules that are equally non-negotiable but not in the
invariants test suite:

- **Originals are immutable.** Files under `OUTPUT_DIR/originals/`
  are written once and never modified in place. Derivatives
  (`OUTPUT_DIR/derivatives/…`) are disposable and regenerable.
- **Database is the metadata authority.** No absolute filesystem
  paths in the DB — only filenames and relative references resolved
  at runtime.
- **No Dash.** The legacy Dash UI is deprecated. All new routes are
  Flask + Jinja2 (blueprints under `web/blueprints/` or routes on
  `web/web_interface.py`).
- **Deletion order.** Hard delete: attempt file removal first,
  then DB record removal. Missing-file MUST NOT abort the DB removal.

## Soft rules (SOFT invariants — preferred patterns for new code)

These are not test-enforced as facts about the whole codebase. Legacy
call sites that violate them still exist. **Do not extend that pile;
new code follows the rule.**

- **`S-01` Route thinness.** Blueprint handlers and routes parse the
  request, call a service, return a response. No business rules, no
  raw SQL, no file-processing pipelines inline.
- **`S-02` Service responsibility.** Use-case logic lives in dedicated
  service modules, not in `web/services/db_service.py` as a SQL
  pass-through.
- **`S-03` Runtime state ownership.** Prefer injectable stateful
  services over module-level mutable globals in blueprints.
- **`S-04` IO placement.** Subprocess calls, large file IO, hardware
  metrics — through dedicated services, not in route handlers.
- **`S-05` Path and image-op centralization.** New storage-path
  construction goes through `utils.path_manager.PathManager`. New
  image transformations go through `utils.image_ops`. (`O-03` and
  `O-04` in `INVARIANTS.md` retired the claim that this is *already*
  exclusive — it is the rule for new code, not the state of legacy.)

## Style and changes

- **Conventional Commits** for messages: `feat:`, `fix:`, `docs:`,
  `refactor:`, `test:`, `chore:`. Subject ≤ 72 chars, imperative.
- **Type hints** required on new code (Python 3.12 syntax: `list[T]`,
  `X | None`, `match`).
- **Tests** belong in `tests/`. Add a test when the change touches
  detection, persistence, path resolution, deletion, or auth.
- **Comments:** only when the *why* is non-obvious. Don't narrate
  what well-named code already says.
- **No unsolicited refactors.** A bug fix fixes the bug; cleanup is
  a separate change. Don't bundle.

## Localization

User-facing labels, status text, helper copy, and new UI-near
documentation default to **English**. Bird species names and other
locale-derived content stay localized where they belong. Comments
and identifiers are English.

## Deploy targets (informational)

- **Docker:** `docker-compose.example.yml` → standard Docker host.
- **Raspberry Pi 5:** the primary appliance target. Pre-built images
  exist; see `rpi/` for build pipeline.

Do not write production code that assumes x86 SIMD, CUDA, or
glibc-only behavior — RPi (aarch64, musl-adjacent on some builds) is
a first-class target.

## Out of scope (do not propose without explicit ask)

- Migrating the gallery to a SPA framework (React/Vue/etc.).
- Replacing SQLite with a server DB.
- Cloud storage as the primary tier.
- Reintroducing Dash.
- Adding a "manual labeling" workflow to the Review queue (Review
  corrects, it doesn't enter labels from scratch).

## Where things live

| Area                            | Path                                  |
|---------------------------------|---------------------------------------|
| Web routes & templates          | `web/`, `templates/`                  |
| Detection / classification      | `detectors/`                          |
| Core business logic & DB        | `core/`                               |
| Path / image / GC utilities     | `utils/`                              |
| Ingest pipeline                 | `ingest/`                             |
| Camera & streaming              | `camera/`                             |
| Analytics                       | `analytics/`                          |
| RPi build & deploy              | `rpi/`, `systemd/`                    |
| Operational scripts             | `scripts/`, `infra/`                  |
| Tests                           | `tests/`                              |
| User docs                       | `docs/`                               |
| Optional Companion (LLM) layer  | `berta_openai/`                       |

## Reporting back

After a change: state what changed, which invariants were touched,
and what tests were added or run. Don't write a summary document —
the diff and the commit message are the record.
