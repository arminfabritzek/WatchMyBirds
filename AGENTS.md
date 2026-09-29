# AGENTS.md — WatchMyBirds

WatchMyBirds is a single-station bird-detection appliance: object detection →
species classification, with Flask/Jinja2, SQLite, and local image storage.
Raspberry Pi 5 (aarch64, CPU) and Docker are first-class targets.
Human contributors: [CONTRIBUTING.md](CONTRIBUTING.md).

## Working agreement

- Check `git status --short` and the relevant diff first. Preserve existing
  work; do not reset, stash, or overwrite unrelated changes.
- Keep a short `[ ]` / `[x]` checklist in the conversation. Carry authorized
  local implementation and verification through to a reviewable result.
- Ask when a missing decision materially changes scope. Deployment, live
  camera or production-data changes, pushing, and publishing need action-specific
  authorization; existing authorization in the conversation counts.
- When contributors work concurrently, agree file ownership and re-read shared
  files before editing. Integrate their changes instead of reverting them.
- Keep fixes focused. Do not bundle unsolicited refactors.

## Read for the affected surface

| Change | Read before editing |
|--------|---------------------|
| Service boundaries, storage, image lifecycle, deletion, architecture tests | [INVARIANTS.md](docs/INVARIANTS.md) and [ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Templates, frontend JS or CSS | Binding sections and affected component in [UI_STANDARD.md](docs/UI_STANDARD.md) |
| Settings or config loading | [CONFIGURATION.md](docs/CONFIGURATION.md) |
| Outbound data flows | [PRIVACY.md](docs/PRIVACY.md) |

For other edits, read the relevant implementation, tests, and supporting docs;
a documentation-only edit needs the sources for its claims.

For architecture rules, authority is **INVARIANTS.md > ARCHITECTURE.md > other
project documents**. Report code/test/doc disagreements instead of silently
weakening a rule. Rule demotions require the rationale, test update, and version
bump described in INVARIANTS.md. Update UI_STANDARD.md in the same change when
a shared UI rule changes.

## Contracts to preserve

The full import policy, exceptions, and required module list live in
INVARIANTS.md. Quick map:

- **H-01:** Web services use the documented import allowlist and file-scoped
  exceptions. Nearby legacy imports do not authorize new exceptions.
- **H-02 / H-03:** Core and detector services do not import web, Flask, or Werkzeug.
- **H-04:** Detector-service imports follow the explicit directed dependency table.
- **H-05:** Preserve the required modules.

These checks cover static imports and module presence, not all runtime behavior.
S-01–S-05 guide new work: thin routes, dedicated use-case services, injectable
state, service-owned IO, `PathManager` for storage paths, and `utils.image_ops`
for shared transformations. Legacy deviations are not permission to add more.

- Preserve originals after capture/import finalization. Never overwrite an
  existing original or edit it for a correction. Retention may remove originals;
  preserve derivatives for retained observations that no longer have originals.
- SQLite is the metadata authority. Store filenames/relative references, not
  absolute filesystem paths. Path getters alone do not prove containment;
  use the checked path described in ARCHITECTURE.md for untrusted filenames.
- Hard deletion attempts file removal before DB removal. Missing files must
  not abort DB cleanup. See ARCHITECTURE.md for known persistence gaps.
- Use Flask/Jinja2 for new UI. SPA migration, a server DB, cloud primary storage,
  and reintroducing Dash are out of scope without an explicit request.
- General blank-canvas annotation is out of scope. Existing `Add missing bird`
  on a stored image with an available original is the explicit exception
  (UI_STANDARD.md §0d). Manual objects must not fabricate model scores,
  detector proposals, or event approval. Offered-box corrections remain in scope.

## Environment and validation

- Python minimum is 3.12 (`pyproject.toml`); CI/RPi use 3.12. Docker pins its own
  runtime in `Dockerfile`. Do not bump runtimes unilaterally or require CUDA/x86 SIMD.
- Prefer the repo `.venv/`. Start the app with `python main.py` (default port 8050).
- Runtime dependencies: `requirements.txt` plus `requirements-aesthetic.txt`.
  The latter isolates the torch/open_clip CPU index, is included in standard
  builds, and is default-ON. Opt out with `AESTHETIC_TAG_ENABLED=False`, not by
  omitting its dependencies. `requirements-companion.txt` is an optional extra
  for the default-OFF Companion.
- Python tool configuration lives in `pyproject.toml`. Run from the repo root;
  use `.venv/bin/python -m` when available. `<files>` below is a placeholder.

| Check | Command |
|-------|---------|
| Lint affected Python | `python -m ruff check <files>` |
| Check / apply formatting | `python -m ruff format --check <files>` / `python -m ruff format <files>` |
| Targeted tests | `python -m pytest <files>` |
| Hard architecture checks | `python -m pytest tests/test_import_boundaries.py -m arch_hard` |
| CI-equivalent lint/test selection | `python -m ruff check .` then `python -m pytest -q -m "not host_env"` |

Use targeted checks during iteration. Broaden for cross-cutting changes or PR
preparation; `host_env` tests need a suitable host. Do not mass-format unrelated
files. For documentation-only changes, check claims, links, and `git diff --check`;
a full suite is unnecessary. Add tests for detection, persistence, path resolution,
deletion, and auth changes; tests belong in `tests/`.

## Style and language

- Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`),
  imperative subject, at most 72 characters.
- New Python functions/methods require type hints (`list[T]`, `X | None`).
- Comments explain non-obvious reasons; do not narrate well-named code.
- UI labels, helper/status text, UI-near docs, comments, and identifiers default
  to English. Species names and other locale-derived content remain localized.

## Navigation

- UI: `web/`, `templates/`, `assets/`; analytics: `web/blueprints/analytics.py`,
  `core/analytics_core.py`.
- Pipeline/storage: `detectors/`, `core/`, `utils/`; inbox ingestion:
  `core/ingest_core.py`, `utils/ingest.py`.
- Camera/deployment: `camera/`, `rpi/`, `systemd/`, `scripts/`, `infra/`,
  `.github/workflows/`, `docker-compose.example.yml`.
- Optional Companion: `web/services/companion/`, `web/blueprints/companion.py`.

## Completion

Review the final diff and fix check failures caused by the change. For UI changes,
inspect affected flows in a browser at desktop/narrow widths and in Light/Dark;
report unavailable visual verification explicitly. A template test is insufficient.

Report what changed, affected invariants, checks/results, and remaining limitations.
Distinguish local verification from deployment/live-device verification and identify
pre-existing failures. Do not create a separate summary document unless requested;
the diff and, when committed, the commit message are the record.
