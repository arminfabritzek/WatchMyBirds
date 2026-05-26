# Changelog

## Unreleased

### Added

- **Telemetry Phase 2 — consent + aggregation + instant toggle.** Three
  improvements on top of the Phase 1 heartbeat:
  - **First-Run consent checkbox** in `setup_password.html` so new
    operators see the opt-in question once at install time, not buried
    in Settings. Default unchecked. Submitting the password without
    ticking the box leaves telemetry off, exactly as before — proactive
    transparency, not nudge-toward-yes.
  - **24-hour individual-heartbeat retention.** A second Worker cron at
    04:30 UTC aggregates yesterday's heartbeats into a `daily_aggregates`
    table by (date, cohort, app_version, detector_variant) and deletes
    the raw rows. The aggregate has no `installation_id` — the
    per-install timeline is gone after 24h. Privacy claim upgraded from
    "90-day TTL on raw rows" to "24h on raw rows + cohort counts kept."
  - **Instant toggle-on response.** A `threading.Event` in
    `telemetry_service` lets the toggle endpoint poke the scheduler
    out of its 5-minute sleep, so the first heartbeat after toggle-on
    fires within ~10ms instead of waiting up to one full tick.
  - 9 new tests (3 for Event wake-up, 2 for first-run consent, 4
    already in Phase 1 for `_detect_detector_variant`).

- **Anonymous opt-in usage heartbeat (default OFF).** A new
  Settings → Privacy section lets operators optionally send one
  anonymous JSON payload per UTC day so we can count active
  installations. Default is off and there is no banner, popup, or
  weekly nag — the toggle is the only enable surface.
  - Payload: 8 small fields (random installation ID, app version,
    OS family, arch, CPU count, rounded RAM in GB, Python version,
    detector variant). No IP, no country, no locale, no hostname,
    no MAC, no kernel/Pi-model strings, no exact RAM bytes, no image
    or detection data.
  - Storage: Cloudflare D1 with `jurisdiction=eu` (Free-tier-native
    EU data residency, distinct from the Enterprise-only Data
    Localization Suite). 90-day retention enforced by a daily cron
    `DELETE` inside the same Worker.
  - Worker source is in this repo at `infra/telemetry-worker/`,
    deployed to `https://heartbeat-wmb.starmin.de`. The Worker
    rejects requests without the `WatchMyBirds-Heartbeat/<version>`
    User-Agent (404, no recon hint), validates the payload shape
    strictly, and drops all CF-injected metadata before writing.
  - Operator controls: toggle off (pings stop, ID preserved),
    rotate ID (next ping counted as a fresh install), endpoint
    override in `settings.yaml` (point at any URL or `/dev/null`),
    firewall-blockable hostname (separate from any other WMB
    endpoint).
  - Full data policy: `docs/PRIVACY.md` (or `/privacy` in the
    running app, no login required). Footer of every page links to
    the policy. README has a Privacy H2 above the fold.
  - 26 unit tests cover the strict default-OFF guarantee, the
    8-field payload shape, the no-PII allowlist, UUID lifecycle
    (lazy-gen, persistence, rotate), and atomic last-sent file
    semantics.
  - First-Run consent screen and aggregated public DAU dashboard
    are deliberately out of scope here; they ship in a follow-up
    after this introduction release has run for a real interval.

- **USB stick backup (write-only v1).** Daily automatic snapshots of the
  SQLite database, captured imagery, and installed app code to an optional
  USB stick (label `WMB-BACKUP`, ext4). Protects against SD-card death,
  the single most common hardware failure on a long-running Raspberry Pi.
  - Online SQLite snapshot via the `.backup` pragma — no app stop needed.
  - rsync `--link-dest` deduplication: each daily snapshot uses ~5% of
    live data, not 100%, by hardlinking unchanged files between snapshots.
  - `COMPLETED` marker for crash-consistent recovery; orphaned in-progress
    snapshots are pruned on the next run.
  - Kind-aware retention: scheduled keeps 7 daily / 4 weekly / 6 monthly;
    manual keeps the latest 3. Corrupt snapshots are never auto-deleted.
  - Settings → Tools & System gains a "USB Backup" card with stick state,
    free-space bar, recent snapshots list, and a "Backup now" button.
  - Five new endpoints under `/api/v1/system/backup/*` (status, list,
    trigger, delete, verify).
  - Restore is **not** part of this release — recovery in v1 is a manual
    procedure on a separate Linux machine, documented in
    `docs/USB_BACKUP.md`. UI restore + OTA pre-update snapshot hook ship
    in v2.

## 0.2.0 - 2026-04-20

Headline release focused on a new detector stack, a guided review workflow,
and a richer model management story. This version introduces breaking
configuration changes (see below) and raises the quality of the built-in
species catalog.

### Highlights

- **New YOLOX-based detector.** Swap the legacy FasterRCNN locator for a
  YOLOX raw-output backend with automatic format sniffing. Thresholds now
  live with the model (`model_metadata.json`) so each variant ships with
  its own confidence floor.
- **Model variants & live precision switch.** Manage multiple detector
  variants (Tiny / S / fp32 / int8_qdq) from the settings UI. Switch
  precision at runtime without restarting the service.
- **Non-bird species as first-class citizens.** Squirrel, cat,
  marten/mustelid, and hedgehog detections flow through a dedicated
  scoring track, get their own review artwork, and show up on every
  surface (gallery, stream, analytics) as their own species.
- **BirdEvent review workflow.** Group related detections into events so
  reviewers confirm or relabel a whole sighting in one step. The Review
  desk gained direct panel lookup and lazy species-picker loading for
  faster triage on large backlogs.
- **Telegram overhaul.** New `TELEGRAM_MODE` enum (`off` / `live` /
  `daily` / `interval`), optional `DEVICE_NAME` prefix, and a manual
  send button for the daily report.
- **Live stream reliability.** Two-phase go2rtc probe keeps reverse-proxy
  setups stable; bbox overlays now persist across navigation and zoom.
- **Save-threshold mode.** Auto / Manual toggle — Auto derives the save
  threshold from the model's detection floor plus a locked offset.
- **Ops polish.** Tighter log format, quieter go2rtc retries, newest-
  first log view, and a detector-variant benchmark tool.

### Breaking changes

- `CONFIDENCE_THRESHOLD_DETECTION` is retired. Detection confidence is
  now model-owned; set per-variant overrides in the Settings UI if
  needed.
- New setting `SAVE_THRESHOLD_MODE` defaults to `auto`. Existing
  `SAVE_THRESHOLD` values are honoured only when the mode is `manual`.
- The legacy FasterRCNN post-NMS format is no longer loadable. Startup
  auto-cleanup removes legacy artefacts before the detector initialises
  so the Hugging Face autofetch pulls the current YOLOX release.

### Security & supply chain

- All GitHub Actions pinned to immutable SHA hashes (with trailing
  version comment) to guard against action-repo compromise.
- Docker base image pinned to its multi-arch index digest so amd64 and
  arm64 builds stay reproducible.
- `numpy` pinned to an exact version alongside the rest of the
  requirements.
- Added Dependabot (pip / github-actions / docker) and a CodeQL workflow
  covering Python, JavaScript/TypeScript, and Actions.
- First-boot flow requires an admin password on the Raspberry Pi
  appliance; login rate-limits at 5 attempts per 5 minutes per IP.
- CSRF token check on all state-changing requests; session cookies use
  `HttpOnly` and `SameSite=Lax`.

### Community

- Added issue forms (bug report, feature request) and a pull-request
  template with a security and secrets checklist.
- Adopted the Contributor Covenant 2.1 as the project Code of Conduct.
- Security vulnerabilities are routed through GitHub Security Advisories
  (see `SECURITY.md`).

## 0.1.1 - 2026-04-04

This patch release consolidates the April 4, 2026 improvements into a single
versioned release so GitHub releases and Docker tags can point at the same
stable build.

### Highlights

- Upgrade Raspberry Pi build and runtime paths to Python 3.12.
- Harden the Raspberry Pi setup flow by requiring an admin password and
  tightening the first-boot experience.
- Improve the review workflow with refined quick species selection states and a
  larger review species artwork set.
- Refresh setup and deployment documentation for the Python 3.12 baseline.
- Improve the live stream overlay so the clock stays responsive and shows only
  temperature plus humidity.

### Included 2026-04-04 commits

- `855a1c1` `build(rpi): migrate build and runtime to python 3.12`
- `20b9c91` `security(rpi): require admin password and harden setup flow`
- `49d773d` `docs: refresh python 3.12 and setup guidance`
- `67665c5` `feat(review): refine quick species selection states`
- `9dab33d` `assets: expand review species artwork`
- `e9d3191` `Make stream clock overlay responsive and show temp + humidity only`
