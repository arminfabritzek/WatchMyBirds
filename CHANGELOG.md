# Changelog

## Unreleased

### Added

- Detail views now let reviewers correct offered birds, withdraw a species
  confirmation, and record a box verdict without answering unrelated questions.
- Training Export now offers a box-verdict walkthrough and explains when missing
  originals exclude labeled birds from training-ready counts.

### Fixed

- Species answers now establish bird presence for training eligibility. A
  backfill command repairs earlier answers while leaving contradictions for
  human review.

## 0.5.7 - 2026-09-17

### Added

- Raspberry Pi users can now preview and restore USB snapshots through Settings,
  with authenticated confirmation, independent progress during app restart,
  complete checkpoints, health checks, retry, and rollback after failed startup.
  Recovery requests work within the hardened app sandbox; journaled swaps are
  durably ordered, rollback refuses a running app, and source-device credentials
  and camera files are never imported.
- Storage Retention can now run automatically once per day through a separate
  opt-in, processes large image collections in bounded batches, coordinates
  concurrent protection changes, and preserves clear results across restarts.
- Nightly jobs retain their latest run status across app restarts. Settings
  shows start and finish times, results, and errors, including interrupted runs
  and runs that finish after a stop request.

### Changed

- The navigation labels the model-training dataset download as Training Export.
- The live status bar starts collapsed for new browser visitors while preserving
  each browser's saved visibility preference.
- Settings groups backup and migration controls under Data & Backups.
- The aesthetic tagger's stop feedback now clarifies that its current run
  finishes before the stop request takes effect.

### Fixed

- Live streams show connection feedback, start without an artificial delay,
  and retain the correct aspect ratio and reconnect behavior.
- Raspberry Pi image builds preserve installed data assets and gate builds on
  tests; release tags and Docker image metadata identify the source being built.
  Docker images include the offline snapshot recovery command.
- Runtime and image-processing dependencies receive maintenance updates.

- USB backups exclude local development-tool state, retain copy diagnostics,
  and show current stages and final outcomes in Settings. Recovery ignores
  regenerable model caches in older snapshots, including their internal symlinks.

- Large canonical training exports write archives to disk and clean up completed
  or interrupted downloads. Browser backups use native download streaming.
- Backup merge preserves image associations on filename conflicts and reports
  database failures. Database replacement requires offline recovery.
- USB snapshots validate expected originals before completion. Recovery validates
  snapshot metadata and checksums, stages data before replacement, and retains
  the previous output directory as a recovery checkpoint.

### Documentation

- Clarified optional telemetry notices to describe installation history,
  its use for activity and version trends, and actual retention behavior.
  Corrected anonymity and deletion claims in Privacy, Settings, and setup.
- README badges show the combined release and Docker build status, latest
  stable version, and a direct link to the Raspberry Pi release downloads.

## 0.5.6 - 2026-09-10

### Fixed

- Camera capture now keeps retrying after repeated RTSP connection failures,
  respecting recovery cooldowns instead of leaving bird detection without frames.
- Direct stream mode now opens the MJPEG live view without probing go2rtc;
  expected go2rtc connection failures no longer log a full traceback.
- Release builds now prepare changelog entries before building the Raspberry Pi
  image. When curated notes are missing, entries are generated from commits since
  the last release and reused for both release notes and the archived changelog.

### Documentation

- Added Blue Iris NVR to the community-tested camera list.

## 0.5.5 - 2026-09-03

### Changed

- The live status rail can now be hidden per browser, and its retired
  preset-overlay editing controls no longer crowd the header.

### Fixed

- Dense navigation, review controls, filter bars, Settings fields, and image
  actions now adapt across laptop and phone layouts, with touch-friendly
  targets that stay inside the viewport.
- GitHub releases now publish the curated `Unreleased` notes and archive them
  under the released version automatically, so release pages no longer omit
  the actual changes.

## 0.5.4 - 2026-09-01

### Added

- Compatible PTZ cameras can now centre a point clicked in the live view and
  visually verify each bounded movement, while follow mode steers pan and tilt
  together to keep moving birds in frame.
- Field benchmark tools can now export confirmed station evidence and compare
  detector variants or wildlife models across false-positive and provisional
  recall trade-offs.
- Compatible PTZ cameras now expose autofocus and manual focus controls in the
  live stream, while unsupported controls stay hidden.
- Analytics now includes an evidence-backed station report that records
  measured station effort and admits only explicitly supported observations.
- Gallery and review surfaces now share full-image no-bird confirmation,
  per-box species decisions, and canonical review progress.
- A selectable iNaturalist Birds classifier now identifies 964 bird taxa
  locally and is the default, while the original WatchMyBirds ONNX classifier
  remains available in Settings.
- Species display names can now be switched to English in Settings,
  alongside German and Norwegian.
- Selected images can be downloaded together as a ZIP from the species view,
  including favourites collected across different days.
- Direct bounding-box correction lets reviewers drag an offered box while
  preserving canonical human facts, detection presence, and refreshed crops
  for downstream training.
- Canonical training bundles replace the legacy export pool with
  schema-versioned OD, classifier, and negative views, explicit inclusion
  reasons, and an optional FiftyOne inspection workflow.

### Fixed

- Login links and protected-route redirects now preserve destination query
  parameters, so filtered or paginated views reopen after authentication.
- Bulk trash from species and subgallery views now returns to the parent index
  instead of reloading a stale page.
- RTSP startup and recovery now revalidate the camera's current resolution
  before reading raw FFmpeg frames, preventing stale cached dimensions from
  combining parts of consecutive frames after a camera resolution change.
- Trash and rejected crops no longer create whole-image bird-absent labels;
  "No birds in full image" always previews the full frame and requires fresh
  confirmation.
- PTZ discovery rewrites camera-advertised ONVIF addresses to the configured
  host when firmware reports an unreachable internal address.
- The "Stop" control on the nightly aesthetic (CLIP) tagger no longer
  implies an instant halt. The worker finishes the current image before
  stopping; the status text and button tooltip now say so.

### Changed

- The README and security guidance now state the project's agent-led
  development and automated-assurance limits, including that the appliance
  has not been independently audited for direct exposure to untrusted networks.
- The optional training-data sharing invitation now lives on the Export page;
  favouriting a bird is never interrupted by it.
- The Canonical Dataset page now leads with the open review queue and keeps
  inclusion and exclusion manifests collapsed until they are needed.
- PTZ follow mode now performs a short, bounded search along the bird's recent
  trajectory when detections briefly disappear, then stops safely if the target
  is not reacquired.
- Gallery thumbnails and detail modals now offer direct `Crop | Full` and
  `Focus | Full` view controls, with compact modal navigation and secondary
  detection actions collected in overflow menus.
- Dependency and CI maintenance: numpy, torch, torchvision, safetensors,
  llama-cpp-python, and several Docker/CI actions bumped to current
  versions.
- Corrected the aesthetic-tagger documentation: the tagger ships enabled
  by default, and the separate `requirements-aesthetic.txt` exists for
  CPU-index isolation, not because the feature is optional.

## 0.5.0 - 2026-06-23

### Added

- **Original-file retention lifecycle (default OFF).** An optional
  policy that deletes full-resolution originals past a configurable age
  while keeping their database rows, derivatives, favourites, and
  statistics. Three postures — `off`, `conservative`, and `reclaim` —
  let an operator choose how aggressively disk is freed; `conservative`
  reproduces prior behaviour, `reclaim` stops treating "unreviewed" as a
  reason to keep an original. Export-relevant training data, favourites,
  unreviewed images (under `conservative`), and images whose display
  derivatives don't yet exist are always protected. Originals are only
  ever whole-file deleted, never modified. A retired original serves a
  persistent "Original retired" state in the gallery — including inside
  the detail modals — with a preview download in place of the dead
  full-resolution button.
- **Favorites-only gallery filter.** A "Favorites only" toggle on the
  shared filter bar narrows `/species`, the species overview, and the
  per-day gallery to favourited detections. Mixed events never leak
  non-favourite members into a card, filmstrip, or modal, and the choice
  persists per browser.
- **Reversible "Move Review Queue to Trash" bulk action.** For operators
  who don't work the Review desk, a preview-first action empties the
  entire review queue into Trash using the existing reversible
  primitives — no files are deleted, and items restore from Trash.

### Fixed

- Clean up stale `ffmpeg` child processes before a stream restart, so a
  restart no longer leaves orphaned encoders behind.
- Preserve the "retired original" state correctly inside gallery modals.

### Security

- The detector-precision endpoint no longer surfaces raw exception text
  in its 400 response; error messages are routed through an allow-list
  so unrelated internals cannot leak to the client.

## 0.4.3 - 2026-06-13

### Added

- **Telemetry — consent + aggregation + instant toggle.** Three
  improvements on top of the initial heartbeat:
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
    pre-existing for `_detect_detector_variant`).

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
    deployed to a Cloudflare Workers endpoint (the operator-facing
    URL ships in `config.py` and `docs/PRIVACY.md`). The Worker
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
