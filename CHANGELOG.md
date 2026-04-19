# Changelog

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
