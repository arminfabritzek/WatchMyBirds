# Configuration Reference

This document describes all configuration options for WatchMyBirds.

---

## Configuration Model

WatchMyBirds uses a **two-layer configuration model**:

### Boot / Infrastructure Settings
- Loaded once at startup from environment variables (`.env` or Docker)
- Shown as **read-only** in the Settings UI
- Changes require a service restart

### Runtime Settings
- Stored in `OUTPUT_DIR/settings.yaml`
- Editable via the Settings UI without restart
- Applied immediately

### Merge Order
```
defaults → environment variables → settings.yaml
```

Runtime edits update `settings.yaml` only — the `.env` file is never mutated.

---

## Environment Variables

Set these in your `.env` file, `docker-compose.yml`, or system environment.

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG_MODE` | `False` | Enable verbose logging and debug behavior |
| `OUTPUT_DIR` | `/output` | Base directory for images and `images.db` |
| `INGEST_DIR` | `/ingest` | Directory for bulk image ingestion |
| `MODEL_BASE_PATH` | `/models` | Base directory for AI model files |

### Camera Source Resolution

| Variable | Default | Description |
|----------|---------|-------------|
| `CAMERA_URL` | `""` | User-facing camera source URL (usually RTSP) |
| `STREAM_SOURCE_MODE` | `auto` | Source policy: `auto`, `relay`, or `direct` |
| `GO2RTC_STREAM_NAME` | `camera` | Relay stream name used for `rtsp://<host>:8554/<name>` |
| `GO2RTC_API_BASE` | `http://127.0.0.1:1984` | Go2RTC health probe endpoint |
| `GO2RTC_CONFIG_PATH` | `./go2rtc.yaml` | Writable go2rtc config file path synchronized by the app |
| `STREAM_FPS_CAPTURE` | `5.0` | Capture FPS throttle (reduces CPU load) |
| `STREAM_FPS` | `0` | UI MJPEG feed throttle (0 = unlimited) |
| `STREAM_WIDTH_OUTPUT_RESIZE` | `640` | Width for live stream preview in the UI |

Resolver behavior:
- `auto`: use relay when go2rtc probe succeeds; otherwise use `CAMERA_URL` directly
- `relay`: always use go2rtc relay URL
- `direct`: always use `CAMERA_URL`

Deployment-specific `GO2RTC_CONFIG_PATH` values:
- Docker Compose: `/output/go2rtc.yaml`
- RPi appliance (`app.service`): `/opt/app/data/output/go2rtc.yaml`

### Detection & Classification

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIDENCE_THRESHOLD_DETECTION` | `0.55` | Detector confidence threshold |
| `CLASSIFIER_CONFIDENCE_THRESHOLD` | `0.55` | Classifier confidence for gallery summaries |
| `DETECTION_INTERVAL_SECONDS` | `2.0` | Pause between detection cycles (seconds) |
| `SAVE_THRESHOLD` | `0.55` | Minimum confidence to save an image |

> ⚠️ **Rule:** Ensure `CONFIDENCE_THRESHOLD_DETECTION <= SAVE_THRESHOLD`  
> If detection threshold is higher, candidates are filtered before the save decision.

### Location & Daylight

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCATION_DATA` | `52.516, 13.377` | GPS lat/lon for EXIF metadata (`"lat, lon"`) |
| `DAY_AND_NIGHT_CAPTURE` | `True` | Enable daylight gating for detections |
| `DAY_AND_NIGHT_CAPTURE_LOCATION` | `Berlin` | City name for Astral daylight check |

### Notifications

| Variable | Default | Description |
|----------|---------|-------------|
| `TELEGRAM_ENABLED` | `False` | Enable Telegram notifications |
| `TELEGRAM_COOLDOWN` | `5` | Cooldown (seconds) between alerts |
| `TELEGRAM_BOT_TOKEN` | — | Bot token (env-only, never in settings.yaml) |
| `TELEGRAM_CHAT_ID` | — | Chat ID for notifications |

### Security

| Variable | Default | Description |
|----------|---------|-------------|
| `EDIT_PASSWORD` | `watchmybirds` | Password for protected UI actions |
| `FLASK_SECRET_KEY` | (auto-generated) | Session signing key |

> ⚠️ **Change the default password immediately in production!**

### Performance

| Variable | Default | Description |
|----------|---------|-------------|
| `CPU_LIMIT` | `0` | CPU affinity cap (`0` disables affinity; `1+` pins to first N available CPUs) |

### Deep Scan Stability

| Key | Default | Description |
|-----|---------|-------------|
| `DEEP_SCAN_GATE_ENABLED` | `True` | When `True`, live detection and classification loops pause while a Deep Scan job runs (prevents resource contention on RPi). Set `False` to allow concurrent Deep Scan + live detection. Runtime-changeable via Settings UI. |

---

## Important Notes

### Inbox Import Requirements (EXIF Date/Time + GPS)

Inbox imports (files uploaded/copied into the Inbox) can be configured to require
EXIF metadata before a file is ingested into the main database.

If enabled, the app will **skip** inbox files unless they contain:
- EXIF capture timestamp: `DateTimeOriginal` (preferred) or `DateTimeDigitized`
- EXIF GPS coordinates: `GPSLatitude` and `GPSLongitude`

Runtime settings:
- `INBOX_REQUIRE_EXIF_DATETIME` (default: `True`)
- `INBOX_REQUIRE_EXIF_GPS` (default: `True`)

Skipped files are moved to `inbox/skipped/YYYYMMDD/`.

### Ingest vs. Live Operation

| Mode | Behavior |
|------|----------|
| **Live Mode** | Only saves images with valid detections (filtered by `SAVE_THRESHOLD`) |
| **Ingest Tool** | Saves **every** image to ensure hash idempotency and reproducibility |

The Ingest behavior is intentionally different to prevent re-ingest loops.

### Threshold Interaction

```
┌─────────────────────────────┐
│  Raw Frame                  │
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  Detector                   │
│  (CONFIDENCE_THRESHOLD_     │
│   DETECTION filters boxes)  │
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  Save Decision              │
│  (SAVE_THRESHOLD decides    │
│   if image is persisted)    │
└─────────────────────────────┘
```

---

## Tested Cameras

| Hardware | Protocol | Notes |
|----------|----------|-------|
| Low-cost PTZ camera | RTSP | Tested with various brands |
| Raspberry Pi Camera | HTTP | Via MotionEye OS |
| USB Webcam | Direct | Device index (0, 1, ...) |

---

## Example .env File

```bash
# Core
OUTPUT_DIR=/data/watchmybirds/output
INGEST_DIR=/data/watchmybirds/ingest
MODEL_BASE_PATH=/data/watchmybirds/models

# Camera
CAMERA_URL=rtsp://admin:password@192.168.1.100:554/stream1
STREAM_SOURCE_MODE=auto
GO2RTC_API_BASE=http://go2rtc:1984
GO2RTC_CONFIG_PATH=/output/go2rtc.yaml
STREAM_FPS_CAPTURE=5.0

# Detection
CONFIDENCE_THRESHOLD_DETECTION=0.55
DETECTION_INTERVAL_SECONDS=2.0

# Security
EDIT_PASSWORD=your-secure-password

# Notifications (optional)
TELEGRAM_ENABLED=True
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_CHAT_ID=987654321
```

---

## Inbox (Web Upload)

The Inbox feature allows uploading images through the web UI for later processing.

### Directory Structure

```
OUTPUT_DIR/
└── inbox/
    ├── pending/           # Uploaded files waiting for processing
    ├── processed/YYYYMMDD/  # Successfully ingested files
    ├── skipped/YYYYMMDD/    # Duplicate files (by SHA-256 hash)
    └── error/               # Files that failed to process
```

### Upload Limits

| Limit | Value |
|-------|-------|
| Max files per upload | 20 |
| Max file size | 50 MB |
| Allowed formats | `.jpg`, `.jpeg`, `.png` |

### Processing Policy

- **Detection must be stopped** before processing inbox files
- Processing is manual (button click required)
- A snapshot of `pending/` is taken at start; new uploads during processing remain for next run
- Duplicates are detected by SHA-256 hash and moved to `skipped/`

---

## Backup & Migration

The Backup feature creates streaming `.tar.gz` archives for data migration.

### Backup Contents

| Component | Default | Description |
|-----------|---------|-------------|
| Database (`images.db`) | ✅ Included | All detections, classifications, metadata |
| Originals | ✅ Included | Original JPEG images with EXIF |
| Derivatives | ❌ Excluded | Thumbnails and optimized versions (can be regenerated) |
| Settings | ✅ Included | `settings.yaml` configuration |

### Backup Policy

- **Detection must be stopped** before creating a backup
- Archives are streamed directly (no local storage on the appliance)
- Filename format: `watchmybirds_backup_YYYYMMDD_HHMMSS.tar.gz`

---

## See Also

- [Architecture](ARCHITECTURE.md) — System design overview
- [Invariants](INVARIANTS.md) — Core rules
- [Security Policy](../SECURITY.md) — Hardening measures
