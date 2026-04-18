# WatchMyBirds API Documentation

> **Version:** 1.0.0  
> **Base URL:** `/api`  
> **Last Updated:** 2026-02-04

This document describes the **existing** API endpoints as they are currently implemented.
It serves as the reference for the `/api/v1` migration.

---

## Response Format Convention

All API endpoints follow this response pattern:

### Success Response
```json
{
  "status": "success",
  ...additional fields...
}
```

### Error Response
```json
{
  "status": "error",
  "message": "Human-readable error description"
}
```

**HTTP Status Codes:**
- `200` - Success
- `400` - Bad Request (missing/invalid parameters)
- `404` - Resource not found
- `500` - Internal Server Error

---

## Endpoints by Domain

### 1. Status & Control

#### GET `/api/status`
Returns system status including detection state and deep scan progress.

**Response:**
```json
{
  "detection_paused": false,
  "detection_running": true,
  "restart_required": false,
  "deep_scan_active": false,
  "deep_scan_queue_pending": 0,
  "deep_scan_candidates_remaining": 42
}
```

| Field | Type | Description |
|-------|------|-------------|
| `detection_paused` | bool | Whether the detection loop is paused |
| `detection_running` | bool | Inverse of `detection_paused` |
| `restart_required` | bool | Whether a service restart is pending |
| `deep_scan_active` | bool | `true` while a Deep Scan job is executing (live DET+CLS loops are gated) |
| `deep_scan_queue_pending` | int | Number of Deep Scan jobs waiting in the worker queue |
| `deep_scan_candidates_remaining` | int | DB count of nightly-eligible orphan images (never-scanned + error retries) |

---

#### POST `/api/detection/pause`
Pauses the detection loop.

**Response:**
```json
{
  "status": "success",
  "message": "Detection paused"
}
```

---

#### POST `/api/detection/resume`
Resumes the detection loop.

**Response:**
```json
{
  "status": "success",
  "message": "Detection resumed"
}
```

---

### 2. Settings

#### GET `/api/v1/settings`
Returns current application settings.

**Response:**
```json
{
  "VIDEO_SOURCE": "0",
  "SAVE_THRESHOLD": 0.65,
  "SAVE_THRESHOLD_MODE": "auto",
  "DEBUG_MODE": false,
  "TELEGRAM_ENABLED": false,
  "TELEGRAM_BOT_TOKEN": "",
  "TELEGRAM_CHAT_ID": "",
  "TELEGRAM_COOLDOWN": 300,
  "LOCATION_DATA": {
    "latitude": 48.1351,
    "longitude": 11.5820,
    "city": "Munich"
  },
  "EXIF_GPS_ENABLED": true,
  "SPECIES_COMMON_NAME_LOCALE": "DE",
  "MOTION_DETECTION_ENABLED": true,
  "MOTION_SENSITIVITY": 500
}
```

> ⚠️ **Breaking change (0.2.0):** the `CONFIDENCE_THRESHOLD_DETECTION`
> key has been removed from the response. The detector's confidence
> floor is now model-owned and exposed via
> `GET /api/v1/models/detector` (see below) instead of being operator-
> editable.

---

#### POST `/api/v1/settings`
Updates application settings. Only runtime-modifiable keys are accepted.

**Request Body:**
```json
{
  "SAVE_THRESHOLD_MODE": "manual",
  "SAVE_THRESHOLD": 0.50,
  "SPECIES_COMMON_NAME_LOCALE": "NO"
}
```

**Response:**
```json
{
  "status": "success"
}
```

> **Note:** Legacy endpoints `GET/POST /api/settings`, `POST /api/settings/update`,
> and `POST /api/settings/ingest` have been removed.
> Use `POST /api/v1/settings` for settings and `POST /api/ingest/start` for ingest.

---

#### GET `/api/v1/models/detector`
Returns the active detector variant, its runtime calibration, and the
list of variants available for a live switch.

**Response:**
```json
{
  "model_dir": "/models/object_detection",
  "active": {
    "id": "20260417_1636_yolox_tiny_640_mosaic0p5",
    "source": "latest_models",
    "env_pin_value": null,
    "hf_latest_id": "20260417_1636_yolox_tiny_640_mosaic0p5",
    "runtime_matches_on_disk": true
  },
  "runtime": {
    "model_id": "20260417_1636_yolox_tiny_640_mosaic0p5",
    "output_format": "yolox_raw",
    "input_size": [640, 640],
    "num_classes": 5,
    "class_names": ["bird", "squirrel", "cat", "marten_mustelid", "hedgehog"],
    "conf_threshold_default": 0.15,
    "iou_threshold_default": 0.5
  },
  "metadata": { "variant": "tiny", "metrics": {"bird_recall": 0.993} },
  "variants": [
    {"id": "20260417_1636_yolox_tiny_640_mosaic0p5", "is_active": true,  "is_available_locally": true,  "is_hf_latest": true},
    {"id": "20260417_1512_yolox_s_640_mosaic0p5",    "is_active": false, "is_available_locally": false, "is_hf_latest": false}
  ]
}
```

`runtime.conf_threshold_default` is the active detection-confidence
floor, read from the active variant's `model_metadata.json`. This is
the value the Settings page shows next to the Save Threshold field
so the user can see which calibration the save gate is tracking.

#### POST `/api/v1/models/detector/install`
Downloads a known-but-missing variant (weights + labels + YAML +
metrics) from HuggingFace. `model_id` must be a key listed in the
GET response's `variants`. Whitelist-gated — arbitrary strings are
rejected.

**Request Body:**
```json
{ "model_id": "20260417_1512_yolox_s_640_mosaic0p5" }
```

**Response:**
```json
{
  "status": "success",
  "model_id": "20260417_1512_yolox_s_640_mosaic0p5",
  "already_installed": false,
  "weights_path": "/models/object_detection/20260417_1512_yolox_s_640_mosaic0p5_best.onnx",
  "labels_path":  "/models/object_detection/20260417_1512_yolox_s_640_mosaic0p5_labels.json",
  "model_config_path": "/models/object_detection/20260417_1512_yolox_s_640_mosaic0p5_model_config.yaml",
  "metrics_path":      "/models/object_detection/20260417_1512_yolox_s_640_mosaic0p5_metrics.json"
}
```

#### POST `/api/v1/models/detector/pin`
Switches the active variant by rewriting `latest_models.json` and
regenerating `model_metadata.json` from the new variant's YAML. The
DetectionService is cleared for a live reload on the next inference
cycle (~1–2 s, no service restart).

**Request Body:**
```json
{ "model_id": "20260417_1512_yolox_s_640_mosaic0p5" }
```

If a systemd env-var pin is set, this endpoint still rewrites the
on-disk pointer (so the next env-pin-free startup picks up the UI
choice) but reports `env_pin_overrides: true` in the response so the
UI can warn that the in-memory detector stays on the env-pin value
until the pin is removed.

---

### 3. ONVIF Camera Discovery

#### GET `/api/onvif/discover`
Scans the network for supported IP cameras.

**Response:**
```json
{
  "status": "success",
  "cameras": [
    {
      "ip": "198.51.100.10",
      "port": 80,
      "name": "Camera 1",
      "manufacturer": "Reolink",
      "model": "RLC-520A"
    }
  ]
}
```

**Empty Result:**
```json
{
  "status": "success",
  "cameras": []
}
```

---

#### POST `/api/onvif/get_stream_uri`
Retrieves an RTSP stream URI for a configured camera.

**Request Body:**
```json
{
  "ip": "198.51.100.10",
  "port": 80,
  "username": "viewer",
  "password": "example-password"
}
```

**Response:**
```json
{
  "status": "success",
  "uri": "rtsp://viewer:example-password@198.51.100.10:554/stream"
}
```

---

### 4. Camera Management

#### GET `/api/cameras`
Lists all saved cameras.

**Response:**
```json
{
  "status": "success",
  "cameras": [
    {
      "id": 1,
      "name": "Front Yard Camera",
      "ip": "198.51.100.10",
      "port": 80,
      "username": "admin"
    }
  ]
}
```

---

#### POST `/api/cameras`
Adds a new camera.

**Request Body:**
```json
{
  "name": "Front Yard Camera",
  "ip": "198.51.100.10",
  "port": 80,
  "username": "viewer",
  "password": "example-password"
}
```

**Response:**
```json
{
  "status": "success",
  "camera": {
    "id": 1,
    "name": "Front Yard Camera",
    "ip": "198.51.100.10",
    "port": 80
  }
}
```

---

#### PUT `/api/cameras/<camera_id>`
Updates an existing camera.

**Request Body:**
```json
{
  "name": "Updated Name",
  "password": "new_password"
}
```

**Response:**
```json
{
  "status": "success"
}
```

---

#### DELETE `/api/cameras/<camera_id>`
Deletes a camera.

**Response:**
```json
{
  "status": "success"
}
```

---

#### POST `/api/cameras/<camera_id>/test`
Tests camera connection.

**Response:**
```json
{
  "status": "success",
  "message": "Camera connection successful"
}
```

---

#### POST `/api/cameras/<camera_id>/use`
Sets camera as active video source.

**Response:**
```json
{
  "status": "success",
  "message": "Video source updated"
}
```

---

### 5. Analytics

#### GET `/api/analytics/summary`
Returns detection analytics summary.

**Response:**
```json
{
  "total_detections": 1234,
  "total_species": 15,
  "total_days": 42,
  "species_counts": {
    "Parus major": 450,
    "Cyanistes caeruleus": 320,
    "Erithacus rubecula": 210
  }
}
```

---

#### GET `/api/analytics/time_of_day`
Returns detection time distribution.

**Response:**
```json
{
  "points": [0.5, 6.25, 7.0, 12.5, 18.75],
  "peak_hour": 7,
  "histogram": [0, 0, 0, 5, 12, 45, 120, 85, 42, 15, 8, 3, ...]
}
```

---

#### GET `/api/analytics/species_activity`
Returns species activity heatmap data.

**Response:**
```json
{
  "heatmap": {
    "Parus major": [0, 0, 0, 5, 12, 25, ...],
    "Cyanistes caeruleus": [0, 0, 2, 8, 15, 20, ...]
  }
}
```

---

#### GET `/api/daily_species_summary`
Returns species summary for a specific date.

**Query Parameters:**
- `date` (optional): ISO date string (YYYY-MM-DD), defaults to today

**Response:**
```json
{
  "date": "2026-02-04",
  "species": [
    {"name": "Parus major", "count": 15, "common_name": "Kohlmeise"},
    {"name": "Cyanistes caeruleus", "count": 8, "common_name": "Blaumeise"}
  ]
}
```

---

### 6. System

#### GET `/api/system/stats`
Returns system resource statistics.

**Response:**
```json
{
  "status": "success",
  "cpu": 45.2,
  "ram": 62.5,
  "temp": 52.0,
  "disk": {
    "total_gb": 128.0,
    "used_gb": 45.5,
    "free_gb": 82.5,
    "percent": 35.5
  }
}
```

---

#### GET `/api/system/versions`
Returns system and build metadata (legacy route).

**Response:**
```json
{
  "app_version": "0.1.0",
  "git_commit": "abc1234",
  "build_date": "2026-03-11T10:00:00Z",
  "deploy_type": "rpi",
  "kernel": "6.6.51+rpt-rpi-v8",
  "os": "Debian GNU/Linux 13 (trixie)",
  "bootloader": "2024-09-23"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `app_version` | string | Semantic version from `APP_VERSION` file |
| `git_commit` | string | Short (7-char) commit hash |
| `build_date` | string | ISO build timestamp |
| `deploy_type` | string | `docker`, `rpi`, or `dev` |
| `kernel` | string | OS kernel version |
| `os` | string | Pretty OS name |
| `bootloader` | string | RPi bootloader version (or `Unknown`) |

---

#### GET `/api/v1/system/versions`
Returns system and build metadata plus V1-only extras.

**Response:**
```json
{
  "status": "success",
  "app_version": "0.1.0",
  "git_commit": "abc1234",
  "build_date": "2026-03-11T10:00:00Z",
  "deploy_type": "rpi",
  "kernel": "6.6.51+rpt-rpi-v8",
  "os": "Debian GNU/Linux 13 (trixie)",
  "bootloader": "2024-09-23",
  "python_version": "3.12.12",
  "opencv_version": "4.8.1"
}
```

The shared metadata subset (`app_version`, `git_commit`, `build_date`,
`deploy_type`, `kernel`, `os`, `bootloader`) is identical across both routes.
V1 additionally returns `status`, `python_version`, and `opencv_version`.

---

#### POST `/api/system/shutdown`
Initiates system shutdown (Raspberry Pi only).

**Response:**
```json
{
  "status": "success",
  "message": "System is shutting down..."
}
```

---

#### POST `/api/system/restart`
Initiates system restart (Raspberry Pi only).

**Response:**
```json
{
  "status": "success",
  "message": "System is restarting..."
}
```

---

### 7. Snapshot

#### GET `/api/snapshot`
Captures current camera frame as JPEG.

**Response:** Binary JPEG image with `Content-Type: image/jpeg`

---

### 8. Inbox (Web Upload)

#### POST `/api/inbox`
Uploads images to inbox for processing.

**Request:** Multipart form with `files[]` containing images.

**Response:**
```json
{
  "status": "success",
  "uploaded": 5,
  "files": ["image1.jpg", "image2.jpg", ...]
}
```

---

#### GET `/api/inbox/status`
Returns inbox processing status.

**Response:**
```json
{
  "status": "success",
  "pending": 5,
  "processing": false
}
```

---

#### POST `/api/inbox/process`
Starts processing uploaded inbox images.

**Response:**
```json
{
  "status": "success",
  "message": "Processing started for 5 images"
}
```

---

### 9. Ingest

#### POST `/api/ingest/start`
Triggers ingest from configured directory.

**Response:**
```json
{
  "status": "success"
}
```

---

### 10. Backup & Restore

#### GET `/api/backup/stats`
Returns backup storage statistics.

**Response:**
```json
{
  "status": "success",
  "total_images": 1234,
  "total_size_mb": 5678.9,
  "db_size_mb": 12.5
}
```

---

#### POST `/api/backup/create`
Creates a backup archive.

**Response:**
```json
{
  "status": "success",
  "filename": "watchmybirds_backup_20260204_133000.zip",
  "size_mb": 1234.5
}
```

---

#### POST `/api/restore/upload`
Uploads a backup file for restoration.

**Request:** Multipart form with `file` containing backup archive.

**Response:**
```json
{
  "status": "success",
  "temp_file": "/tmp/restore_xyz.zip"
}
```

---

#### POST `/api/restore/analyze`
Analyzes uploaded backup for contents.

**Response:**
```json
{
  "status": "success",
  "images": 1234,
  "database": true,
  "config": true
}
```

---

#### POST `/api/restore/apply`
Applies the restoration.

**Request Body:**
```json
{
  "restore_images": true,
  "restore_database": true,
  "restore_config": false
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Restore completed"
}
```

---

#### GET `/api/restore/status`
Returns restore operation status.

**Response:**
```json
{
  "status": "success",
  "active": false,
  "progress": 100
}
```

---

#### POST `/api/restore/cleanup`
Cleans up temporary restore files.

**Response:**
```json
{
  "status": "success"
}
```

---

## Migration Plan: `/api` → `/api/v1`

1. ✅ Document current endpoints (this file)
2. ⏳ Create `/api/v1` blueprint
3. ⏳ Implement versionedroutes with identical behavior
4. ⏳ Add 5 mandatory API tests
5. ⏳ Migrate frontend to `/api/v1`
6. ⏳ Deprecate legacy `/api` routes

---

## Changelog

### 2026-02-13
- **Deep Scan Stability Hardening:**
  - `GET /api/status` now includes `deep_scan_active`, `deep_scan_queue_pending`, `deep_scan_candidates_remaining`.
  - `POST /api/review/analyze/<filename>` accepts `?force=1` to bypass no-hit DB exclusion.
- **UI: Review Modal "Preview only" Fallback** (no API change):
  - Review modal (`orphan_modal.html`) now shows a visible amber "Preview only" badge when the full-size image fails to load and falls back to thumbnail. Previously the fallback was silent.
- **UI: Global Zoom Preference Persistence** (no API change):
  - Smart Zoom toggle state (`Full` / `Zoom`) is now persisted globally via `localStorage` key `wmb_modal_zoom_pref`. Preference survives page reloads and modal navigation. Applies wherever bbox-based zoom is available (Gallery modals).

### 2026-02-04
- Initial documentation created from code inventory
