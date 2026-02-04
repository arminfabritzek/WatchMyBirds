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
Returns system status including detection state.

**Response:**
```json
{
  "detection_paused": false,
  "detection_running": true,
  "restart_required": false
}
```

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

#### GET `/api/settings`
Returns current application settings.

**Response:**
```json
{
  "VIDEO_SOURCE": "0",
  "CONFIDENCE_THRESHOLD": 0.6,
  "SAVE_THRESHOLD": 0.65,
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
  "MOTION_DETECTION_ENABLED": true,
  "MOTION_SENSITIVITY": 500
}
```

---

#### POST `/api/settings`
Updates application settings.

**Request Body:**
```json
{
  "VIDEO_SOURCE": "rtsp://192.168.1.100:554/stream",
  "CONFIDENCE_THRESHOLD": 0.7
}
```

**Response:**
```json
{
  "status": "success"
}
```

---

#### POST `/api/settings/update`
Alternative endpoint for settings update (same as POST `/api/settings`).

---

#### POST `/api/settings/ingest`
Triggers manual ingest from configured ingest directory.

**Response:**
```json
{
  "status": "success",
  "message": "Ingest completed"
}
```

---

### 3. ONVIF Camera Discovery

#### GET `/api/onvif/discover`
Scans network for ONVIF-compatible cameras.

**Response:**
```json
{
  "status": "success",
  "cameras": [
    {
      "ip": "192.168.1.100",
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
Retrieves RTSP stream URI for a camera.

**Request Body:**
```json
{
  "ip": "192.168.1.100",
  "port": 80,
  "username": "admin",
  "password": "secret"
}
```

**Response:**
```json
{
  "status": "success",
  "uri": "rtsp://admin:secret@192.168.1.100:554/stream"
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
      "ip": "192.168.1.100",
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
  "ip": "192.168.1.100",
  "port": 80,
  "username": "admin",
  "password": "secret"
}
```

**Response:**
```json
{
  "status": "success",
  "camera": {
    "id": 1,
    "name": "Front Yard Camera",
    "ip": "192.168.1.100",
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
Returns software version information.

**Response:**
```json
{
  "status": "success",
  "app_version": "2026.02.04",
  "python_version": "3.11.6",
  "opencv_version": "4.8.1"
}
```

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

### 2026-02-04
- Initial documentation created from code inventory
