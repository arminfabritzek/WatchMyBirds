"""
WatchMyBirds Core Package.

This package contains the core business logic of the application,
separated from the web layer. All database, image, and detection
operations are coordinated through core modules.

ARCHITECTURE RULES:
- core/ modules may only import from:
  - Python standard library
  - utils/ (legacy adapters)
  - detectors/ (for detection-related orchestration)
  - camera/ (for camera access)
  - config (for global configuration)

- core/ modules MUST NOT import from:
  - web/ (no Flask dependencies)
  - flask, werkzeug, or any web-specific packages

- All new business logic should be placed here, not in web/
"""

__all__ = [
    "gallery_core",
    "settings_core",
    "onvif_core",
    "analytics_core",
    "detections_core",
    "backup_restore_core",
    "db_core",
    "ingest_core",
    "path_core",
]
