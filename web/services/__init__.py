"""
WatchMyBirds Services Package.

This package contains service layer classes that encapsulate business logic,
separating it from Flask routes for better testability and maintainability.

ARCHITECTURE RULE (HARD invariant H-01):
- Services may import from core/*, stdlib/typing, config, logging_config,
  utils.*, and other web.services.* modules
- Services MUST NOT import directly from camera/* or detectors/*
"""

from web.services import (
    analytics_service,
    auth_service,
    backup_restore_service,
    detections_service,
    gallery_service,
    health_service,
    ingest_service,
    onvif_service,
    path_service,
    ptz_service,
    settings_service,
    usb_backup_service,
    usb_format_service,
)

__all__ = [
    "analytics_service",
    "auth_service",
    "backup_restore_service",
    "db_service",
    "detections_service",
    "gallery_service",
    "health_service",
    "ingest_service",
    "onvif_service",
    "path_service",
    "ptz_service",
    "settings_service",
    "usb_backup_service",
    "usb_format_service",
]
