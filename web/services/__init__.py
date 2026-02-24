"""
WatchMyBirds Services Package.

This package contains service layer classes that encapsulate business logic,
separating it from Flask routes for better testability and maintainability.

ARCHITECTURE RULE:
- Services may ONLY import from core/* modules
- Services MUST NOT import directly from utils/, camera/, detectors/
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
    settings_service,
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
    "settings_service",
]
