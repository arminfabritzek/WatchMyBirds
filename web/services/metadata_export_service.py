"""Metadata-export service — thin web wrapper over the core.

All the work (DB reads, XMP build, copy production) lives in
``core.metadata_export_core`` because it needs ``utils.*`` IO, which the
web-service import boundary forbids here (arch_hard). This module only
re-exports the core entry points the routes call.

Shared by both human-download surfaces:
  - detail-modal "Download" (single image)
  - edit-page "Download Selected" (batch ZIP, same helper per entry)
"""

from __future__ import annotations

from core.metadata_export_core import (
    build_event_metadata,
    burn_in_enabled,
    export_filename,
    produce_copy_bytes,
    resolve_image_for_detection,
)

__all__ = [
    "build_event_metadata",
    "burn_in_enabled",
    "export_filename",
    "produce_copy_bytes",
    "resolve_image_for_detection",
]
