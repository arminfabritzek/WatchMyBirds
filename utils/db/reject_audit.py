"""
Reject-audit log — metadata-only record of classifier-rejected detections.

The detection-manager routes detections through ``decide_label`` (see
:mod:`detectors.cls_config`). When the classifier returns
``decision_level='reject'`` we skip image/crop/detection-row persistence
entirely and instead record a single metadata row here.

The audit row carries enough signal to reconstruct *why* a detection
was dropped — bbox position, OD class + confidence, classifier top-1
+ probability, decision context — without needing the frame itself.
Cluster queries against ``bbox_x``/``bbox_y`` then surface static
background objects (the tree-branch / fat-ball FP pattern).

If you later need a visual sample for a specific cluster, flip the
operator-facing audit-capture toggle for a bounded window — the
default stays metadata-only.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any


def insert_reject_audit(conn: sqlite3.Connection, row: dict[str, Any]) -> int:
    """Append one reject-audit row. Best-effort by contract.

    All fields except ``frame_timestamp`` are nullable. Callers should
    pass everything they have; missing keys default to NULL via
    ``row.get()``. The function returns the new ``audit_id`` so callers
    can correlate (e.g. with a log line) but never raises on a partial
    payload — losing one audit row must not block the detection
    pipeline that just paid for the inference.
    """
    cur = conn.execute(
        """
        INSERT INTO reject_audit (
            created_at,
            frame_timestamp,
            frame_width,
            frame_height,
            bbox_x,
            bbox_y,
            bbox_w,
            bbox_h,
            od_class_name,
            od_confidence,
            raw_species_name,
            top1_prob,
            decision_state,
            decision_reasons,
            detector_model_id,
            classifier_model_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            row.get("created_at") or datetime.now(UTC).isoformat(),
            row.get("frame_timestamp"),
            row.get("frame_width"),
            row.get("frame_height"),
            row.get("bbox_x"),
            row.get("bbox_y"),
            row.get("bbox_w"),
            row.get("bbox_h"),
            row.get("od_class_name"),
            row.get("od_confidence"),
            row.get("raw_species_name"),
            row.get("top1_prob"),
            row.get("decision_state"),
            row.get("decision_reasons"),
            row.get("detector_model_id"),
            row.get("classifier_model_id"),
        ),
    )
    conn.commit()
    return int(cur.lastrowid or 0)
