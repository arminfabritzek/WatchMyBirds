"""
User-Interaction Ground-Truth Queries.

Every explicit user interaction on the WatchMyBirds web UI is a
ground-truth statement about a frame or detection. This module exposes
four SQL queries that lift those statements out of the live DB into
training-ready row shapes for export to Pipeline-Dev.

The four buckets — and what makes a row qualify:

1. **Hard-Negatives** (``fetch_hard_negatives``):
   - ``images.review_status = 'no_bird'``
   - ``detections.status = 'active'``
   Source: bulk-reject in moderation, "verwerfen" on the Unclear page,
   or a single-image no-bird mark. Every active detection on a
   ``no_bird`` image is a verified false-positive crop.

2. **Confirmed Positives** (``fetch_confirmed_positives``):
   - ``lower(detections.decision_level) = 'species'``
   - ``detections.status = 'active'``
   - ``images.review_status`` IS NULL or != ``'no_bird'`` (defense
     against frames the user later rejected wholesale)
   - **AND** the user has performed an *explicit* species-level
     action on the row — either ``species_source='manual'`` (came
     from ``confirm_unclear_detections``, a Review-Queue re-label,
     or any other manual write site) OR ``manual_species_override``
     is non-empty (user picked a species via the picker).
   Source: species labels the user explicitly authored. The earlier
   filter — "decision_level=species AND user did not contradict" —
   was a **bug**: 95% of `decision_level=species` rows are pure
   pipeline output (``species_source='model_top1'``) and would
   feed the model its own predictions back as ground-truth
   (classic confirmation-bias loop). The strict filter rejects
   those even though it shrinks the bucket dramatically — explicit
   user action is the contract.

3. **Species Re-Labels** (``fetch_species_relabels``):
   - ``detections.manual_species_override IS NOT NULL`` and non-empty
   - and different from ``raw_species_name`` (no-op overrides excluded)
   - ``detections.status = 'active'``
   Source: an explicit user correction via Moderation bulk-relabel or
   Review-Queue edit. The most valuable training signal because it
   simultaneously carries a positive (correct species) and an implicit
   negative (model's wrong guess).

4. **Favorites** (``fetch_favorites``):
   - ``detections.is_favorite = 1``
   - ``detections.rating_source = 'manual'`` (defense against the
     legacy backfill that could otherwise stamp auto-tag rows as
     favorites — see ``trash.py:463`` and ``connection.py:570``)
   - ``detections.status = 'active'``
   Source: explicit heart-click in the gallery UI. The codebase
   treats this as the HUMAN gold-label (``gallery_core.py:542``);
   fav wins over every algorithmic ranking. For training: the
   strongest "this image is exemplary for this species" signal
   the system has — Pipeline-Dev should sampling-weight these
   highest.

Explicitly NOT included (deliberately):
- Trash (``status='deleted'``) — ambiguous: could be FP or "ugly bird".
- Star ratings (``rating > 0``) — 98% are ``rating_source='auto'`` from
  the aesthetic tagger, not a user statement. Favorites are the
  separate explicit channel.
- ``species_review`` / ``reject`` decision_levels — these are
  *pipeline* uncertainty, not user statements.

Every row carries enough provenance for Pipeline-Dev to gate by model
version, time range, and originating user action.

The queries are read-only and never write back to the DB. They return
per-detection row dicts; **frame-integrity** (i.e. dropping frames
whose siblings are not all user-confirmed) is the export service's
responsibility, not this module's — see
``web/services/user_groundtruth_export_service.py::_apply_frame_integrity``.

The export-batches book-keeping lives in a separate service module
(Slice 2 of plan ``2026-05-22_FEATURE_user-groundtruth-export-for-pipeline-dev``).
"""

from __future__ import annotations

import sqlite3
from typing import Any


def fetch_hard_negatives(
    conn: sqlite3.Connection,
    since: str | None = None,
    until: str | None = None,
) -> list[dict[str, Any]]:
    """Return every active detection on an image flagged ``no_bird``.

    Each row is a verified false-positive crop — the box the detector
    drew enclosed something the user explicitly declared not-a-bird.

    Args:
        conn: Open SQLite connection with ``row_factory = sqlite3.Row``.
        since: ISO-8601 timestamp (inclusive). Filters on
            ``images.review_updated_at``. ``None`` means no lower bound.
        until: ISO-8601 timestamp (exclusive). ``None`` means no upper
            bound.

    Returns:
        List of dicts, newest user-action first. Keys:
        - ``detection_id`` (int)
        - ``image_filename`` (str)
        - ``bbox_x``, ``bbox_y``, ``bbox_w``, ``bbox_h`` (float)
        - ``od_confidence`` (float | None)
        - ``od_class_name`` (str | None) — what the detector thought it was
        - ``detector_model_version`` (str | None) — for model-cohort
          filtering downstream
        - ``classifier_model_version`` (str | None)
        - ``frame_width``, ``frame_height`` (int | None)
        - ``user_action_at`` (str | None) — ``images.review_updated_at``
        - ``bucket`` (str) — constant ``"hard_negatives"``
    """
    sql, params = _build_hard_negatives_query(since, until)
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict_hn(r) for r in rows]


def fetch_confirmed_positives(
    conn: sqlite3.Connection,
    since: str | None = None,
    until: str | None = None,
) -> list[dict[str, Any]]:
    """Return detections the user explicitly confirmed as a species.

    A row qualifies if **all** of these hold:
    - ``decision_level='species'`` (pipeline reached species-level)
    - ``status='active'`` (not in trash)
    - parent image is not flagged ``no_bird``
    - the user has *explicitly* acted on the row's species — either
      ``species_source='manual'`` (any manual write site:
      ``confirm_unclear_detections``, Review-Queue re-label, bulk
      relabel) OR ``manual_species_override`` is non-empty.

    The explicit-user-action gate is the central correctness
    guarantee. Without it, ~95% of rows would be pure pipeline
    output (``species_source='model_top1'``), and feeding those back
    as training "ground-truth" creates a confirmation-bias loop where
    the model learns from its own predictions.

    Args:
        conn: Open SQLite connection with row factory.
        since, until: ISO timestamps, see ``fetch_hard_negatives``.

    Returns:
        List of dicts, newest first. Keys (in addition to base):
        - ``species`` (str) — final species label
          (``manual_species_override`` if set, else ``raw_species_name``)
        - ``species_source`` (str | None) — always 'manual'-family
          (the legacy 'model_top1' rows are filtered out)
        - ``user_action_at`` (str | None) — ``species_updated_at``
        - ``bucket`` (str) — constant ``"confirmed_positives"``
    """
    sql, params = _build_confirmed_positives_query(since, until)
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict_cp(r) for r in rows]


def fetch_species_relabels(
    conn: sqlite3.Connection,
    since: str | None = None,
    until: str | None = None,
) -> list[dict[str, Any]]:
    """Return detections where the user corrected the species label.

    A row qualifies if ``manual_species_override`` is non-empty AND
    different from ``raw_species_name`` — pure no-op overrides
    (e.g. operator re-confirming the same species) are excluded so
    Pipeline-Dev's classifier gets only rows with actual disagreement
    signal.

    Args:
        conn: Open SQLite connection with row factory.
        since, until: ISO timestamps on ``species_updated_at``.

    Returns:
        List of dicts, newest first. Keys (in addition to base):
        - ``model_predicted_species`` (str | None) — ``raw_species_name``
        - ``user_corrected_species`` (str) — ``manual_species_override``
        - ``species_source`` (str | None)
        - ``user_action_at`` (str | None) — ``species_updated_at``
        - ``bucket`` (str) — constant ``"species_relabels"``
    """
    sql, params = _build_species_relabels_query(since, until)
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict_rl(r) for r in rows]


def fetch_favorites(
    conn: sqlite3.Connection,
    since: str | None = None,
    until: str | None = None,
) -> list[dict[str, Any]]:
    """Return detections explicitly marked as favorite by the user.

    Filters on **both** ``is_favorite=1`` AND
    ``rating_source='manual'`` to exclude the legacy backfill path
    where the aesthetic tagger could have stamped favorites
    automatically. The combined predicate matches exactly the heart-
    click write site in ``web/blueprints/trash.py:463``.

    Args:
        conn: Open SQLite connection with row factory.
        since, until: ISO timestamps. The detections table has no
            dedicated ``favorited_at`` column today, so the window
            filter falls back to ``species_updated_at`` — the closest
            user-action timestamp the row carries. If a row lacks both
            (very old rows pre-dating the column), it slips through
            unconstrained.

    Returns:
        List of dicts, newest first. Keys (in addition to base):
        - ``species`` (str | None) — final label (override-wins-over-raw)
        - ``species_source`` (str | None)
        - ``user_action_at`` (str | None) — ``species_updated_at``
        - ``bucket`` (str) — constant ``"favorites"``
    """
    sql, params = _build_favorites_query(since, until)
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict_fav(r) for r in rows]


def count_pending_by_bucket(
    conn: sqlite3.Connection,
    since: str | None = None,
    until: str | None = None,
) -> dict[str, int]:
    """Return per-bucket row counts without materializing the data.

    Used by the export-batch preview page so the operator sees
    "X new hard-negatives waiting" without paying the full COCO
    serialization cost. Mirrors the four ``fetch_*`` predicates
    exactly — a divergence here would show stale counts vs the
    actual export, so the queries are kept in sync via the shared
    ``_*_predicate_sql`` helpers below.

    Note: these are **per-detection** counts, before frame-integrity
    filtering. The actual export may drop frames whose siblings are
    not all user-confirmed; the preview intentionally over-reports
    so the operator sees the upper-bound signal volume.
    """
    return {
        "hard_negatives": _count_with(
            conn, _build_hard_negatives_query(since, until)
        ),
        "confirmed_positives": _count_with(
            conn, _build_confirmed_positives_query(since, until)
        ),
        "species_relabels": _count_with(
            conn, _build_species_relabels_query(since, until)
        ),
        "favorites": _count_with(
            conn, _build_favorites_query(since, until)
        ),
    }


# ---------------------------------------------------------------------------
# Query builders — kept separate so count_pending_by_bucket reuses the exact
# same WHERE clause as the fetch_* functions. Wrapping a fetch in `SELECT
# COUNT(*) FROM (...)` is the safe pattern for that.
# ---------------------------------------------------------------------------


def _time_window_clause(
    column: str,
    since: str | None,
    until: str | None,
) -> tuple[str, list[Any]]:
    """Build an optional ``AND <column> >= ? AND <column> < ?`` clause.

    Both bounds are optional. ``since`` is inclusive, ``until`` exclusive
    (half-open intervals compose cleanly across consecutive batches:
    today's ``until`` is tomorrow's ``since``).
    """
    parts: list[str] = []
    params: list[Any] = []
    if since is not None:
        parts.append(f" AND {column} >= ?")
        params.append(since)
    if until is not None:
        parts.append(f" AND {column} < ?")
        params.append(until)
    return "".join(parts), params


def _build_hard_negatives_query(
    since: str | None,
    until: str | None,
) -> tuple[str, list[Any]]:
    window, params = _time_window_clause("i.review_updated_at", since, until)
    sql = f"""
        SELECT
            d.detection_id,
            d.image_filename,
            d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h,
            d.od_confidence,
            d.od_class_name,
            d.detector_model_version,
            d.classifier_model_version,
            d.frame_width,
            d.frame_height,
            i.review_updated_at AS user_action_at
        FROM detections d
        JOIN images i ON i.filename = d.image_filename
        WHERE d.status = 'active'
          AND i.review_status = 'no_bird'
          {window}
        ORDER BY i.review_updated_at DESC, d.detection_id DESC
    """
    return sql, params


def _build_confirmed_positives_query(
    since: str | None,
    until: str | None,
) -> tuple[str, list[Any]]:
    window, params = _time_window_clause("d.species_updated_at", since, until)
    # Explicit-user-action gate: either species_source is 'manual'
    # (any manual write site stamps that — confirm_unclear_detections,
    # bulk-relabel, review-queue edit) OR manual_species_override is
    # non-empty (user actively picked a species via the picker).
    # Without this gate, pure pipeline predictions
    # (species_source='model_top1') would leak through and create a
    # confirmation-bias loop in re-training.
    sql = f"""
        SELECT
            d.detection_id,
            d.image_filename,
            d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h,
            d.od_confidence,
            d.od_class_name,
            d.detector_model_version,
            d.classifier_model_version,
            d.frame_width,
            d.frame_height,
            COALESCE(
                NULLIF(d.manual_species_override, ''),
                d.raw_species_name
            ) AS species,
            d.species_source,
            d.species_updated_at AS user_action_at
        FROM detections d
        JOIN images i ON i.filename = d.image_filename
        WHERE d.status = 'active'
          AND lower(COALESCE(d.decision_level, '')) = 'species'
          AND (i.review_status IS NULL OR i.review_status != 'no_bird')
          AND (
              lower(COALESCE(d.species_source, '')) = 'manual'
              OR (
                  d.manual_species_override IS NOT NULL
                  AND d.manual_species_override != ''
              )
          )
          {window}
        ORDER BY d.species_updated_at DESC, d.detection_id DESC
    """
    return sql, params


def _build_species_relabels_query(
    since: str | None,
    until: str | None,
) -> tuple[str, list[Any]]:
    window, params = _time_window_clause("d.species_updated_at", since, until)
    sql = f"""
        SELECT
            d.detection_id,
            d.image_filename,
            d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h,
            d.od_confidence,
            d.od_class_name,
            d.detector_model_version,
            d.classifier_model_version,
            d.frame_width,
            d.frame_height,
            d.raw_species_name AS model_predicted_species,
            d.manual_species_override AS user_corrected_species,
            d.species_source,
            d.species_updated_at AS user_action_at
        FROM detections d
        WHERE d.status = 'active'
          AND d.manual_species_override IS NOT NULL
          AND d.manual_species_override != ''
          AND d.manual_species_override != COALESCE(d.raw_species_name, '')
          {window}
        ORDER BY d.species_updated_at DESC, d.detection_id DESC
    """
    return sql, params


def _build_favorites_query(
    since: str | None,
    until: str | None,
) -> tuple[str, list[Any]]:
    # Window predicate on species_updated_at — the heart-click stamps
    # rating_source='manual' but reuses the species_updated_at column
    # for "last user touched this row" semantics. If the row was never
    # touched on the species axis, species_updated_at is NULL and the
    # window predicate naturally rejects it under SQLite's NULL-vs-
    # comparison semantics (which is the conservative default —
    # an unwindowed favorites query has since=None and gets every fav).
    window, params = _time_window_clause("d.species_updated_at", since, until)
    sql = f"""
        SELECT
            d.detection_id,
            d.image_filename,
            d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h,
            d.od_confidence,
            d.od_class_name,
            d.detector_model_version,
            d.classifier_model_version,
            d.frame_width,
            d.frame_height,
            COALESCE(
                NULLIF(d.manual_species_override, ''),
                d.raw_species_name
            ) AS species,
            d.species_source,
            d.species_updated_at AS user_action_at
        FROM detections d
        WHERE d.status = 'active'
          AND d.is_favorite = 1
          AND d.rating_source = 'manual'
          {window}
        ORDER BY d.species_updated_at DESC, d.detection_id DESC
    """
    return sql, params


def _count_with(
    conn: sqlite3.Connection,
    query_and_params: tuple[str, list[Any]],
) -> int:
    sql, params = query_and_params
    wrapped = f"SELECT COUNT(*) AS n FROM ({sql})"
    row = conn.execute(wrapped, params).fetchone()
    if row is None:
        return 0
    # sqlite3.Row supports both index and key access; plain tuple via index.
    try:
        return int(row["n"])
    except (TypeError, IndexError, KeyError):
        return int(row[0])


# ---------------------------------------------------------------------------
# Row-to-dict converters — kept separate per bucket so the bucket constant
# and any per-bucket post-processing live in exactly one place.
# ---------------------------------------------------------------------------


def _common_fields(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "detection_id": int(row["detection_id"]),
        "image_filename": row["image_filename"],
        "bbox_x": _maybe_float(row["bbox_x"]),
        "bbox_y": _maybe_float(row["bbox_y"]),
        "bbox_w": _maybe_float(row["bbox_w"]),
        "bbox_h": _maybe_float(row["bbox_h"]),
        "od_confidence": _maybe_float(row["od_confidence"]),
        "od_class_name": row["od_class_name"],
        "detector_model_version": row["detector_model_version"],
        "classifier_model_version": row["classifier_model_version"],
        "frame_width": _maybe_int(row["frame_width"]),
        "frame_height": _maybe_int(row["frame_height"]),
        "user_action_at": row["user_action_at"],
    }


def _row_to_dict_hn(row: sqlite3.Row) -> dict[str, Any]:
    out = _common_fields(row)
    out["bucket"] = "hard_negatives"
    return out


def _row_to_dict_cp(row: sqlite3.Row) -> dict[str, Any]:
    out = _common_fields(row)
    out["species"] = row["species"]
    out["species_source"] = row["species_source"]
    out["bucket"] = "confirmed_positives"
    return out


def _row_to_dict_rl(row: sqlite3.Row) -> dict[str, Any]:
    out = _common_fields(row)
    out["model_predicted_species"] = row["model_predicted_species"]
    out["user_corrected_species"] = row["user_corrected_species"]
    out["species_source"] = row["species_source"]
    out["bucket"] = "species_relabels"
    return out


def _row_to_dict_fav(row: sqlite3.Row) -> dict[str, Any]:
    out = _common_fields(row)
    out["species"] = row["species"]
    out["species_source"] = row["species_source"]
    out["bucket"] = "favorites"
    return out


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
