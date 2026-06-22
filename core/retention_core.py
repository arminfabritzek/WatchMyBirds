"""Artifact lifecycle / retention — Policy, Planner, Executor (V1).

V1 scope: full-resolution **originals only**. Originals are never modified,
only whole-file deleted, and only when their display derivatives already
exist. Database rows and all derivatives (optimized/thumbs/crops/preview)
are preserved.

Layering: this module is the only retention code that imports `utils.*`
(allowed for core/* under H-02). The web service wrapper imports this
module and nothing from utils (H-01 enforcement-test rule).
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from core import user_groundtruth_core
from utils.db import retention as retention_db
from utils.path_manager import PathManager

Decision = tuple[str, str | None]

VALID_POSTURES = ("off", "conservative", "reclaim")


def resolve_posture_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Resolve the operator-facing posture into the V1 boolean knobs.

    ``RETENTION_POSTURE`` is the authority. When absent (legacy installs), it
    derives from ``RETENTION_ENABLED``: false -> off, true -> conservative.
    An explicit posture overrides the legacy flag; an unknown value falls
    back to the conservative (safe) posture.

    Returns a NEW settings dict with ``RETENTION_ENABLED`` and
    ``RETENTION_PROTECT_UNREVIEWED`` set so the unchanged ``decide`` and
    Planner consume posture without knowing it exists.
    """
    raw = settings.get("RETENTION_POSTURE", None)
    posture = str(raw or "").strip().lower()
    if raw is None or posture == "":
        # Legacy install with no posture: derive from the boolean flag.
        posture = (
            "off" if not settings.get("RETENTION_ENABLED", False) else "conservative"
        )
    elif posture not in VALID_POSTURES:
        # Explicit-but-invalid value falls back to the safe posture.
        posture = "conservative"

    resolved = dict(settings)
    if posture == "off":
        resolved["RETENTION_ENABLED"] = False
    elif posture == "reclaim":
        resolved["RETENTION_ENABLED"] = True
        resolved["RETENTION_PROTECT_UNREVIEWED"] = False
    else:  # conservative
        resolved["RETENTION_ENABLED"] = True
        resolved["RETENTION_PROTECT_UNREVIEWED"] = True
    resolved["RETENTION_POSTURE"] = posture
    return resolved


def decide(facts: dict[str, Any], settings: dict[str, Any]) -> Decision:
    """Pure retention decision for a single original.

    Returns ("delete", None) when the original is eligible for removal, or
    ("protect", reason) with the first failing rule's reason otherwise.

    `facts` must carry: age_days, original_present, derivatives_present,
    is_favorite, export_relevant, review_status. `settings` carries the
    RETENTION_* knobs. No IO is performed here.
    """
    # P0 — master switch.
    if not settings.get("RETENTION_ENABLED", False):
        return "protect", "disabled"

    # P1 — original still on disk (per the DB marker).
    if not facts.get("original_present", 1):
        return "protect", "already_deleted"

    # P2 — older than the retention window (strictly greater).
    window = int(settings.get("RETENTION_DAYS", 90))
    if int(facts.get("age_days", 0)) <= window:
        return "protect", "too_recent"

    # P3 — display derivatives must already exist (immutability safety
    # valve: the original is the only regeneration source).
    if not facts.get("derivatives_present", False):
        return "protect", "missing_derivative"

    # P5 — export-relevant training data is never silently deletable
    # (checked before the configurable protections; it has no off-switch).
    if facts.get("export_relevant", False):
        return "protect", "export_relevant"

    # P4 — favourites, protected by default.
    if settings.get("RETENTION_PROTECT_FAVORITES", True) and facts.get(
        "is_favorite", False
    ):
        return "protect", "favorite"

    # P6 — unreviewed images, conservatively protected by default.
    if settings.get("RETENTION_PROTECT_UNREVIEWED", True):
        review_status = facts.get("review_status")
        if review_status in (None, "", "untagged"):
            return "protect", "unreviewed"

    return "delete", None


# ---------------------------------------------------------------------------
# Planner (read-only) — the dry-run engine.
# ---------------------------------------------------------------------------


@dataclass
class RetentionPlan:
    deletable: list[dict[str, Any]] = field(default_factory=list)
    protected_counts: dict[str, int] = field(default_factory=dict)
    estimated_bytes: int = 0


def _age_days(timestamp: str, now: dt.datetime) -> int:
    """Whole days between a `YYYYMMDD_HHMMSS` capture stamp and `now`.

    Unparseable stamps are treated as age 0 (too_recent → protected),
    the conservative direction.
    """
    if not timestamp or len(timestamp) < 8:
        return 0
    try:
        captured = dt.datetime.strptime(timestamp[:15], "%Y%m%d_%H%M%S")
    except ValueError:
        try:
            captured = dt.datetime.strptime(timestamp[:8], "%Y%m%d")
        except ValueError:
            return 0
    captured = captured.replace(tzinfo=dt.UTC)
    return max(0, (now - captured).days)


def _derivatives_present(path_mgr, filename: str, thumb_names: list[str]) -> bool:
    """The display derivatives that replace the original exist on disk.

    P3 safety valve: the original is the only regeneration source, so we
    only delete it once the artifacts the app serves instead are present —
    the optimized WebP AND at least one real thumbnail. Thumbnails are
    crop-keyed in production (``<stem>_crop_N.webp``, stored in
    ``detections.thumbnail_path``); ``thumb_names`` carries those exact
    names resolved from the DB. Orphan images with no active detection fall
    back to the preview thumb.
    """
    optimized = path_mgr.get_derivative_path(filename, "optimized")
    if not optimized.exists():
        return False

    thumbs_dir = path_mgr.thumbs_dir / path_mgr.get_date_folder(
        path_mgr.extract_date_from_filename(filename)
    )
    if any((thumbs_dir / name).exists() for name in thumb_names):
        return True

    # Orphan fallback: the preview thumb is the display derivative when no
    # crop thumb exists.
    return path_mgr.get_preview_thumb_path(filename).exists()


def build_plan(
    conn,
    output_dir: str,
    settings: dict[str, Any],
    now: dt.datetime | None = None,
) -> RetentionPlan:
    """Partition candidate originals into deletable vs protected.

    Read-only: stats files and reads the DB, performs no writes/deletes.
    """
    if now is None:
        now = dt.datetime.now(dt.UTC)

    # Construct directly (not the cached get_path_manager singleton) so the
    # planner is an honest function of the output_dir argument.
    path_mgr = PathManager(output_dir)

    # Pre-filter to originals already past the window (day-resolution prefix;
    # the per-row age check below refines it). Rows newer than this can never
    # be deletable, so excluding them keeps the planner off the full table.
    window = int(settings.get("RETENTION_DAYS", 90))
    cutoff = (now - dt.timedelta(days=window)).strftime("%Y%m%d")
    candidates = retention_db.iter_candidate_images(conn, cutoff_prefix=cutoff)

    # Resolve export-relevance and the per-image thumbnail names once for the
    # whole candidate set (one query each), rather than per row.
    filenames = [c["filename"] for c in candidates]
    export_relevant = user_groundtruth_core.is_export_relevant_any(conn, filenames)
    thumb_names = retention_db.thumbnail_names_for_images(conn, filenames)

    plan = RetentionPlan()
    protected: Counter[str] = Counter()

    for cand in candidates:
        filename = cand["filename"]
        age_days = _age_days(cand["timestamp"], now)
        facts = {
            "age_days": age_days,
            "original_present": cand["original_present"],
            "derivatives_present": _derivatives_present(
                path_mgr, filename, thumb_names.get(filename, [])
            ),
            "is_favorite": cand["is_favorite"],
            "export_relevant": filename in export_relevant,
            "review_status": cand["review_status"],
        }
        action, reason = decide(facts, settings)
        if action == "delete":
            original_path = path_mgr.get_original_path(filename)
            try:
                size = original_path.stat().st_size
            except OSError:
                size = 0
            plan.deletable.append(
                {"filename": filename, "bytes": size, "age_days": age_days}
            )
            plan.estimated_bytes += size
        else:
            protected[reason or "unknown"] += 1

    plan.protected_counts = dict(protected)
    return plan


# ---------------------------------------------------------------------------
# Executor — file delete first, then DB status/log. Per-file commit.
# ---------------------------------------------------------------------------


def execute_plan(
    conn,
    output_dir: str,
    settings: dict[str, Any],
    now: dt.datetime | None = None,
) -> dict[str, int]:
    """Delete deletable originals and record the outcome.

    Order per file: ``_safe_delete`` (file) FIRST, then mark the DB
    (original_present=0 + original_deleted_at) and commit. A missing file
    is recorded as "already gone" and still updates the DB — a missing
    file never blocks the status write. An ``_safe_delete`` "error"
    (e.g. out-of-OUTPUT_DIR, unlink failure) is counted and skips only
    that file's DB update.

    Returns {"deleted", "freed_bytes", "missing", "errors"}.
    """
    # Imported lazily so the pure Policy half of this module has no IO deps.
    from pathlib import Path

    from utils.file_gc import _safe_delete

    plan = build_plan(conn, output_dir, settings, now=now)
    path_mgr = PathManager(output_dir)
    out_path = Path(output_dir)

    deleted = 0
    freed_bytes = 0
    missing = 0
    errors = 0

    for item in plan.deletable:
        filename = item["filename"]
        original_path = path_mgr.get_original_path(filename)
        result = _safe_delete(original_path, out_path)

        if result in ("error", "skipped"):
            # Never touched the file -> do not mark the DB as deleted.
            errors += 1
            continue

        # "deleted" or "missing": record the original as gone and commit.
        retention_db.mark_original_deleted(conn, filename, now_iso(now))
        conn.commit()

        if result == "deleted":
            deleted += 1
            freed_bytes += int(item.get("bytes", 0))
        elif result == "missing":
            missing += 1

    return {
        "deleted": deleted,
        "freed_bytes": freed_bytes,
        "missing": missing,
        "errors": errors,
    }


def now_iso(now: dt.datetime | None = None) -> str:
    """ISO-8601 UTC timestamp for the deletion marker."""
    if now is None:
        now = dt.datetime.now(dt.UTC)
    return now.isoformat()


# Conn-opening entry points. The web service delegates here so it never
# imports utils/opens a connection itself (H-01 enforcement-test rule).
_PROTECTION_KEYS = (
    "too_recent",
    "missing_derivative",
    "favorite",
    "unreviewed",
    "export_relevant",
    "already_deleted",
    "disabled",
)


def _settings_from_config() -> dict[str, Any]:
    from config import get_config

    cfg = get_config()
    return resolve_posture_settings(
        {
            "RETENTION_POSTURE": cfg.get("RETENTION_POSTURE"),
            "RETENTION_ENABLED": cfg.get("RETENTION_ENABLED", False),
            "RETENTION_DAYS": cfg.get("RETENTION_DAYS", 90),
            "RETENTION_PROTECT_FAVORITES": cfg.get("RETENTION_PROTECT_FAVORITES", True),
            "RETENTION_PROTECT_UNREVIEWED": cfg.get(
                "RETENTION_PROTECT_UNREVIEWED", True
            ),
        }
    )


def preview() -> dict[str, Any]:
    """Dry-run: counts, reclaimable bytes, protection breakdown. No IO writes."""
    from config import get_config
    from utils.db import closing_connection

    cfg = get_config()
    output_dir = str(cfg["OUTPUT_DIR"])
    settings = _settings_from_config()

    with closing_connection() as conn:
        plan = build_plan(conn, output_dir, settings)

    protected = {k: plan.protected_counts.get(k, 0) for k in _PROTECTION_KEYS}
    return {
        "posture": settings.get("RETENTION_POSTURE", "conservative"),
        "enabled": bool(settings["RETENTION_ENABLED"]),
        "retention_days": int(settings["RETENTION_DAYS"]),
        "deletable": {
            "count": len(plan.deletable),
            "estimated_bytes": plan.estimated_bytes,
        },
        "protected": protected,
        "sample": plan.deletable[:20],
    }


def run() -> dict[str, int]:
    """Execute retention against the live DB/output dir."""
    from config import get_config
    from utils.db import closing_connection

    cfg = get_config()
    output_dir = str(cfg["OUTPUT_DIR"])
    settings = _settings_from_config()

    with closing_connection() as conn:
        return execute_plan(conn, output_dir, settings)


def is_original_retention_deleted(filename: str) -> bool:
    """True iff the image row exists and its original was removed by retention.

    Reflects the permanent ``original_present = 0`` marker, independent of
    whether retention is currently enabled (enable -> delete -> disable must
    still report 410). The serving layer keeps this off the hot path by only
    calling it when the original file is actually missing on disk.
    """
    from utils.db import closing_connection

    with closing_connection() as conn:
        row = conn.execute(
            "SELECT original_present FROM images WHERE filename = ?",
            (filename,),
        ).fetchone()
    return row is not None and not row["original_present"]
