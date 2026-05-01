#!/usr/bin/env python3
"""
Nightly aesthetic auto-tagger for WatchMyBirds.

Computes a CLIP "facing-camera" score on every new detection from the previous
day, writes it to detections.aesthetic_score, and optionally sets is_favorite=1
for the top-N per species per day --- but only for species where the score has
been validated to track human judgement (see
agent_handoff/lab/experiments/aesthetic_tagger/aesthetic_*/ directories).

Pigeons / large birds are intentionally NOT auto-tagged, because validation
showed clip_facing_camera does not generalize to them (AUC 0.35 on 56-image
out-of-sample test set).

Usage (on RPi, run nightly via systemd timer):
    /opt/app/.venv-aesthetic/bin/python /opt/app/scripts/aesthetic_tag_nightly.py
    /opt/app/.venv-aesthetic/bin/python /opt/app/scripts/aesthetic_tag_nightly.py --since 2026-04-29
    /opt/app/.venv-aesthetic/bin/python /opt/app/scripts/aesthetic_tag_nightly.py --dry-run

Design notes:
- Uses a SEPARATE venv from the main app, because torch+open_clip is heavy
  (~1.5 GB) and we don't want to slow down the main detector pipeline. The
  job is offline / non-realtime, so latency doesn't matter.
- Skips detections that already have aesthetic_score populated. Idempotent:
  re-runs are no-ops if all data is fresh.
- Only writes is_favorite=1 (the auto tag) on detections in TAGGABLE_SPECIES.
  All other detections still get an aesthetic_score (for analytics) but no
  favorite flag.
- Existing manual is_favorite=1 (rating_source='manual') is preserved: this
  job only ever SETS rating_source='auto' on detections that don't already
  have a manual favorite.
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --- Configuration ---------------------------------------------------------

# Species filter. Only these CLS labels can become auto-favorites.
#
# Rationale: HUMAN review of 5-day mixed-species output showed that rare
# species (Phoenicurus, Phylloscopus, Sylvia, Aegithalos, Poecile, Passer,
# Turdus_sp.) are mostly mis-classifications -- the CLS stage guesses an
# exotic species on uncertain crops instead of staying on 'unknown'. Tagging
# those is worse than not tagging them. Common, reliably-classified species
# (great tit, blue tit, pigeons) are tagged.
#
# Add a species here only after enough HUMAN validation labels prove the
# CLS classification is reliable for it.
TAGGABLE_SPECIES: set[str] = {
    "Parus_major",          # Kohlmeise (great tit)
    "Cyanistes_caeruleus",  # Blaumeise (blue tit)
    "Columba_palumbus",     # Ringeltaube (pigeon)
}

# Don't tag CLS-rejected detections: 'unknown' often means the classifier
# bailed because the crop was bad. Tagging the "best of the unknowns" leads
# to back-of-bird and partial-bird picks. Re-enable only with evidence.
TAG_UNKNOWN_SPECIES = False

# Minimum aesthetic_score required for auto-tagging. Detections below this
# threshold get a score (for analytics) but no is_favorite flag, even if
# they're top-3 in their bucket. Set to 0.0 to disable the threshold.
# Value 0.15 chosen empirically: most "obviously bad" picks (back of bird,
# motion blur) score below this.
MIN_SCORE_FOR_TAG = 0.15

# Detections must have passed all upstream Pipeline-Stages before the
# aesthetic tagger considers them. The Pi runs:
#   1. detector  (od_class_name='bird', status='active')
#   2. classifier (cls_class_name + cls_confidence)
#   3. decision policy (decision_state in 'confirmed' | 'uncertain' | ...)
# We only tag confirmed detections so that the CLS species name is trusted.
# Set to None to allow all decision states (not recommended in production).
REQUIRED_DECISION_STATE: str | None = "confirmed"

# How many detections per (species, day) to mark as is_favorite_auto.
TOP_N_PER_SPECIES_PER_DAY = 3

# CLIP model + prompt pair. Tuned on agent_handoff/lab/experiments/aesthetic_tagger/aesthetic_sanity.
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
CLIP_PROMPT_POSITIVE = "a bird with its face, head and chest visible toward the viewer"
CLIP_PROMPT_NEGATIVE = "a bird seen from behind, showing only its back or tail"

# Default paths on RPi (overridable via env).
DB_PATH = Path(os.environ.get("WMB_DB_PATH", "/opt/app/data/output/images.db"))
CROPS_ROOT = Path(os.environ.get("WMB_CROPS_ROOT", "/opt/app/data/output/derivatives/thumbs"))
LOG_PATH = Path(os.environ.get("WMB_AESTHETIC_LOG", "/opt/app/data/logs/aesthetic_tag.log"))


def setup_logging(verbose: bool) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH),
            logging.StreamHandler(sys.stdout),
        ],
    )


# --- DB helpers ------------------------------------------------------------

def fetch_unscored_detections(
    conn: sqlite3.Connection,
    since: str,
    limit: int | None = None,
) -> list[dict]:
    """Detections that need scoring: created since `since`, never scored, and
    (if REQUIRED_DECISION_STATE is set) confirmed by the upstream pipeline."""
    where_extra = ""
    params: list = [since]
    if REQUIRED_DECISION_STATE is not None:
        where_extra = " AND d.decision_state = ?"
        params.append(REQUIRED_DECISION_STATE)

    sql = f"""
    SELECT d.detection_id, d.image_filename, d.thumbnail_path, d.created_at,
           COALESCE(c.cls_class_name, 'unknown') AS species,
           d.is_favorite, d.rating_source, d.aesthetic_score, d.aesthetic_score_at,
           d.decision_state
    FROM detections d
    LEFT JOIN classifications c ON c.detection_id = d.detection_id
        AND c.rank = 1 AND c.status = 'active'
    WHERE d.status = 'active'
      AND d.od_class_name = 'bird'
      AND d.created_at >= ?
      AND (d.aesthetic_score IS NULL OR d.aesthetic_score_at IS NULL)
      AND d.thumbnail_path IS NOT NULL
      {where_extra}
    ORDER BY d.created_at DESC
    """
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def write_score(conn: sqlite3.Connection, det_id: int, score: float, ts: str) -> None:
    conn.execute(
        "UPDATE detections SET aesthetic_score = ?, aesthetic_score_at = ? "
        "WHERE detection_id = ?",
        (float(score), ts, int(det_id)),
    )


def apply_auto_favorites(conn: sqlite3.Connection, since: str, dry_run: bool) -> dict:
    """
    For each (species, day) bucket: pick the top N detections by aesthetic_score
    and set is_favorite=1, rating_source='auto'. By default, ALL species are
    eligible (TAGGABLE_SPECIES is empty); set the constant to a non-empty set
    to restrict. Pre-existing manual favorites are never overwritten.

    Detections whose aesthetic_score is below MIN_SCORE_FOR_TAG are excluded
    even if they win their bucket -- this prevents "best-of-a-bad-day" tags
    on species the model couldn't make sense of.
    """
    # Build optional species filter clause.
    if TAGGABLE_SPECIES:
        species_filter = list(TAGGABLE_SPECIES)
        if TAG_UNKNOWN_SPECIES:
            species_filter.append("unknown")
        placeholders = ",".join("?" * len(species_filter))
        species_clause = f"AND COALESCE(c.cls_class_name, 'unknown') IN ({placeholders})"
        species_params: tuple = tuple(species_filter)
    elif not TAG_UNKNOWN_SPECIES:
        species_clause = "AND c.cls_class_name IS NOT NULL"
        species_params = ()
    else:
        species_clause = ""
        species_params = ()

    # Optional decision-state gate: only confirmed detections.
    decision_clause = ""
    decision_params: tuple = ()
    if REQUIRED_DECISION_STATE is not None:
        decision_clause = "AND d.decision_state = ?"
        decision_params = (REQUIRED_DECISION_STATE,)

    sql = f"""
    WITH ranked AS (
      SELECT d.detection_id,
             COALESCE(c.cls_class_name, 'unknown') AS species,
             substr(d.created_at, 1, 10) AS day,
             d.aesthetic_score,
             d.is_favorite,
             d.rating_source,
             ROW_NUMBER() OVER (
               PARTITION BY COALESCE(c.cls_class_name, 'unknown'),
                            substr(d.created_at, 1, 10)
               ORDER BY d.aesthetic_score DESC
             ) AS rn
      FROM detections d
      LEFT JOIN classifications c ON c.detection_id = d.detection_id
          AND c.rank = 1 AND c.status = 'active'
      WHERE d.status = 'active'
        AND d.od_class_name = 'bird'
        AND d.created_at >= ?
        AND d.aesthetic_score IS NOT NULL
        AND d.aesthetic_score >= ?
        {species_clause}
        {decision_clause}
    )
    SELECT detection_id, species, day, aesthetic_score, is_favorite, rating_source
    FROM ranked WHERE rn <= ?
    """
    cur = conn.execute(
        sql,
        (since, MIN_SCORE_FOR_TAG, *species_params, *decision_params, TOP_N_PER_SPECIES_PER_DAY),
    )
    rows = cur.fetchall()

    by_species: dict[str, int] = {}
    skipped_manual = 0
    newly_tagged: list[int] = []

    for det_id, species, day, score, is_fav, source in rows:
        # Don't touch manually-favorited detections.
        if is_fav and source == "manual":
            skipped_manual += 1
            continue
        newly_tagged.append(det_id)
        by_species[species] = by_species.get(species, 0) + 1

    if not dry_run and newly_tagged:
        conn.executemany(
            "UPDATE detections SET is_favorite = 1, rating_source = 'auto' "
            "WHERE detection_id = ?",
            [(d,) for d in newly_tagged],
        )

    return {
        "total_tagged": len(newly_tagged),
        "skipped_manual": skipped_manual,
        "by_species": by_species,
    }


# --- CLIP scoring ----------------------------------------------------------

def load_clip_model(device: str):
    """Lazy import + load. Returns (model, preprocess, text_features)."""
    import open_clip
    import torch

    log = logging.getLogger(__name__)
    log.info(f"Loading CLIP {CLIP_MODEL_NAME} ({CLIP_PRETRAINED}) on {device}...")
    t0 = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=device
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    model.eval()

    # Pre-compute text features once (they're constant).
    with torch.no_grad():
        tokens = tokenizer([CLIP_PROMPT_POSITIVE, CLIP_PROMPT_NEGATIVE]).to(device)
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    log.info(f"CLIP ready in {time.time() - t0:.1f}s")
    return model, preprocess, text_features


def score_image(model, preprocess, text_features, image_path: Path, device: str) -> float:
    """Returns probability that the image matches the positive prompt (0..1)."""
    from PIL import Image
    import torch

    img = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        img_feat = model.encode_image(image_tensor)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        # 100 * cos-sim is standard CLIP scale; softmax over the two prompts.
        logits = (100.0 * img_feat @ text_features.T).softmax(dim=-1)
    return float(logits[0, 0].item())


def resolve_crop_path(thumbnail_path: str, image_filename: str) -> Path | None:
    """Crops live in <CROPS_ROOT>/<YYYY-MM-DD>/<thumbnail_filename>.
    The DB stores only the thumbnail filename, so we derive the date from
    image_filename which starts with YYYYMMDD."""
    if not thumbnail_path or not image_filename:
        return None
    if len(image_filename) < 8:
        return None
    yyyymmdd = image_filename[:8]
    day_dir = f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"
    p = CROPS_ROOT / day_dir / thumbnail_path
    return p if p.exists() else None


# --- Main ------------------------------------------------------------------

def pick_device() -> str:
    """Pi 5 is CPU-only. Detect MPS / CUDA for dev hosts."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def main_with_args(argv: list[str] | None = None) -> int:
    """Entry point usable from tests (pass argv list) or CLI (None = sys.argv)."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--since", default=None,
        help="Earliest created_at (ISO date). Defaults to yesterday 00:00 UTC.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Cap detections processed per run (smoke testing).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Compute scores but do NOT write to DB.",
    )
    p.add_argument(
        "--skip-tagging", action="store_true",
        help="Score only; do NOT update is_favorite. Use to backfill aesthetic_score.",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    setup_logging(args.verbose)
    log = logging.getLogger(__name__)

    if args.since is None:
        # Default: yesterday 00:00 UTC. Catches everything from the prior calendar day.
        since_dt = (datetime.now(timezone.utc) - timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        since = since_dt.isoformat()
    else:
        since = args.since

    log.info(f"Aesthetic tagger starting; since={since}, dry_run={args.dry_run}, "
             f"db={DB_PATH}, crops={CROPS_ROOT}")

    if not DB_PATH.exists():
        log.error(f"DB not found: {DB_PATH}")
        return 2

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA busy_timeout = 30000")  # 30s, in case detector pipeline holds locks

    try:
        unscored = fetch_unscored_detections(conn, since=since, limit=args.limit)
        log.info(f"Found {len(unscored)} detections needing aesthetic_score")

        if not unscored:
            log.info("Nothing to score; exiting.")
            return 0

        device = pick_device()
        model, preprocess, text_features = load_clip_model(device)

        scored = 0
        skipped_missing = 0
        t_start = time.time()

        for i, det in enumerate(unscored, 1):
            crop_path = resolve_crop_path(det["thumbnail_path"], det["image_filename"])
            if crop_path is None:
                skipped_missing += 1
                if skipped_missing <= 5:
                    log.warning(f"crop missing for det {det['detection_id']}: {det['thumbnail_path']}")
                continue

            try:
                score = score_image(model, preprocess, text_features, crop_path, device)
            except Exception as exc:
                log.error(f"score failed for det {det['detection_id']}: {exc!r}")
                continue

            if not args.dry_run:
                write_score(conn, det["detection_id"], score, datetime.now(timezone.utc).isoformat())
                if scored % 50 == 0:
                    conn.commit()

            scored += 1
            if scored % 25 == 0:
                elapsed = time.time() - t_start
                rate = scored / elapsed if elapsed > 0 else 0
                log.info(f"  [{i}/{len(unscored)}] scored={scored}, missing={skipped_missing}, "
                         f"rate={rate:.1f} img/s")

        if not args.dry_run:
            conn.commit()

        log.info(f"Scored {scored} detections in {time.time() - t_start:.1f}s "
                 f"(skipped_missing={skipped_missing})")

        # Tagging step: only after scores are committed.
        if not args.skip_tagging:
            tagging_stats = apply_auto_favorites(conn, since=since, dry_run=args.dry_run)
            if not args.dry_run:
                conn.commit()
            log.info(f"Auto-tagged {tagging_stats['total_tagged']} detections as favorite "
                     f"(skipped {tagging_stats['skipped_manual']} manual favorites)")
            for sp, n in tagging_stats["by_species"].items():
                log.info(f"   {sp}: {n}")
        else:
            log.info("Tagging step skipped (--skip-tagging).")

        return 0

    finally:
        conn.close()


def main() -> int:
    """CLI shim: parses sys.argv via argparse."""
    return main_with_args(None)


if __name__ == "__main__":
    sys.exit(main())
