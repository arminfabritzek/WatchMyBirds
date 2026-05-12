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
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Redirect HF cache to a writable location BEFORE huggingface_hub gets
# imported transitively via open_clip. huggingface_hub.constants reads
# HF_HOME / XDG_CACHE_HOME at module-import time and freezes the resolved
# cache path into module globals — setting HF_HOME after the first import
# has no effect. Must run before any `import open_clip`, hence module-top.
# Container deploys set XDG_CACHE_HOME=/tmp/fontconfig (fontconfig workaround)
# which the runtime user can't write to. Root the cache inside OUTPUT_DIR
# so it sits on the mounted volume and survives container rebuilds.
if not os.environ.get("HF_HOME"):
    _hf_output_dir = os.environ.get("OUTPUT_DIR", "/opt/app/data/output")
    os.environ["HF_HOME"] = os.path.join(_hf_output_dir, "huggingface")

# --- Configuration ---------------------------------------------------------

# Species filter. Only these CLS labels can become auto-favorites
# (i.e. get is_gallery_eligible=1 for the top-N per day). All species
# still get an aesthetic_score written for analytics — this set only
# gates the gallery-pick decision.
#
# Currently empty: temporarily allowing every CLS-labelled species
# through so per-species pick quality can be evaluated across the
# full classifier output, not just the validated three. Revert to a
# conservative subset (e.g. {Parus_major, Cyanistes_caeruleus,
# Columba_palumbus}) if the rare-species CLS hallucination problem
# returns — Phoenicurus, Phylloscopus, Sylvia, Aegithalos, Poecile,
# Passer, Turdus_sp. were mostly mis-classifications in the original
# review.
#
# Add a species back to a non-empty set only after enough validation
# labels prove the CLS classification is reliable for it.
TAGGABLE_SPECIES: set[str] = set()
# Conservative reference for restore:
# TAGGABLE_SPECIES: set[str] = {
#     "Parus_major",          # Kohlmeise (great tit)
#     "Cyanistes_caeruleus",  # Blaumeise (blue tit)
#     "Columba_palumbus",     # Ringeltaube (pigeon)
# }

# Don't tag CLS-rejected detections: 'unknown' often means the classifier
# bailed because the crop was bad. Tagging the "best of the unknowns" leads
# to back-of-bird and partial-bird picks. Re-enable only with evidence.
#
# This stays False even with TAGGABLE_SPECIES empty: "all CLS-labelled
# species" still excludes unknown. The combination (empty TAGGABLE +
# False UNKNOWN) maps to the SQL clause `AND c.cls_class_name IS NOT
# NULL` — every species the classifier could put a confident name on,
# but nothing it punted to 'unknown'.
TAG_UNKNOWN_SPECIES = False

# Minimum aesthetic_score required for auto-tagging. Detections below this
# threshold get a score (for analytics) but no is_gallery_eligible flag,
# even if they're top-3 in their bucket. Set to 0.0 to disable the threshold.
#
# Calibration history (no dates — see git log for those):
# - 0.15 was tuned against an older "facing camera" prompt pair where
#   genuine picks routinely scored 0.5–0.7. It caught only obvious junk.
# - 0.30 was attempted once the prompts began combining pose AND
#   sharpness; the sharper prompts compress the distribution downward
#   so even the best pick of the day topped out around 0.28, producing
#   zero auto-tags on otherwise good days.
# - 0.20 was calibrated against an earlier prompt pair that included
#   "in profile" in the negative — that pair produced a wider score
#   spread because rear-vs-front separation was looser.
# - An earlier 0.03 threshold was tuned on a smaller lab pool with a
#   narrower score distribution; on the wider live distribution it
#   let too many candidates through and the floor wasn't doing useful
#   work. The per-species top-3 logic downstream still filtered, but
#   the floor was no longer load-bearing.
# - 0.10 is calibrated against the current live CLIP-rescore
#   distribution: every HUMAN-favorited pick scores at or above it,
#   so the floor doesn't lose preferred picks while still cutting the
#   candidate pool roughly in half. Should track TELEGRAM_MIN_AESTHETIC_SCORE.
MIN_SCORE_FOR_TAG = 0.10

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

# CLIP model + prompt pair. Tuned for "cute/funny bird portrait" picks:
# the discriminating signal is whether the bird's EYE is visible and the
# head is turned toward the camera. Strict side-profile shots with the
# eye toward the lens are acceptable; only fully-rear / head-turned-away
# poses should be penalised. The word "profile" is deliberately avoided
# in both prompts — it conflates two cases the operator wants to treat
# differently (eye-visible side view = good, eye-hidden rear view = bad).
# Both prompts retain a sharpness term so a sharp 3/4 view ranks above
# a blurry frontal one. Existing per-species AUCs from earlier Lab runs
# (agent_handoff/lab/experiments/aesthetic_tagger/aesthetic_sanity) are
# no longer directly applicable; expect MIN_SCORE_FOR_TAG and
# TELEGRAM_MIN_AESTHETIC_SCORE to need re-calibration once a few days
# of fresh picks are in.
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
CLIP_PROMPT_POSITIVE = "a sharp, well-focused close-up portrait of a bird with its eye visible and its head turned toward the camera"
CLIP_PROMPT_NEGATIVE = "a blurry or back-facing photo of a bird showing its back, tail, or the back of its head, with the eye hidden and the face turned away from the camera"

# Default paths. Precedence per knob: explicit WMB_* env var → derived
# from the app's canonical OUTPUT_DIR (works on Pi systemd, NAS Docker
# `/output`, and local `./data/output` alike) → legacy `/opt/app/data`
# fallback for old systemd units that pre-date OUTPUT_DIR awareness.
# Logs live inside OUTPUT_DIR (not as a sibling) so the container deploy
# only has to mount one volume.
_OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/opt/app/data/output"))
DB_PATH = Path(os.environ.get("WMB_DB_PATH", str(_OUTPUT_DIR / "images.db")))
CROPS_ROOT = Path(
    os.environ.get("WMB_CROPS_ROOT", str(_OUTPUT_DIR / "derivatives" / "thumbs"))
)
LOG_PATH = Path(
    os.environ.get("WMB_AESTHETIC_LOG", str(_OUTPUT_DIR / "logs" / "aesthetic_tag.log"))
)


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
    *,
    rescore: bool = False,
    species_filter: str | None = None,
    per_species_cap: int | None = None,
) -> list[dict]:
    """Detections that need scoring: created since `since`, never scored, and
    (if REQUIRED_DECISION_STATE is set) confirmed by the upstream pipeline.

    With ``rescore=True``, the ``aesthetic_score IS NULL`` filter is dropped
    so already-scored detections are re-evaluated. Used when prompts or
    the threshold change and we want the existing data brought up to date.

    With ``species_filter`` set to a CLS class name (e.g. ``"Parus_major"``),
    only that species is considered. Useful for targeted re-scoring.

    With ``per_species_cap`` set, returns at most N detections per CLS
    species, ranked within each species by ``score DESC, bbox_quality
    DESC, created_at DESC`` — i.e. the detector's own best guesses first.
    Used by the pre-Telegram bridge to keep the run bounded; pairs with
    ``rescore=False`` so the bridge only fills gaps the nightly missed.
    Unknown (CLS-null) detections are excluded from this path because
    ``apply_auto_favorites`` would discard their scores under the default
    ``TAG_UNKNOWN_SPECIES=False`` policy. Mutually exclusive with
    ``limit`` (global LIMIT would defeat the per-species fairness).
    """
    if per_species_cap is not None and limit is not None:
        raise ValueError("pass per_species_cap or limit, not both")

    where_clauses: list[str] = []
    params: list = [since]
    if REQUIRED_DECISION_STATE is not None:
        where_clauses.append("AND d.decision_state = ?")
        params.append(REQUIRED_DECISION_STATE)
    if species_filter is not None:
        where_clauses.append("AND COALESCE(c.cls_class_name, 'unknown') = ?")
        params.append(species_filter)

    score_filter = (
        ""
        if rescore
        else "AND (d.aesthetic_score IS NULL OR d.aesthetic_score_at IS NULL)"
    )

    if per_species_cap is not None:
        # Bridge-only path: rank detections per CLS species and keep top N.
        # Exclude unknown (CLS-null) because their scores are unusable
        # downstream under TAG_UNKNOWN_SPECIES=False.
        sql = f"""
        WITH eligible AS (
          SELECT d.detection_id, d.image_filename, d.thumbnail_path, d.created_at,
                 c.cls_class_name AS species,
                 d.is_favorite, d.rating_source, d.aesthetic_score, d.aesthetic_score_at,
                 d.decision_state,
                 ROW_NUMBER() OVER (
                   PARTITION BY c.cls_class_name
                   ORDER BY COALESCE(d.score, 0) DESC,
                            COALESCE(d.bbox_quality, 0) DESC,
                            d.created_at DESC
                 ) AS rn
          FROM detections d
          LEFT JOIN classifications c ON c.detection_id = d.detection_id
              AND c.rank = 1 AND c.status = 'active'
          WHERE d.status = 'active'
            AND d.od_class_name = 'bird'
            AND d.created_at >= ?
            {score_filter}
            AND d.thumbnail_path IS NOT NULL
            AND (d.decision_level IS NULL OR lower(d.decision_level) != 'reject')
            AND c.cls_class_name IS NOT NULL
            {" ".join(where_clauses)}
        )
        SELECT detection_id, image_filename, thumbnail_path, created_at,
               species, is_favorite, rating_source, aesthetic_score,
               aesthetic_score_at, decision_state
        FROM eligible
        WHERE rn <= ?
        """
        params.append(int(per_species_cap))
    else:
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
          {score_filter}
          AND d.thumbnail_path IS NOT NULL
          AND (d.decision_level IS NULL OR lower(d.decision_level) != 'reject')
          {" ".join(where_clauses)}
        ORDER BY d.created_at DESC
        """
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]


def write_score(conn: sqlite3.Connection, det_id: int, score: float, ts: str) -> None:
    conn.execute(
        "UPDATE detections SET aesthetic_score = ?, aesthetic_score_at = ? "
        "WHERE detection_id = ?",
        (float(score), ts, int(det_id)),
    )


def apply_auto_favorites(conn: sqlite3.Connection, since: str, dry_run: bool) -> dict:
    """
    For each (species, day) bucket: pick the top N detections by aesthetic_score
    and set is_gallery_eligible=1. By default, ALL species are eligible
    (TAGGABLE_SPECIES is empty); set the constant to a non-empty set to restrict.

    is_favorite (HUMAN gold-label) is NEVER touched by this job. The two
    columns are deliberately decoupled:
      - is_favorite      = HUMAN clicked the heart, used as training label
      - is_gallery_eligible = model-decided gallery candidate, badged in UI

    Re-tagging is idempotent: rows already at is_gallery_eligible=1 simply
    stay at 1. We do not unset stale eligibles in this pass — that policy
    (decay / re-evaluation) is a follow-up question.

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
        species_clause = (
            f"AND COALESCE(c.cls_class_name, 'unknown') IN ({placeholders})"
        )
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
             d.is_gallery_eligible,
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
        AND (d.decision_level IS NULL OR lower(d.decision_level) != 'reject')
        {species_clause}
        {decision_clause}
    )
    SELECT detection_id, species, day, aesthetic_score, is_gallery_eligible
    FROM ranked WHERE rn <= ?
    """
    cur = conn.execute(
        sql,
        (
            since,
            MIN_SCORE_FOR_TAG,
            *species_params,
            *decision_params,
            TOP_N_PER_SPECIES_PER_DAY,
        ),
    )
    rows = cur.fetchall()

    by_species: dict[str, int] = {}
    already_eligible = 0
    newly_tagged: list[int] = []

    for det_id, species, _day, _score, was_eligible in rows:
        # Already model-picked → no UPDATE needed, but still count in by_species
        # for an honest "what would this run pick?" report.
        if was_eligible:
            already_eligible += 1
        else:
            newly_tagged.append(det_id)
        by_species[species] = by_species.get(species, 0) + 1

    if not dry_run and newly_tagged:
        conn.executemany(
            "UPDATE detections SET is_gallery_eligible = 1 WHERE detection_id = ?",
            [(d,) for d in newly_tagged],
        )

    return {
        "total_tagged": len(newly_tagged),
        "already_eligible": already_eligible,
        "by_species": by_species,
    }


# --- CLIP scoring ----------------------------------------------------------


def load_clip_model(device: str):
    """Lazy import + load. Returns (model, preprocess, text_features).

    Note: HF cache redirection happens at module-top, not here — it must
    occur before huggingface_hub's transitive import freezes the cache
    path into its constants module.
    """
    import open_clip
    import torch

    log = logging.getLogger(__name__)
    log.info(f"Loading CLIP {CLIP_MODEL_NAME} ({CLIP_PRETRAINED}) on {device}...")
    log.debug(f"HF_HOME={os.environ.get('HF_HOME')}")
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


def score_image(
    model, preprocess, text_features, image_path: Path, device: str
) -> float:
    """Returns probability that the image matches the positive prompt (0..1)."""
    import torch
    from PIL import Image

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
        # torch is an optional extra; nightly script can still log on CPU.
        pass
    return "cpu"


def main_with_args(argv: list[str] | None = None) -> int:
    """Entry point usable from tests (pass argv list) or CLI (None = sys.argv)."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--since",
        default=None,
        help="Earliest created_at (ISO date). Defaults to yesterday 00:00 UTC.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap detections processed per run (smoke testing). Mutually "
        "exclusive with --per-species-cap.",
    )
    p.add_argument(
        "--per-species-cap",
        type=int,
        default=None,
        help="Score at most N detections per CLS species, ranked within "
        "each species by detector score / bbox quality / created_at. "
        "Used by the pre-Telegram bridge to bound the run while "
        "keeping a fair sample across species. Mutually exclusive "
        "with --limit. Excludes CLS-null (unknown) detections.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute scores but do NOT write to DB.",
    )
    p.add_argument(
        "--skip-tagging",
        action="store_true",
        help="Score only; do NOT update is_favorite. Use to backfill aesthetic_score.",
    )
    p.add_argument(
        "--rescore",
        action="store_true",
        help="Re-score detections that already have a score. Use after prompt or "
        "threshold changes — the regular run skips already-scored detections "
        "for idempotency. Combine with --since to limit scope (e.g. only "
        "today). Resets is_gallery_eligible for the affected detections "
        "before re-scoring so the top-N pick is recomputed cleanly.",
    )
    p.add_argument(
        "--species",
        default=None,
        help="Restrict to a single CLS class (e.g. Parus_major). Useful with "
        "--rescore for targeted fixes.",
    )
    p.add_argument(
        "--throttle-ms",
        type=int,
        default=None,
        help="Sleep N milliseconds between each CLIP inference. Default 0 "
        "(no throttling) — bumps to ~100 ms relieve CPU pressure on "
        "the live detector during pre-Telegram bridge runs. Env "
        "override: WMB_AESTHETIC_THROTTLE_MS.",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    if args.limit is not None and args.per_species_cap is not None:
        p.error("--limit and --per-species-cap are mutually exclusive")

    # Throttle: CLI overrides env overrides default 0.
    if args.throttle_ms is not None:
        throttle_ms = max(0, int(args.throttle_ms))
    else:
        try:
            throttle_ms = max(0, int(os.environ.get("WMB_AESTHETIC_THROTTLE_MS", "0")))
        except ValueError:
            throttle_ms = 0
    throttle_sec = throttle_ms / 1000.0

    setup_logging(args.verbose)
    log = logging.getLogger(__name__)

    # CPU-friendliness for the live detector. The tagger thread runs in
    # the same process as DetectionManager; on a Pi 5 with 4 cores the
    # ONNX OD pipeline and CLIP fight for cores unless we ask the OS to
    # prioritise OD. Two orthogonal knobs:
    #
    #   * os.nice(N) — raises the scheduler "niceness" of this thread.
    #     When OD wants CPU, the kernel preempts the tagger first.
    #     Niceness is best-effort under load; threads in 3 mode behave
    #     better than naive throttling because the OS does the work.
    #
    #   * torch.set_num_threads(N) — caps how many cores CLIP can pin
    #     for matrix ops. The default uses all cores, which leaves OD
    #     scrambling for slots; capping to 1-2 reserves headroom for OD
    #     at the cost of slower per-image CLIP inference.
    #
    # Both knobs are env-overridable so the nightly run (when no live
    # OD is competing) can dial them off via env vars.
    _nice_delta_raw = os.environ.get("WMB_AESTHETIC_NICE", "10")
    try:
        nice_delta = max(0, min(19, int(_nice_delta_raw)))
    except ValueError:
        nice_delta = 10
    if nice_delta > 0:
        try:
            os.nice(nice_delta)
            log.info(f"Tagger niceness raised by +{nice_delta} (live OD gets priority)")
        except (OSError, PermissionError) as exc:
            log.warning(f"Tagger niceness raise failed (continuing): {exc}")

    _thread_cap_raw = os.environ.get("WMB_AESTHETIC_TORCH_THREADS", "2")
    try:
        torch_threads = max(0, int(_thread_cap_raw))
    except ValueError:
        torch_threads = 2
    if torch_threads > 0:
        try:
            import torch  # noqa: WPS433 — local so the slim-image fallback path stays import-safe

            torch.set_num_threads(torch_threads)
            log.info(
                f"Tagger torch thread cap = {torch_threads} (reserves cores for OD)"
            )
        except ImportError:
            # If torch is missing, the dependency check below will skip
            # the whole run anyway.
            pass
        except Exception as exc:
            log.warning(f"Tagger torch thread cap failed (continuing): {exc}")

    if args.since is None:
        # Default: yesterday 00:00 UTC. Catches everything from the prior calendar day.
        since_dt = (datetime.now(UTC) - timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        since = since_dt.isoformat()
    else:
        since = args.since

    log.info(
        f"Aesthetic tagger starting; since={since}, dry_run={args.dry_run}, "
        f"db={DB_PATH}, crops={CROPS_ROOT}"
    )

    if not DB_PATH.exists():
        log.error(f"DB not found: {DB_PATH}")
        return 2

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        "PRAGMA busy_timeout = 30000"
    )  # 30s, in case detector pipeline holds locks

    try:
        unscored = fetch_unscored_detections(
            conn,
            since=since,
            limit=args.limit,
            rescore=args.rescore,
            species_filter=args.species,
            per_species_cap=args.per_species_cap,
        )
        if args.rescore:
            log.info(
                f"Found {len(unscored)} detections to RE-SCORE "
                f"(--rescore active; existing scores will be overwritten)"
            )
        elif args.per_species_cap is not None:
            log.info(
                f"Found {len(unscored)} detections needing aesthetic_score "
                f"(--per-species-cap={args.per_species_cap}, ranked by detector "
                f"score / bbox_quality / created_at)"
            )
        else:
            log.info(f"Found {len(unscored)} detections needing aesthetic_score")

        if not unscored:
            log.info("Nothing to score; exiting.")
            return 0

        # Re-score path: clear is_gallery_eligible on the affected
        # detections so the post-score top-N recompute starts from a
        # clean slate. Without this, a previously eligible bad pick
        # would stay eligible until manually overridden.
        if args.rescore and not args.dry_run:
            det_ids = [d["detection_id"] for d in unscored]
            placeholders = ",".join("?" * len(det_ids))
            conn.execute(
                f"UPDATE detections SET is_gallery_eligible = 0 "
                f"WHERE detection_id IN ({placeholders})",
                det_ids,
            )
            conn.commit()
            log.info(
                f"Reset is_gallery_eligible for {len(det_ids)} detections "
                f"before re-scoring"
            )

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
                    log.warning(
                        f"crop missing for det {det['detection_id']}: {det['thumbnail_path']}"
                    )
                continue

            try:
                score = score_image(model, preprocess, text_features, crop_path, device)
            except Exception as exc:
                log.error(f"score failed for det {det['detection_id']}: {exc!r}")
                continue

            if not args.dry_run:
                write_score(
                    conn, det["detection_id"], score, datetime.now(UTC).isoformat()
                )
                # Commit every 10 scores instead of 50: keeps each
                # write-lock window short (~30 ms instead of ~150 ms),
                # which is friendlier to concurrent readers like the
                # health check.
                if scored % 10 == 0:
                    conn.commit()

            scored += 1
            if scored % 25 == 0:
                elapsed = time.time() - t_start
                rate = scored / elapsed if elapsed > 0 else 0
                log.info(
                    f"  [{i}/{len(unscored)}] scored={scored}, missing={skipped_missing}, "
                    f"rate={rate:.1f} img/s"
                )

            # Optional throttle: yield CPU between inferences so the
            # live detector pipeline keeps its frame budget. Default
            # 0 — the bridge run on a Pi 5 with idle headroom doesn't
            # need it. Bump via --throttle-ms or
            # WMB_AESTHETIC_THROTTLE_MS when the bridge starves the
            # detector (visible in the log as DET frames > 1500ms).
            if throttle_sec > 0:
                time.sleep(throttle_sec)

        if not args.dry_run:
            conn.commit()

        log.info(
            f"Scored {scored} detections in {time.time() - t_start:.1f}s "
            f"(skipped_missing={skipped_missing})"
        )

        # Tagging step: only after scores are committed.
        if not args.skip_tagging:
            tagging_stats = apply_auto_favorites(
                conn, since=since, dry_run=args.dry_run
            )
            if not args.dry_run:
                conn.commit()
            log.info(
                f"Marked {tagging_stats['total_tagged']} detections as gallery-eligible "
                f"(already eligible from prior run: {tagging_stats['already_eligible']})"
            )
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
