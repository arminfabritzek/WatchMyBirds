import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import cv2

from config import get_config
from core.analysis_queue import analysis_queue
from web.services import db_service, gallery_service

logger = logging.getLogger(__name__)
config = get_config()


@dataclass
class _DeepReviewDetectionData:
    """
    Duck-typed payload compatible with PersistenceService.save_detection().
    """

    bbox: tuple[int, int, int, int]
    confidence: float
    class_name: str = ""
    cls_class_name: str = ""
    cls_confidence: float = 0.0
    score: float = 0.0
    agreement_score: float = 0.0


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


def check_deep_analysis_eligibility(
    filename: str, force: bool = False
) -> tuple[bool, str]:
    """
    Deep analysis is allowed only for unreviewed orphan review items.
    When force=True, the no-hit exclusion (deep_scan_last_result='none') is skipped
    but the image must still exist and be unreviewed with no detections.
    """
    with db_service.closing_connection() as conn:
        row = conn.execute(
            """
            SELECT
                i.review_status,
                i.deep_scan_last_result,
                EXISTS(
                    SELECT 1
                    FROM detections d
                    WHERE d.image_filename = i.filename
                ) AS has_detections
            FROM images i
            WHERE i.filename = ?
            LIMIT 1
            """,
            (filename,),
        ).fetchone()

    if not row:
        return False, "Image not found."

    review_status = row["review_status"]
    if review_status not in (None, "untagged"):
        return False, "Image is already reviewed."

    if row["has_detections"]:
        return (
            False,
            "Deep analysis is only allowed for review items without detections.",
        )

    # No-hit exclusion: skip images already deep-scanned with no detections
    # force=True bypasses ONLY this DB filter (Constraint #4)
    if not force:
        last_result = row["deep_scan_last_result"]
        if last_result == "none":
            return False, "Already deep-scanned with no detections."

    return True, ""


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


def submit_analysis_job(filename: str, force: bool = False) -> bool:
    """
    Submits a file for deep analysis.
    Returns True if submitted, False if not eligible or already pending.

    force=True bypasses the DB no-hit exclusion but still respects
    queue dedup for pending/running jobs (Constraint #4).
    """
    is_eligible, reason = check_deep_analysis_eligibility(filename, force=force)
    if not is_eligible:
        logger.info(f"Skipping deep analysis for {filename}: {reason}")
        return False

    # Optionally reset deep_scan_last_result on force re-scan
    if force:
        try:
            with db_service.closing_connection() as conn:
                conn.execute(
                    "UPDATE images SET deep_scan_last_result = NULL WHERE filename = ?",
                    (filename,),
                )
        except Exception as e:
            logger.warning(f"Could not reset deep_scan_last_result for {filename}: {e}")

    try:
        ok = analysis_queue.enqueue({"filename": filename})
        if not ok:
            logger.info(f"Dedup: {filename} already pending in queue")
        return ok
    except Exception as e:
        logger.error(f"Failed to submit analysis job: {e}")
        return False


# ---------------------------------------------------------------------------
# Detection payload builder
# ---------------------------------------------------------------------------


def _build_detection_payload(
    detection_manager,
    frame,
    raw_detection: dict,
) -> tuple[_DeepReviewDetectionData, str]:
    """
    Build a normalized detection payload using the same CLS + scoring logic
    as the regular detection pipeline.
    """
    x1 = int(raw_detection["x1"])
    y1 = int(raw_detection["y1"])
    x2 = int(raw_detection["x2"])
    y2 = int(raw_detection["y2"])
    bbox = (x1, y1, x2, y2)
    od_conf = float(raw_detection["confidence"])

    cls_name = ""
    cls_conf = 0.0
    classifier_model_id = detection_manager.classifier_model_id or ""

    crop_rgb = detection_manager.crop_service.create_classification_crop(
        frame=frame,
        bbox=bbox,
        size=detection_manager.SAVE_RESOLUTION_CROP,
        margin_percent=0.1,
        to_rgb=True,
    )
    if crop_rgb is not None:
        cls_result = detection_manager.classification_service.classify(crop_rgb)
        cls_name = cls_result.class_name
        cls_conf = cls_result.confidence
        if not detection_manager.classifier_model_id and cls_result.model_id:
            detection_manager.classifier_model_id = cls_result.model_id
        classifier_model_id = (
            detection_manager.classifier_model_id or cls_result.model_id or ""
        )

    if cls_conf > 0:
        score = 0.5 * od_conf + 0.5 * cls_conf
        agreement_score = min(od_conf, cls_conf)
    else:
        score = od_conf
        agreement_score = od_conf

    payload = _DeepReviewDetectionData(
        bbox=bbox,
        confidence=od_conf,
        class_name=raw_detection.get("class_name", "bird"),
        cls_class_name=cls_name,
        cls_confidence=cls_conf,
        score=score,
        agreement_score=agreement_score,
    )
    return payload, classifier_model_id


# ---------------------------------------------------------------------------
# Deep scan DB helpers
# ---------------------------------------------------------------------------


def _record_deep_scan_start(filename: str) -> None:
    """Increment attempt count and record timestamp at job start."""
    try:
        with db_service.closing_connection() as conn:
            conn.execute(
                """
                UPDATE images
                SET deep_scan_attempt_count = COALESCE(deep_scan_attempt_count, 0) + 1,
                    deep_scan_last_attempt_at = ?
                WHERE filename = ?
                """,
                (datetime.utcnow().isoformat(), filename),
            )
    except Exception as e:
        logger.warning(f"Could not record deep scan start for {filename}: {e}")


def _record_deep_scan_result(filename: str, result: str) -> None:
    """Set deep_scan_last_result ('none', 'found', 'error')."""
    try:
        with db_service.closing_connection() as conn:
            conn.execute(
                "UPDATE images SET deep_scan_last_result = ? WHERE filename = ?",
                (result, filename),
            )
    except Exception as e:
        logger.warning(f"Could not record deep scan result for {filename}: {e}")


# ---------------------------------------------------------------------------
# Worker function (called by AnalysisQueue._worker_loop)
# ---------------------------------------------------------------------------


def process_deep_analysis_job(detection_manager, job_data: dict):
    """
    Worker function to process analysis jobs.
    Executed by the background thread.
    """
    filename = job_data.get("filename")
    if not filename:
        return

    is_eligible, reason = check_deep_analysis_eligibility(filename)
    if not is_eligible:
        logger.info(f"Skipping deep analysis for {filename}: {reason}")
        return

    logger.info(f"Starting ID-Deep-Scan for {filename}")

    # Record attempt start
    _record_deep_scan_start(filename)

    # Resolve path
    output_dir = config["OUTPUT_DIR"]
    paths = gallery_service.get_image_paths(output_dir, filename)
    img_path = str(paths["original"])

    # Load image
    frame = cv2.imread(img_path)
    if frame is None:
        logger.error(f"Could not read image for analysis: {img_path}")
        _record_deep_scan_result(filename, "error")
        return

    # Run scan
    try:
        start_ts = datetime.now()
        # This blocks the detector lock!
        detections = detection_manager.run_exhaustive_scan(frame)
        duration_sec = (datetime.now() - start_ts).total_seconds()
        logger.info(f"Deep scan: {len(detections)} objects in {duration_sec:.2f}s")
    except Exception as e:
        logger.error(f"Deep scan failed: {e}", exc_info=True)
        _record_deep_scan_result(filename, "error")
        return

    if not detections:
        logger.info(f"Deep scan for {filename} found nothing.")
        _record_deep_scan_result(filename, "none")
        return

    # Save results via the standard persistence + classification path.
    try:
        with db_service.closing_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM detections WHERE image_filename = ?",
                (filename,),
            ).fetchone()
            existing_count = int(row["cnt"]) if row else 0

        saved_count = 0
        scan_idx = 0
        for raw_det in detections:
            conf = float(raw_det["confidence"])
            if conf < 0.1:
                continue

            scan_idx += 1
            try:
                payload, classifier_model_id = _build_detection_payload(
                    detection_manager=detection_manager,
                    frame=frame,
                    raw_detection=raw_det,
                )
            except Exception as e:
                logger.error(
                    f"Failed to build deep-scan payload for {filename}: {e}",
                    exc_info=True,
                )
                continue
            method = raw_det.get("method", "unknown")

            result = detection_manager.persistence_service.save_detection(
                image_filename=filename,
                detection=payload,
                frame=frame,
                detector_model_id=f"deep_scan_{method}",
                classifier_model_id=classifier_model_id,
                crop_index=existing_count + scan_idx,
            )
            if result.success:
                saved_count += 1

        logger.info(f"Saved {saved_count} new detections from deep scan for {filename}")
        _record_deep_scan_result(filename, "found")
    except Exception as e:
        logger.error(f"Error saving deep scan results: {e}", exc_info=True)
        _record_deep_scan_result(filename, "error")


# ---------------------------------------------------------------------------
# Nightly candidate query (Primary DB filter — Constraint #5)
# ---------------------------------------------------------------------------


def _fetch_orphan_review_filenames() -> list[str]:
    """
    Return only review items that currently have no detections AND were not
    already deep-scanned with 'none' result.  Images with 'error' result ARE
    retried so they get another chance.
    """
    with db_service.closing_connection() as conn:
        rows = conn.execute(
            """
            SELECT i.filename
            FROM images i
            WHERE (i.review_status IS NULL OR i.review_status = 'untagged')
              AND i.filename IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1
                  FROM detections d
                  WHERE d.image_filename = i.filename
              )
              AND (i.deep_scan_last_result IS NULL OR i.deep_scan_last_result = 'error')
            ORDER BY i.timestamp ASC
            """
        ).fetchall()
    return [row["filename"] for row in rows]


def count_deep_scan_candidates() -> int:
    """Count nightly-eligible deep scan candidates (used by /api/status)."""
    with db_service.closing_connection() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM images i
            WHERE (i.review_status IS NULL OR i.review_status = 'untagged')
              AND i.filename IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1
                  FROM detections d
                  WHERE d.image_filename = i.filename
              )
              AND (i.deep_scan_last_result IS NULL OR i.deep_scan_last_result = 'error')
            """
        ).fetchone()
    return int(row["cnt"]) if row else 0


# ---------------------------------------------------------------------------
# Nightly sweep
# ---------------------------------------------------------------------------


def start_nightly_analysis_sweep(interval=900):
    """
    Background thread that checks if it's night and enqueues review items.
    interval: check interval in seconds (default 15 minutes).
    """
    from web.services.weather_service import get_current_weather

    def sweep_loop():
        logger.info(f"Nightly scan sweep started (interval={interval}s)")
        while True:
            try:
                weather = get_current_weather()
                is_day = weather.get("is_day", 1)  # Default to day if unknown

                if is_day == 0:
                    # Only orphan review items (no detections) are eligible.
                    filenames = _fetch_orphan_review_filenames()
                    if filenames:
                        logger.info(
                            "Night mode active: Enqueuing %d orphan review items for deep scan.",
                            len(filenames),
                        )
                        for filename in filenames:
                            submit_analysis_job(filename)
            except Exception as e:
                logger.error(f"Error in nightly sweep: {e}")

            time.sleep(interval)

    t = threading.Thread(target=sweep_loop, name="NightlyAnalysisSweeper", daemon=True)
    t.start()
