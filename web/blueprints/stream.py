import threading
import time

import cv2
from flask import Blueprint, Response, jsonify, request

from config import get_config
from logging_config import get_logger
from web.security import safe_log_value as _slv

logger = get_logger(__name__)
config = get_config()

stream_bp = Blueprint("stream", __name__)

_output_resize_width = config["STREAM_WIDTH_OUTPUT_RESIZE"]
_video_feed_semaphore = threading.BoundedSemaphore(2)

_detection_manager = None


def init_stream_bp(detection_manager=None):
    global _detection_manager
    _detection_manager = detection_manager


@stream_bp.route("/video_feed")
def video_feed():
    stream_fps = float(config.get("STREAM_FPS", 5.0) or 5.0)
    stream_fps = max(1.0, stream_fps)
    frame_interval = 1.0 / stream_fps

    if not _video_feed_semaphore.acquire(blocking=False):
        logger.info(
            "MJPEG /video_feed refused (cap=2 reached) ip=%s",
            _slv(request.remote_addr),
        )
        return (
            "Stream cap reached; close other tabs and retry.\n",
            503,
            {"Content-Type": "text/plain; charset=utf-8", "Retry-After": "5"},
        )

    def generate():
        MAX_STREAM_SECONDS = 30 * 60

        stream_started = time.time()
        try:
            while True:
                if time.time() - stream_started > MAX_STREAM_SECONDS:
                    logger.info(
                        "MJPEG /video_feed self-closed after %ss ip=%s",
                        MAX_STREAM_SECONDS,
                        _slv(request.remote_addr),
                    )
                    return
                loop_start = time.time()
                frame = _detection_manager.get_display_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue

                try:
                    h, w = frame.shape[:2]
                    output_h = int(h * _output_resize_width / w) if w else h
                    resized = cv2.resize(frame, (_output_resize_width, output_h))
                    ok, buffer = cv2.imencode(
                        ".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 80]
                    )
                    if not ok:
                        continue
                except Exception as e:
                    logger.debug(f"Failed to encode MJPEG frame: {e}")
                    time.sleep(0.05)
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )

                elapsed = time.time() - loop_start
                sleep_for = frame_interval - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
        finally:
            _video_feed_semaphore.release()

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@stream_bp.route("/api/snapshot")
def snapshot_api():
    from datetime import datetime

    frame = _detection_manager.get_display_frame()
    if frame is None:
        return jsonify({"error": "No frame available"}), 503

    h, w = frame.shape[:2]
    output_h = int(h * _output_resize_width / w) if w else h
    resized = cv2.resize(frame, (_output_resize_width, output_h))

    _, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 90])

    filename = f"snapshot_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"

    return Response(
        buffer.tobytes(),
        mimetype="image/jpeg",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
