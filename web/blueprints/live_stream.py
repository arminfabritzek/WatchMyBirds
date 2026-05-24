"""Live-stream blueprint — pushes UI events (detections, status) via SSE.

Routes:
- GET /api/v1/stream/detections/sse — Server-Sent Events stream consumed
  by the LED-ticker on /stream.

The actual event production happens in detection_manager via
utils.live_event_bus.get_bus().publish(...). This blueprint is a pure
read endpoint.
"""

from __future__ import annotations

import json

from flask import Blueprint, Response, stream_with_context

from logging_config import get_logger
from utils.live_event_bus import get_bus

logger = get_logger(__name__)

live_stream_bp = Blueprint("live_stream", __name__, url_prefix="/api/v1/stream")


@live_stream_bp.get("/detections/sse")
def detections_sse() -> Response:
    bus = get_bus()
    sub_id, q = bus.subscribe()

    @stream_with_context
    def generate():
        # Initial event so the client knows it's connected and can
        # exit any "no signal" idle state immediately.
        yield f"event: hello\ndata: {json.dumps({'sub_id': sub_id})}\n\n"
        try:
            for event in bus.stream(q):
                if event is None:
                    # SSE comment — invisible to EventSource but keeps the
                    # TCP/HTTP pipe alive through reverse-proxies.
                    yield ": ping\n\n"
                    continue
                payload = json.dumps(event, default=str)
                yield f"event: {event.get('type', 'message')}\ndata: {payload}\n\n"
        finally:
            bus.unsubscribe(sub_id)

    response = Response(generate(), mimetype="text/event-stream")
    # Tell nginx and friends NOT to buffer this — SSE depends on
    # immediate flushing per event. ("Connection" is a hop-by-hop
    # header forbidden in WSGI applications per PEP 3333; the server
    # sets it for us.)
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response
