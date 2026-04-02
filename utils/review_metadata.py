"""Shared review constants and timestamp formatting helpers."""

from __future__ import annotations

from datetime import datetime

REVIEW_STATUS_UNTAGGED = "untagged"
REVIEW_STATUS_CONFIRMED_BIRD = "confirmed_bird"
REVIEW_STATUS_NO_BIRD = "no_bird"

BBOX_REVIEW_CORRECT = "correct"
BBOX_REVIEW_WRONG = "wrong"
VALID_BBOX_REVIEW_STATES = frozenset({BBOX_REVIEW_CORRECT, BBOX_REVIEW_WRONG})


def format_review_timestamp(timestamp: str | None) -> str:
    """Format review timestamps as ``YYYY-MM-DD HH:MM:SS`` when possible."""
    timestamp = str(timestamp or "")
    if len(timestamp) < 15:
        return timestamp[:15]
    try:
        dt = datetime.strptime(timestamp[:15], "%Y%m%d_%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return timestamp[:15]
