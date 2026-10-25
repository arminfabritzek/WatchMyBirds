"""Bounded, read-only projections from the photo planner's recent capture cohort."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.deletion_plan import PhotoPlan

FORECAST_WARNING_MONTHS = 3
RECENT_DAYS = 7
MONTH_DAYS = 365.25 / 12


class StorageForecast:
    def __init__(self, now: datetime, *, enabled: bool) -> None:
        self.now = now
        self.enabled = enabled
        self.oldest_age = 0.0
        self.recent_bytes = 0
        self.kept_rows = 0
        self.permanent_bytes = 0
        self.reclaimable_bytes = 0

    def add(self, photo: PhotoPlan) -> None:
        captured = photo.captured_at
        age = (self.now - captured).total_seconds() / 86400 if captured else -1
        self.oldest_age = max(self.oldest_age, age)
        recent = 0 <= age < RECENT_DAYS
        if recent:
            self.kept_rows += sum(
                not self.enabled or row.decision.action == "keep" for row in photo.rows
            )
        for item in photo.files:
            if not item.present or item.error:
                continue
            deadline = item.decision.deadline if self.enabled else None
            if deadline is not None:
                self.reclaimable_bytes += item.size_bytes
            if recent:
                self.recent_bytes += item.size_bytes
                if deadline is None:
                    self.permanent_bytes += item.size_bytes

    def inputs(self) -> dict[str, Any]:
        return dict(
            sample_days=min(RECENT_DAYS, self.oldest_age),
            recent_bytes=self.recent_bytes,
            kept_rows=self.kept_rows,
            permanent_bytes=self.permanent_bytes,
            reclaimable_bytes=self.reclaimable_bytes,
        )


def forecast_capacity(
    inputs: dict[str, Any],
    *,
    free_bytes: int,
    reserve_bytes: int,
    database: dict[str, Any],
) -> dict[str, Any]:
    days = inputs["sample_days"]
    if days < 1:
        return dict(status="insufficient_history")
    records = sum(database.get("records", {}).values())
    used_db = max(
        0, database.get("allocated_bytes", 0) - database.get("reusable_bytes", 0)
    )
    metadata_rate = inputs["kept_rows"] / days * used_db / records if records else 0
    permanent_rate = inputs["permanent_bytes"] / days + metadata_rate
    # Estimate the lasting trend after planned reclamation. Finite retention
    # reaches a plateau; pressure cleanup can shorten its normal deadlines.
    headroom = max(0, free_bytes - reserve_bytes + inputs["reclaimable_bytes"])
    months = headroom / permanent_rate / MONTH_DAYS if permanent_rate > 0 else None
    return dict(
        status="estimated" if months is not None else "not_foreseeable",
        months=months,
        warning=months is not None and months < FORECAST_WARNING_MONTHS,
        warning_months=FORECAST_WARNING_MONTHS,
        sample_days=days,
        saved_bytes_per_day=inputs["recent_bytes"] / days,
        net_bytes_per_day=permanent_rate,
        metadata_bytes_per_day=metadata_rate,
        planned_reclaimable_bytes=inputs["reclaimable_bytes"],
    )
