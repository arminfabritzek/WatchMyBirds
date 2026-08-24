"""
Analytics Blueprint.

Handles analytics routes:
- GET /api/analytics/summary - Summary statistics
- GET /api/analytics/time-of-day - Time distribution KDE
- GET /api/analytics/species-activity - Per-species activity
- GET /api/analytics/event-intelligence - Event/retention pressure summary
- GET /analytics - Server-rendered analytics dashboard
"""

from calendar import month_abbr, monthrange
from collections import Counter
from datetime import date, datetime, timedelta

import numpy as np
from flask import Blueprint, jsonify, render_template, request

from config import get_config
from core.biodiversity import (
    _parse_event_start,
    hill_numbers,
    is_resolved_species,
    pielou_evenness,
    resolved_species_event_counts,
    sample_coverage,
    shannon_entropy,
    simpson_index,
    species_event_counts,
    species_niche_pca,
)
from core.station_report import (
    ObservationEffort,
    build_candidate_review_priorities,
    build_model_validation,
    calculate_observation_effort,
    partition_events_by_evidence,
)
from logging_config import get_logger
from utils.db.analytics import (
    fetch_event_intelligence_summary,
    fetch_simulation_data,
    fetch_weather_analytics,
    fetch_weather_event_activity,
)
from utils.db.events import EffortStats, calculate_effort, get_events_cached
from web.security import error_response_simple as _error_response_simple
from web.services import cache_service, db_service

logger = get_logger(__name__)

analytics_bp = Blueprint("analytics", __name__)

# Cache TTL for heavy /analytics aggregations. Bounded staleness window
# acceptable because review/moderation routes drop the cache via
# @invalidates("analytics.") on every state change.
_ANALYTICS_TTL = 300  # 5 minutes
_ACTIVITY_MIN_EVENTS = 20
_ACTIVITY_MIN_DAYS = 5


def _empty_observation_effort() -> ObservationEffort:
    return ObservationEffort(
        available=False,
        sample_count=0,
        coverage_start=None,
        coverage_end=None,
        window_hours=0.0,
        app_hours=0.0,
        camera_hours=0.0,
        stream_hours=0.0,
        detector_hours=0.0,
        observation_hours=0.0,
        day_observation_hours=0.0,
        night_observation_hours=0.0,
        unknown_light_hours=0.0,
        ptz_active_hours=0.0,
        outage_hours=0.0,
        weather_exposure_hours={},
    )


def _load_station_report_cohorts(conn, min_score: float):
    model_events = get_events_cached(conn, min_score=min_score)
    verified, candidates, evidence = partition_events_by_evidence(conn, model_events)
    return verified, candidates, evidence


def _event_day_count(events) -> int:
    return len(
        {
            parsed.date()
            for event in events
            if (parsed := _parse_event_start(event.start_time)) is not None
        }
    )


def _events_within_effort(events, effort: ObservationEffort):
    if not effort.available or not effort.coverage_start or not effort.coverage_end:
        return []
    try:
        start = datetime.fromisoformat(effort.coverage_start).astimezone().replace(tzinfo=None)
        end = datetime.fromisoformat(effort.coverage_end).astimezone().replace(tzinfo=None)
    except ValueError:
        return []
    return [
        event
        for event in events
        if (parsed := _parse_event_start(event.start_time)) is not None
        and start <= parsed <= end
    ]


def _apply_observation_weather_exposure(
    weather_activity: dict, effort: ObservationEffort
) -> dict:
    if not effort.available:
        return {
            "conditions": [],
            "matched_events": 0,
            "unmatched_events": 0,
            "effort_available": False,
        }
    conditions = []
    for condition in weather_activity.get("conditions", []):
        code = int(condition.get("condition_code") or 0)
        exposure = float(effort.weather_exposure_hours.get(code, 0.0))
        if exposure <= 0:
            continue
        event_count = int(condition.get("event_count") or 0)
        sufficient = event_count >= 5 and exposure >= 5.0
        item = dict(condition)
        item["exposure_hours"] = round(exposure, 1)
        item["events_per_100_hours"] = (
            round((event_count / exposure) * 100.0, 1) if sufficient else None
        )
        item["sufficient"] = sufficient
        conditions.append(item)
    numeric_rates = [
        item["events_per_100_hours"]
        for item in conditions
        if item["events_per_100_hours"] is not None
    ]
    max_rate = max(numeric_rates, default=0.0)
    for item in conditions:
        rate = item["events_per_100_hours"]
        item["bar_pct"] = round((rate / max_rate) * 100.0, 1) if rate and max_rate else 0.0
    return {
        **weather_activity,
        "conditions": conditions,
        "effort_available": True,
        "effort_basis": "verified events per measured observation hour",
    }


def _cached_event_intelligence(
    min_score: float, event_limit: int, species_limit: int
) -> dict:
    key = f"analytics.event_intelligence:{min_score}:{event_limit}:{species_limit}"

    def build():
        conn = db_service.get_connection()
        try:
            return fetch_event_intelligence_summary(
                conn,
                min_score=min_score,
                event_limit=event_limit,
                species_limit=species_limit,
            )
        finally:
            conn.close()

    return cache_service.cached(key, _ANALYTICS_TTL, build)


def _cached_simulation_data(exclude: str | None) -> dict:
    key = f"analytics.simulation:{exclude or ''}"

    def build():
        conn = db_service.get_connection()
        try:
            return fetch_simulation_data(conn, exclude)
        finally:
            conn.close()

    return cache_service.cached(key, _ANALYTICS_TTL, build)


def _cached_weather_correlation(min_score: float) -> dict:
    def build():
        conn = db_service.get_connection()
        try:
            events, _, _ = _load_station_report_cohorts(conn, min_score)
            return fetch_weather_event_activity(conn, events)
        finally:
            conn.close()

    key = f"analytics.weather_event_activity:{min_score}"
    return cache_service.cached(key, _ANALYTICS_TTL, build)


def _get_species_peak_hour(item: dict) -> float:
    """Return a numeric peak-hour value for deterministic sorting."""
    peak_hour = item.get("peak_hour")
    if peak_hour is not None:
        return float(peak_hour)

    peak_hour_formatted = item.get("peak_hour_formatted", "")
    if isinstance(peak_hour_formatted, str):
        parts = peak_hour_formatted.split(":", 1)
        if parts and parts[0].isdigit():
            return float(parts[0])

    return float("inf")


def _sort_species_activity_by_peak_hour(items: list[dict]) -> list[dict]:
    """Sort species activity by peak hour and break ties by species name."""
    return sorted(
        items,
        key=lambda item: (_get_species_peak_hour(item), item.get("species", "")),
    )


def _event_hour(event) -> float | None:
    parsed = _parse_event_start(event.start_time)
    if parsed is None:
        return None
    return parsed.hour + parsed.minute / 60.0 + parsed.second / 3600.0


def _event_report_summary(events, effort) -> dict:
    resolved_counts = resolved_species_event_counts(events)
    all_counts = species_event_counts(events)
    unresolved = sorted(
        species for species in all_counts if not is_resolved_species(species)
    )
    timestamps = [
        parsed
        for event in events
        if (parsed := _parse_event_start(event.start_time)) is not None
    ]
    return {
        "total_events": len(events),
        "total_photos": sum(event.photo_count for event in events),
        "total_species": len(resolved_counts),
        "unresolved_taxa": unresolved,
        "unresolved_taxa_count": len(unresolved),
        "active_days": effort.active_days,
        "total_days": effort.total_days,
        "date_range": {
            "first": min(timestamps).date().isoformat() if timestamps else None,
            "last": max(timestamps).date().isoformat() if timestamps else None,
        },
        # Compatibility aliases for the public summary API.
        "total_detections": sum(event.photo_count for event in events),
        "event_count": len(events),
    }


def _build_time_of_day(events) -> dict:
    day_count = _event_day_count(events)
    if len(events) < _ACTIVITY_MIN_EVENTS or day_count < _ACTIVITY_MIN_DAYS:
        return {
            "histogram": [],
            "peak_hour": None,
            "peak_hour_formatted": "—",
            "available": False,
            "event_count": len(events),
            "day_count": day_count,
            "minimum_events": _ACTIVITY_MIN_EVENTS,
            "minimum_days": _ACTIVITY_MIN_DAYS,
        }
    hours = [hour for event in events if (hour := _event_hour(event)) is not None]
    if not hours:
        return {
            "histogram": [],
            "peak_hour": None,
            "peak_hour_formatted": "—",
            "available": False,
        }
    hist, _ = np.histogram(hours, bins=24, range=(0, 24))
    max_count = max(hist) if max(hist) > 0 else 1
    peak_idx = int(np.argmax(hist))
    return {
        "histogram": [
            {
                "hour": hour,
                "count": int(count),
                "height_pct": round((count / max_count) * 100, 1),
            }
            for hour, count in enumerate(hist)
        ],
        "peak_hour": peak_idx,
        "peak_hour_formatted": f"{peak_idx:02d}:00",
        "available": True,
        "event_count": len(events),
        "day_count": day_count,
    }


def _build_species_activity(events) -> list[dict]:
    species_hours: dict[str, list[float]] = {}
    for event in events:
        if not is_resolved_species(event.species):
            continue
        hour = _event_hour(event)
        if hour is not None:
            species_hours.setdefault(event.species, []).append(hour)

    series = []
    for species, hours in species_hours.items():
        species_events = [event for event in events if event.species == species]
        if (
            len(species_events) < _ACTIVITY_MIN_EVENTS
            or _event_day_count(species_events) < _ACTIVITY_MIN_DAYS
        ):
            continue
        hist, _ = np.histogram(hours, bins=24, range=(0, 24))
        max_value = max(hist) if max(hist) > 0 else 1
        normalized = hist / max_value
        points = [
            f"{'M' if index == 0 else 'L'} {(index / 23) * 200:.1f} {30 - (value * 28):.1f}"
            for index, value in enumerate(normalized)
        ]
        peak_hour = int(np.argmax(hist))
        series.append(
            {
                "species": species,
                "count": len(hours),
                "peak_hour": peak_hour,
                "peak_hour_formatted": f"{peak_hour:02d}:00",
                "sparkline_path": " ".join(points),
            }
        )
    return _sort_species_activity_by_peak_hour(series)


@analytics_bp.route("/api/analytics/summary", methods=["GET"])
def analytics_summary():
    cfg = get_config()
    min_score = cfg["GALLERY_DISPLAY_THRESHOLD"]
    conn = db_service.get_connection()
    try:
        events, candidates, evidence = _load_station_report_cohorts(conn, min_score)
        summary = _event_report_summary(events, calculate_effort(conn))
        summary.update(evidence)
        summary["model_candidate_photos"] = sum(
            event.photo_count for event in candidates
        )
    finally:
        conn.close()
    return jsonify(summary)


@analytics_bp.route("/api/analytics/time-of-day", methods=["GET"])
def analytics_time_of_day():
    cfg = get_config()
    min_score = cfg["GALLERY_DISPLAY_THRESHOLD"]
    conn = db_service.get_connection()
    try:
        events, _, _ = _load_station_report_cohorts(conn, min_score)
    finally:
        conn.close()

    if not events:
        return jsonify({"points": [], "peak_hour": None, "histogram": []})

    hours_float = [
        hour for event in events if (hour := _event_hour(event)) is not None
    ]

    if not hours_float:
        return jsonify({"points": [], "peak_hour": None, "histogram": []})

    # KDE Approximation via Histogram + Gaussian Smoothing
    bins = 144
    hist, bin_edges = np.histogram(hours_float, bins=bins, range=(0, 24), density=True)

    # Gaussian Smoothing
    sigma = 1.6
    x_vals = np.linspace(-3 * sigma, 3 * sigma, int(6 * sigma) + 1)
    kernel = np.exp(-(x_vals**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)

    smooth_density = np.convolve(hist, kernel, mode="same")

    # Generate Output Points
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    points = []
    max_y = 0
    peak_hour = 0

    for x, y in zip(bin_centers, smooth_density, strict=False):
        points.append({"x": round(float(x), 2), "y": float(y)})
        if y > max_y:
            max_y = y
            peak_hour = x

    # Subsampled Histogram for "Backdrop"
    hist_coarse, edges_coarse = np.histogram(
        hours_float, bins=48, range=(0, 24), density=True
    )
    histogram_points = []
    for i in range(len(hist_coarse)):
        histogram_points.append(
            {
                "x": float((edges_coarse[i] + edges_coarse[i + 1]) / 2),
                "y": float(hist_coarse[i]),
            }
        )

    return jsonify(
        {
            "points": points,
            "peak_hour": round(float(peak_hour), 2),
            "peak_density": float(max_y),
            "histogram": histogram_points,
        }
    )


@analytics_bp.route("/api/analytics/species-activity", methods=["GET"])
def analytics_species_activity():
    cfg = get_config()
    min_score = cfg["GALLERY_DISPLAY_THRESHOLD"]
    conn = db_service.get_connection()
    try:
        events, _, _ = _load_station_report_cohorts(conn, min_score)
    finally:
        conn.close()

    species_times: dict[str, list[float]] = {}
    for event in events:
        if not is_resolved_species(event.species):
            continue
        hour = _event_hour(event)
        if hour is not None:
            species_times.setdefault(event.species, []).append(hour)

    series = []
    for sp, times in species_times.items():
        # Rule: n >= 10 for KDE, else Histogram
        if len(times) < 10:
            # Histogram (1h bins)
            hist, edges = np.histogram(times, bins=24, range=(0, 24), density=False)
            # Normalize to max 1.0
            max_val = np.max(hist)
            if max_val > 0:
                hist = hist / max_val

            centers = (edges[:-1] + edges[1:]) / 2
            points = [
                {"x": float(x), "y": float(y)}
                for x, y in zip(centers, hist, strict=False)
            ]
            peak = centers[np.argmax(hist)]
        else:
            # Numpy Gaussian Smoothing
            bins = 144
            hist, edges = np.histogram(times, bins=bins, range=(0, 24), density=True)

            sigma = 9
            x_vals = np.linspace(-3 * sigma, 3 * sigma, int(6 * sigma) + 1)
            kernel = np.exp(-(x_vals**2) / (2 * sigma**2))
            kernel = kernel / np.sum(kernel)
            smooth = np.convolve(hist, kernel, mode="same")

            # Max Normalization
            max_val = np.max(smooth)
            if max_val > 0:
                smooth = smooth / max_val

            centers = (edges[:-1] + edges[1:]) / 2
            points = [
                {"x": float(x), "y": float(y)}
                for x, y in zip(centers, smooth, strict=False)
            ]
            peak = centers[np.argmax(smooth)]

        series.append(
            {
                "species": sp,
                "points": points,
                "peak_hour": float(peak),
                "count": len(times),
            }
        )

    series = _sort_species_activity_by_peak_hour(series)

    return jsonify(series)


@analytics_bp.route("/api/analytics/visits", methods=["GET"])
def analytics_visits_api():
    """Compatibility endpoint backed by the canonical BirdEvent pipeline."""
    try:
        min_score = get_config()["GALLERY_DISPLAY_THRESHOLD"]
        conn = db_service.get_connection()
        try:
            events, _, _ = _load_station_report_cohorts(conn, min_score)
        finally:
            conn.close()
        species_counts = species_event_counts(events)
        summary = {
            "total_visits": len(events),
            "total_events": len(events),
            "total_detections": sum(event.photo_count for event in events),
            "species_visit_counts": species_counts,
            "avg_visit_duration_sec": round(
                sum(event.duration_sec for event in events) / len(events), 1
            )
            if events
            else 0.0,
            "definition": "BirdEvent",
        }
        top_visits = [
            {
                "species": event.species,
                "start_time": event.start_time,
                "end_time": event.end_time,
                "duration_sec": event.duration_sec,
                "photo_count": event.photo_count,
                "grouping_profile": event.grouping_profile,
            }
            for event in sorted(events, key=lambda item: item.duration_sec, reverse=True)[
                :10
            ]
        ]
        return jsonify({"summary": summary, "top_visits": top_visits})
    except Exception as exc:
        return _error_response_simple("Visits API error", exc)


@analytics_bp.route("/api/analytics/event-intelligence", methods=["GET"])
def analytics_event_intelligence_api():
    """Return BirdEvent and representative-retention summary data."""
    cfg = get_config()
    min_score = cfg["GALLERY_DISPLAY_THRESHOLD"]
    event_limit = min(max(request.args.get("event_limit", 8, type=int), 1), 50)
    species_limit = min(max(request.args.get("species_limit", 8, type=int), 1), 50)
    try:
        data = _cached_event_intelligence(min_score, event_limit, species_limit)
        return jsonify(data)
    except Exception as exc:
        return _error_response_simple("Event intelligence API error", exc)


@analytics_bp.route("/api/analytics/simulation", methods=["GET"])
def analytics_simulation_api():
    """Return simulation data for species removal what-if analysis."""
    exclude = request.args.get("exclude", "")
    try:
        data = _cached_simulation_data(exclude if exclude else None)
        return jsonify(data)
    except Exception as exc:
        return _error_response_simple("Simulation API error", exc)


# --- Biology section helpers and endpoints -----------------------------------
#
# Four read-only views layered on top of the shared event pipeline
# (utils/db/events.py). Pure metric functions live in core/biodiversity.py.
# These power the "Biological Insights" cards at the bottom of /analytics.


def _build_diversity(events) -> dict:
    counts = resolved_species_event_counts(events)
    hills = hill_numbers(counts)
    event_count = sum(counts.values())
    effective_available = event_count >= 30 and len(counts) >= 2

    if hills[0.0] >= 2 and hills[1.0] / hills[0.0] < 0.5:
        dominance_label = "Dominated by a few species"
    elif hills[0.0] >= 2 and hills[1.0] / hills[0.0] >= 0.85:
        dominance_label = "Evenly distributed"
    else:
        dominance_label = "Mixed dominance"

    return {
        "richness": int(hills[0.0]),
        "hill_q1": round(hills[1.0], 2) if effective_available else None,
        "hill_q2": round(hills[2.0], 2) if effective_available else None,
        "shannon": round(shannon_entropy(counts), 3),
        "simpson": round(simpson_index(counts), 3),
        "pielou_evenness": round(pielou_evenness(counts), 3),
        "sample_coverage": (
            round(sample_coverage(counts), 3) if effective_available else None
        ),
        "dominance_label": dominance_label,
        "effective_diversity_available": effective_available,
        "event_count": event_count,
        "minimum_events": 30,
    }


def _build_pca(events, *, min_events_per_species: int = 3) -> dict:
    """Wrap species_niche_pca() into a frontend-friendly dict.

    Min-events filter keeps single-detection rarities out of the PCA chart;
    they would dominate the variance with near-singleton activity profiles.
    Rare species still appear in the species-summary table below.
    """
    pca = species_niche_pca(events, min_events_per_species=min_events_per_species)
    return {
        "ok": pca["ok"],
        "variance_pct": pca["variance_pct"],
        "min_events_filter": min_events_per_species,
        "points": [
            {
                "species": s,
                "x": coord[0],
                "y": coord[1],
                "events": ev_count,
                "peak_hour": peak,
            }
            for s, coord, ev_count, peak in zip(
                pca["species"],
                pca["coords"],
                pca["event_counts"],
                pca["peak_hours"],
                strict=True,
            )
        ],
    }


def _build_species_table(events, effort, *, rate_events=None) -> list[dict]:
    """One row per observed species, sorted by event count descending."""
    counts = resolved_species_event_counts(events)
    if not counts:
        return []
    total_events = sum(counts.values())
    rate_counts = resolved_species_event_counts(rate_events or [])
    photo_by_species: dict[str, int] = {}
    hours_by_species: dict[str, list[int]] = {}
    dates_by_species: dict[str, list[date]] = {}
    evidence_by_species: dict[str, dict[str, object]] = {}
    for ev in events:
        sp = ev.species
        if not is_resolved_species(sp):
            continue
        photo_by_species[sp] = photo_by_species.get(sp, 0) + ev.photo_count
        dt = _parse_event_start(ev.start_time)
        if dt is not None:
            hours_by_species.setdefault(sp, []).append(dt.hour)
            dates_by_species.setdefault(sp, []).append(dt.date())
            current_evidence = evidence_by_species.get(sp)
            if current_evidence is None or dt.date() < current_evidence["date"]:
                evidence_by_species[sp] = {
                    "date": dt.date(),
                    "detection_id": ev.cover_detection_id,
                }

    latest_date = max(
        (day for days in dates_by_species.values() for day in days),
        default=None,
    )

    rows = []
    for species, event_count in counts.items():
        hours = hours_by_species.get(species, [])
        dates = sorted(set(dates_by_species.get(species, [])))
        activity_available = event_count >= _ACTIVITY_MIN_EVENTS and len(dates) >= _ACTIVITY_MIN_DAYS
        peak_hour = max(set(hours), key=hours.count) if hours and activity_available else None
        span_days = (dates[-1] - dates[0]).days + 1 if dates else 0
        occurrence_pattern = (
            "recurring" if len(dates) >= 3 and span_days >= 14 else "sporadic"
        )
        rows.append(
            {
                "species": species,
                "events": event_count,
                "photos": photo_by_species.get(species, 0),
                "events_per_camera_hour": (
                    round(rate_counts.get(species, 0) / effort.observation_hours, 3)
                    if getattr(effort, "available", False)
                    and effort.observation_hours > 0
                    else None
                ),
                "rate_event_count": rate_counts.get(species, 0),
                "peak_hour": peak_hour,
                "activity_available": activity_available,
                "share_pct": round(100.0 * event_count / total_events, 1)
                if total_events
                else 0.0,
                "first_record": dates[0].isoformat() if dates else None,
                "last_record": dates[-1].isoformat() if dates else None,
                "days_present": len(dates),
                "days_since_last": (
                    (latest_date - dates[-1]).days if latest_date and dates else None
                ),
                "occurrence_pattern": occurrence_pattern,
                "evidence_detection_id": (
                    evidence_by_species.get(species, {}).get("detection_id")
                ),
                "evidence_date": (
                    evidence_by_species.get(species, {}).get("date").isoformat()
                    if evidence_by_species.get(species, {}).get("date")
                    else None
                ),
            }
        )
    rows.sort(key=lambda r: (-r["events"], r["species"]))
    return rows


def _build_presence_calendar(events) -> dict:
    parsed = [
        (event, timestamp)
        for event in events
        if is_resolved_species(event.species)
        and (timestamp := _parse_event_start(event.start_time)) is not None
    ]
    if not parsed:
        return {"months": [], "species": []}
    month_keys = sorted({timestamp.strftime("%Y-%m") for _, timestamp in parsed})[-12:]
    counts: dict[str, Counter[str]] = {}
    for event, timestamp in parsed:
        month_key = timestamp.strftime("%Y-%m")
        if month_key not in month_keys:
            continue
        counts.setdefault(str(event.species), Counter())[month_key] += 1
    rows = [
        {
            "species": species,
            "counts": [species_counts.get(month, 0) for month in month_keys],
            "total": sum(species_counts.values()),
        }
        for species, species_counts in counts.items()
    ]
    rows.sort(key=lambda item: (-int(item["total"]), str(item["species"])))
    return {
        "months": month_keys,
        "species": rows,
        "monthly_totals": [
            sum(row["counts"][index] for row in rows)
            for index in range(len(month_keys))
        ],
    }


def _build_quality_metrics(conn) -> dict:
    """Review-status, decision-state, override-rate snapshot."""
    out: dict = {"review_status": {}, "decision_state": {}, "override_rate": 0.0}

    review_rows = conn.execute(
        "SELECT COALESCE(review_status, 'untagged') AS status, COUNT(*) AS n "
        "FROM images GROUP BY COALESCE(review_status, 'untagged')"
    ).fetchall()
    out["review_status"] = {row["status"]: row["n"] for row in review_rows}

    decision_rows = conn.execute(
        "SELECT COALESCE(decision_state, 'unset') AS state, COUNT(*) AS n "
        "FROM detections GROUP BY COALESCE(decision_state, 'unset')"
    ).fetchall()
    out["decision_state"] = {row["state"]: row["n"] for row in decision_rows}

    override_row = conn.execute(
        "SELECT "
        "  SUM(CASE WHEN manual_species_override IS NOT NULL "
        "           AND TRIM(manual_species_override) != '' THEN 1 ELSE 0 END) AS overridden, "
        "  COUNT(*) AS total "
        "FROM detections WHERE COALESCE(status, 'active') = 'active'"
    ).fetchone()
    if override_row and override_row["total"]:
        out["override_rate"] = round(
            (override_row["overridden"] or 0) / override_row["total"], 3
        )
    return out


def _empty_diversity() -> dict:
    return {
        "richness": 0,
        "hill_q1": 0.0,
        "hill_q2": None,
        "shannon": 0.0,
        "simpson": 0.0,
        "pielou_evenness": 0.0,
        "sample_coverage": None,
        "dominance_label": "No data yet",
        "effective_diversity_available": False,
        "event_count": 0,
        "minimum_events": 30,
    }


@analytics_bp.route("/api/analytics/diversity", methods=["GET"])
def analytics_diversity_api():
    """Hill numbers, Shannon, Simpson, Pielou, Chao1, Sample Coverage."""
    cfg = get_config()
    min_score = cfg["GALLERY_DISPLAY_THRESHOLD"]
    try:
        conn = db_service.get_connection()
        try:
            events, _, _ = _load_station_report_cohorts(conn, min_score)
        finally:
            conn.close()
        return jsonify(_build_diversity(events) if events else _empty_diversity())
    except Exception as exc:
        return _error_response_simple("Diversity API error", exc)


@analytics_bp.route("/api/analytics/species-pca", methods=["GET"])
def analytics_species_pca_api():
    """Per-species niche PCA over 24h activity profiles."""
    cfg = get_config()
    min_score = cfg["GALLERY_DISPLAY_THRESHOLD"]
    try:
        conn = db_service.get_connection()
        try:
            events, _, _ = _load_station_report_cohorts(conn, min_score)
        finally:
            conn.close()
        return jsonify(_build_pca(events))
    except Exception as exc:
        return _error_response_simple("Species PCA API error", exc)


@analytics_bp.route("/api/analytics/species-table", methods=["GET"])
def analytics_species_table_api():
    """Per-species summary rows: events, photos, RAI, peak hour, share."""
    cfg = get_config()
    min_score = cfg["GALLERY_DISPLAY_THRESHOLD"]
    try:
        conn = db_service.get_connection()
        try:
            events, _, _ = _load_station_report_cohorts(conn, min_score)
            effort = calculate_observation_effort(conn)
        finally:
            conn.close()
        return jsonify({"rows": _build_species_table(events, effort)})
    except Exception as exc:
        return _error_response_simple("Species table API error", exc)


@analytics_bp.route("/api/analytics/quality-metrics", methods=["GET"])
def analytics_quality_metrics_api():
    """Review-status, decision-state, manual-override snapshot."""
    try:
        conn = db_service.get_connection()
        try:
            data = _build_quality_metrics(conn)
        finally:
            conn.close()
        return jsonify(data)
    except Exception as exc:
        return _error_response_simple("Quality metrics API error", exc)


@analytics_bp.route("/analytics", methods=["GET"])
def analytics_page():
    """Render the station report from one canonical BirdEvent cohort."""
    cfg = get_config()
    min_score = cfg["GALLERY_DISPLAY_THRESHOLD"]
    events = []
    candidate_events = []
    evidence_metrics = {
        "candidate_events": 0,
        "verified_events": 0,
        "assessed_current_events": 0,
        "limited_current_events": 0,
        "human_labeled_candidate_events": 0,
        "human_labeled_candidate_objects": 0,
        "review_actions": 0,
        "rejected_review_actions": 0,
        "unresolved_review_actions": 0,
        "diagnostic_review_actions": 0,
        "limited_review_actions": 0,
        "rejection_rate_pct": 0.0,
        "review_progress_pct": 0.0,
    }
    effort = EffortStats(first_ts=None, last_ts=None, total_days=0, active_days=0)
    observation_effort = _empty_observation_effort()
    model_validation = {
        "available": False,
        "reviewed_predictions": 0,
        "species": [],
        "confidence_bins": [],
        "quality_strata": [],
        "holdout": {"available": False},
    }
    review_priorities = []
    try:
        conn = db_service.get_connection()
        try:
            events, candidate_events, evidence_metrics = _load_station_report_cohorts(
                conn, min_score
            )
            effort = calculate_effort(conn)
            observation_effort = calculate_observation_effort(conn)
            model_validation = build_model_validation(conn)
            review_priorities = build_candidate_review_priorities(
                conn, candidate_events
            )[:8]
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Error fetching station report events: {e}")

    summary = _event_report_summary(events, effort)
    summary.update(evidence_metrics)
    summary["model_candidate_photos"] = sum(
        event.photo_count for event in candidate_events
    )

    # 1c. Event intelligence and representative-retention estimate
    event_intelligence = {
        "summary": {
            "event_count": 0,
            "detection_count": 0,
            "representative_image_count": 0,
            "reducible_image_count": 0,
            "retention_savings_pct": 0.0,
            "avg_photos_per_event": 0.0,
            "compression_ratio": 0.0,
            "largest_event_photo_count": 0,
        },
        "largest_events": [],
        "species_pressure": [],
        "profile_distribution": [],
        "retention_formula": "min(Kmax, 3 + ceil(log2(photo_count)) + bonuses)",
    }
    try:
        event_intelligence = _cached_event_intelligence(
            min_score=min_score,
            event_limit=6,
            species_limit=6,
        )
    except Exception as e:
        logger.error(f"Error fetching event intelligence summary: {e}")

    time_of_day = _build_time_of_day(events)

    # 2b. Activity by Date (toggle: daily/weekly/monthly)
    activity_granularity = (
        request.args.get("activity_granularity", "daily") or ""
    ).lower()
    if activity_granularity not in {"daily", "weekly", "monthly"}:
        activity_granularity = "daily"

    daily_options = [30, 90, 180]
    weekly_options = [12, 26, 52]
    activity_days = request.args.get("activity_days", type=int) or 90
    activity_weeks = request.args.get("activity_weeks", type=int) or 52
    if activity_days not in daily_options:
        activity_days = 90
    if activity_weeks not in weekly_options:
        activity_weeks = 52

    activity_controls = {
        "daily_options": daily_options,
        "daily_days": activity_days,
        "weekly_options": weekly_options,
        "weekly_weeks": activity_weeks,
        "monthly_year": None,
        "monthly_year_options": [],
    }

    daily_activity = {
        "bars": [],
        "dates": [],
        "max_count": 0,
        "total_days": 0,
        "bar_count": 0,
        "bucket_span": 1,
        "granularity": activity_granularity,
        "window_label": "",
        "window_start": None,
        "window_end": None,
    }
    try:
        counts_by_date: dict[str, int] = {}
        for event in events:
            parsed = _parse_event_start(event.start_time)
            if parsed is not None:
                date_iso = parsed.date().isoformat()
                counts_by_date[date_iso] = counts_by_date.get(date_iso, 0) + 1

        if counts_by_date:
            total_days = len(counts_by_date)
            all_dates = [
                datetime.strptime(date_iso, "%Y-%m-%d").date()
                for date_iso in counts_by_date
            ]

            all_dates.sort()
            last_detection_date = all_dates[-1]
            years_with_data = sorted({d.year for d in all_dates})

            selected_year = years_with_data[-1]
            requested_year = request.args.get("activity_year", type=int)
            if requested_year in years_with_data:
                selected_year = requested_year

            activity_controls["monthly_year"] = selected_year
            activity_controls["monthly_year_options"] = years_with_data

            grouped_rows = []
            window_label = ""

            if activity_granularity == "daily":
                window_label = f"Last {activity_days} days"
                start_day = last_detection_date - timedelta(days=activity_days - 1)
                day_ptr = start_day
                while day_ptr <= last_detection_date:
                    day_iso = day_ptr.isoformat()
                    grouped_rows.append(
                        {
                            "start": day_iso,
                            "end": day_iso,
                            "count": int(counts_by_date.get(day_iso, 0)),
                        }
                    )
                    day_ptr += timedelta(days=1)
            elif activity_granularity == "weekly":
                window_label = f"Last {activity_weeks} weeks"
                end_week_start = last_detection_date - timedelta(
                    days=last_detection_date.weekday()
                )
                start_week_start = end_week_start - timedelta(weeks=activity_weeks - 1)
                week_ptr = start_week_start
                while week_ptr <= end_week_start:
                    week_end = week_ptr + timedelta(days=6)
                    week_count = 0
                    for offset in range(7):
                        day_iso = (week_ptr + timedelta(days=offset)).isoformat()
                        week_count += int(counts_by_date.get(day_iso, 0))

                    iso_year, iso_week, _ = week_ptr.isocalendar()
                    grouped_rows.append(
                        {
                            "start": week_ptr.isoformat(),
                            "end": week_end.isoformat(),
                            "count": week_count,
                            "week_label": f"{iso_year}-W{iso_week:02d}",
                        }
                    )
                    week_ptr += timedelta(weeks=1)
            else:
                window_label = f"{selected_year} by month"
                for month in range(1, 13):
                    month_start = date(selected_year, month, 1)
                    month_end = date(
                        selected_year, month, monthrange(selected_year, month)[1]
                    )
                    month_count = 0
                    day_ptr = month_start
                    while day_ptr <= month_end:
                        month_count += int(counts_by_date.get(day_ptr.isoformat(), 0))
                        day_ptr += timedelta(days=1)

                    grouped_rows.append(
                        {
                            "start": month_start.isoformat(),
                            "end": month_end.isoformat(),
                            "count": month_count,
                            "month_label": month_abbr[month],
                        }
                    )

            bucket_starts = [r["start"] for r in grouped_rows]
            bucket_ends = [r["end"] for r in grouped_rows]
            bucket_counts = [int(r["count"]) for r in grouped_rows]
            n = len(bucket_counts)
            max_count = max(bucket_counts) if bucket_counts else 1
            W, H = 800, 120
            PAD_T, PAD_B = 10, 5
            usable_h = H - PAD_T - PAD_B

            bars = []
            if n > 0:
                slot_w = W / n
                bar_w = max(1.0, slot_w * 0.8)
                for i, c in enumerate(bucket_counts):
                    h = (usable_h * (c / max_count)) if max_count > 0 else 0
                    x = i * slot_w + (slot_w - bar_w) / 2
                    y = PAD_T + usable_h - h
                    date_start = bucket_starts[i]
                    date_end = bucket_ends[i]
                    if activity_granularity == "monthly":
                        month_label = grouped_rows[i].get("month_label", date_start[:7])
                        date_label = f"{month_label} {selected_year}"
                    elif activity_granularity == "weekly":
                        week_label = grouped_rows[i].get("week_label", "")
                        date_label = f"{week_label} ({date_start} to {date_end})"
                    elif date_start == date_end:
                        date_label = date_start
                    else:
                        date_label = f"{date_start} to {date_end}"

                    bars.append(
                        {
                            "x": round(x, 2),
                            "y": round(y, 2),
                            "w": round(bar_w, 2),
                            "h": round(h, 2),
                            "count": int(c),
                            "date_label": date_label,
                        }
                    )

            date_labels = []
            if activity_granularity == "monthly":
                date_labels = [month_abbr[m] for m in range(1, 13)]
            elif n > 0:
                label_indices = [0]
                target_label_count = 6
                if n > target_label_count:
                    step = (n - 1) / (target_label_count - 1)
                    for k in range(1, target_label_count - 1):
                        label_indices.append(int(round(step * k)))
                label_indices.append(n - 1)
                label_indices = sorted(set(label_indices))

                for i in label_indices:
                    start_iso = bucket_starts[i]
                    end_iso = bucket_ends[i]
                    if activity_granularity == "weekly":
                        date_labels.append(start_iso[5:])
                    elif start_iso == end_iso:
                        date_labels.append(start_iso[5:])
                    else:
                        date_labels.append(f"{start_iso[5:]}–{end_iso[5:]}")

            daily_activity = {
                "bars": bars,
                "dates": date_labels,
                "max_count": max_count,
                "total_days": total_days,
                "bar_count": len(bars),
                "bucket_span": 1,
                "granularity": activity_granularity,
                "window_label": window_label,
                "window_start": bucket_starts[0] if bucket_starts else None,
                "window_end": bucket_ends[-1] if bucket_ends else None,
            }
    except Exception as e:
        logger.error(f"Error fetching daily activity: {e}")

    species_activity = _build_species_activity(events)

    # 4. Weather Analytics
    weather = {"has_data": False}
    weather_correlation = {
        "conditions": [],
        "matched_events": 0,
        "unmatched_events": len(events),
        "window_minutes": 30,
    }
    try:
        conn = db_service.get_connection()
        try:
            weather = fetch_weather_analytics(conn)
            effort_events = _events_within_effort(events, observation_effort)
            weather_correlation = _apply_observation_weather_exposure(
                fetch_weather_event_activity(conn, effort_events),
                observation_effort,
            )
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Error fetching weather analytics: {e}")

    diversity = _build_diversity(events) if events else _empty_diversity()
    effort_events = _events_within_effort(events, observation_effort)
    species_rows = _build_species_table(
        events, observation_effort, rate_events=effort_events
    )
    presence_calendar = _build_presence_calendar(events)

    return render_template(
        "analytics.html",
        summary=summary,
        time_of_day=time_of_day,
        daily_activity=daily_activity,
        activity_granularity=activity_granularity,
        activity_controls=activity_controls,
        species_activity=species_activity,
        event_intelligence=event_intelligence,
        weather=weather,
        weather_correlation=weather_correlation,
        diversity=diversity,
        species_rows=species_rows,
        effort=effort,
        observation_effort=observation_effort,
        model_validation=model_validation,
        review_priorities=review_priorities,
        presence_calendar=presence_calendar,
        current_path="/analytics",
    )
