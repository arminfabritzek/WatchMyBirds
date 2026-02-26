"""
Analytics Blueprint.

Handles analytics routes:
- GET /api/analytics/summary - Summary statistics
- GET /api/analytics/time-of-day - Time distribution KDE
- GET /api/analytics/species-activity - Per-species activity
- GET /analytics - Server-rendered analytics dashboard
"""

from calendar import month_abbr, monthrange
from datetime import date, datetime, timedelta

import numpy as np
from flask import Blueprint, jsonify, render_template, request

from logging_config import get_logger
from utils.db.analytics import (
    fetch_all_time_daily_counts,
    fetch_bird_visits,
    fetch_simulation_data,
    fetch_weather_analytics,
    fetch_weather_detection_correlation,
)
from web.services import db_service

logger = get_logger(__name__)

analytics_bp = Blueprint("analytics", __name__)


@analytics_bp.route("/api/analytics/summary", methods=["GET"])
def analytics_summary():
    conn = db_service.get_connection()
    try:
        summary = db_service.fetch_analytics_summary(conn)
    finally:
        conn.close()
    return jsonify(summary)


@analytics_bp.route("/api/analytics/time-of-day", methods=["GET"])
def analytics_time_of_day():
    conn = db_service.get_connection()
    try:
        rows = db_service.fetch_all_detection_times(conn)
    finally:
        conn.close()

    if not rows:
        return jsonify({"points": [], "peak_hour": None, "histogram": []})

    # Parse Times to Float Hours
    hours_float = []
    for row in rows:
        t_str = row["time_str"]  # "HHMMSS"
        if len(t_str) == 6:
            h = int(t_str[0:2])
            m = int(t_str[2:4])
            s = int(t_str[4:6])
            val = h + m / 60.0 + s / 3600.0
            hours_float.append(val)
        elif len(t_str) == 8:  # HH:MM:SS fallback
            try:
                h = int(t_str[0:2])
                m = int(t_str[3:5])
                s = int(t_str[6:8])
                val = h + m / 60.0 + s / 3600.0
                hours_float.append(val)
            except Exception:
                pass

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
    conn = db_service.get_connection()
    try:
        rows = db_service.fetch_species_timestamps(conn)
    finally:
        conn.close()

    # Group by species
    species_times = {}
    for r in rows:
        sp = r["species"]
        t_str = (
            r["image_timestamp"][9:15] if len(r["image_timestamp"]) >= 15 else ""
        )  # YYYYMMDD_HHMMSS
        if len(t_str) == 6:
            try:
                h = int(t_str[0:2]) + int(t_str[2:4]) / 60.0 + int(t_str[4:6]) / 3600.0
                if sp not in species_times:
                    species_times[sp] = []
                species_times[sp].append(h)
            except Exception:
                pass

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

    # Sort by median activity time
    for s in series:
        sp = s["species"]
        times = species_times[sp]
        s["median_hour"] = float(np.median(times))

    series.sort(key=lambda x: x["median_hour"])

    return jsonify(series)


@analytics_bp.route("/api/analytics/visits", methods=["GET"])
def analytics_visits_api():
    """Return bird visit clustering (read-only, no DB writes)."""
    try:
        conn = db_service.get_connection()
        try:
            data = fetch_bird_visits(conn)
        finally:
            conn.close()
        # Strip per-visit detection_ids for the summary response (keep it lean)
        summary = data["summary"]
        # Return top-10 longest visits for quick inspection
        top_visits = sorted(
            data["visits"], key=lambda v: v["duration_sec"], reverse=True
        )[:10]
        for v in top_visits:
            v.pop("detection_ids", None)
        return jsonify({"summary": summary, "top_visits": top_visits})
    except Exception as e:
        logger.error(f"Visits API error: {e}")
        return jsonify({"error": str(e)}), 500


@analytics_bp.route("/api/analytics/simulation", methods=["GET"])
def analytics_simulation_api():
    """Return simulation data for species removal what-if analysis."""
    exclude = request.args.get("exclude", "")
    try:
        conn = db_service.get_connection()
        try:
            data = fetch_simulation_data(conn, exclude if exclude else None)
        finally:
            conn.close()
        return jsonify(data)
    except Exception as e:
        logger.error(f"Simulation API error: {e}")
        return jsonify({"error": str(e)}), 500


@analytics_bp.route("/analytics", methods=["GET"])
def analytics_page():
    """Server-rendered analytics dashboard without Svelte - pure Jinja2/CSS."""
    # 1. Summary Stats
    summary = {
        "total_detections": 0,
        "total_species": 0,
        "date_range": {"first": None, "last": None},
    }
    try:
        conn = db_service.get_connection()
        try:
            summary = db_service.fetch_analytics_summary(conn)
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Error fetching analytics summary: {e}")

    # 1b. Override total_detections with visit-grouped total_visits
    try:
        conn = db_service.get_connection()
        try:
            visit_data = fetch_bird_visits(conn)
            visit_summary = visit_data.get("summary", {})
            summary["total_detections"] = visit_summary.get("total_visits", 0)
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Error fetching all-time visits for analytics: {e}")

    # 2. Time of Day Histogram (24 hourly bins)
    time_of_day = {
        "histogram": [],
        "peak_hour": None,
        "peak_hour_formatted": "—",
    }
    try:
        conn = db_service.get_connection()
        try:
            rows = db_service.fetch_all_detection_times(conn)
        finally:
            conn.close()

        hours_float = []
        for row in rows:
            t_str = row["time_str"]
            if len(t_str) == 6:
                h = int(t_str[0:2])
                m = int(t_str[2:4])
                hours_float.append(h + m / 60.0)

        if hours_float:
            # Create 24 hourly bins
            hist, edges = np.histogram(hours_float, bins=24, range=(0, 24))
            max_count = max(hist) if max(hist) > 0 else 1

            histogram_data = []
            for i, count in enumerate(hist):
                histogram_data.append(
                    {
                        "hour": i,
                        "count": int(count),
                        "height_pct": (
                            round((count / max_count) * 100, 1) if max_count > 0 else 0
                        ),
                    }
                )
            time_of_day["histogram"] = histogram_data

            # Peak hour
            peak_idx = np.argmax(hist)
            time_of_day["peak_hour"] = peak_idx
            time_of_day["peak_hour_formatted"] = f"{peak_idx:02d}:00"
    except Exception as e:
        logger.error(f"Error fetching time of day data: {e}")

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
        conn = db_service.get_connection()
        try:
            daily_rows = fetch_all_time_daily_counts(conn)
        finally:
            conn.close()

        if daily_rows:
            total_days = len(daily_rows)
            counts_by_date: dict[str, int] = {}
            all_dates: list[date] = []
            for row in daily_rows:
                date_iso = row["date_iso"]
                count = int(row["count"] or 0)
                counts_by_date[date_iso] = count
                all_dates.append(datetime.strptime(date_iso, "%Y-%m-%d").date())

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

    # 3. Species Activity with Sparklines
    species_activity = []
    try:
        conn = db_service.get_connection()
        try:
            rows = db_service.fetch_species_timestamps(conn)
        finally:
            conn.close()

        # Group by species
        species_times = {}
        for r in rows:
            sp = r["species"]
            t_str = (
                r["image_timestamp"][9:15] if len(r["image_timestamp"]) >= 15 else ""
            )
            if len(t_str) == 6:
                try:
                    h = int(t_str[0:2]) + int(t_str[2:4]) / 60.0
                    if sp not in species_times:
                        species_times[sp] = []
                    species_times[sp].append(h)
                except Exception:
                    pass

        for sp, times in species_times.items():
            if len(times) < 3:
                continue  # Skip species with very few detections

            # Create histogram for sparkline
            hist, edges = np.histogram(times, bins=24, range=(0, 24))
            max_val = max(hist) if max(hist) > 0 else 1
            normalized = hist / max_val

            # Generate SVG path for sparkline
            points = []
            for i, y in enumerate(normalized):
                x = (i / 23) * 200  # Scale to SVG viewBox width
                y_coord = 30 - (y * 28)  # Invert Y, leave some margin
                prefix = "M" if i == 0 else "L"
                points.append(f"{prefix} {x:.1f} {y_coord:.1f}")
            sparkline_path = " ".join(points)

            # Peak hour
            peak_idx = np.argmax(hist)
            peak_formatted = f"{peak_idx:02d}:00"

            species_activity.append(
                {
                    "species": sp,
                    "count": len(times),
                    "peak_hour_formatted": peak_formatted,
                    "sparkline_path": sparkline_path,
                    "median_hour": float(np.median(times)),
                }
            )

        # Sort by median activity time
        species_activity.sort(key=lambda x: x["median_hour"])
    except Exception as e:
        logger.error(f"Error fetching species activity: {e}")

    # 4. Weather Analytics
    weather = {"has_data": False}
    weather_correlation = []
    try:
        conn = db_service.get_connection()
        try:
            weather = fetch_weather_analytics(conn)
            weather_correlation = fetch_weather_detection_correlation(conn)
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Error fetching weather analytics: {e}")

    # 6. Species Removal Simulation (initial load, no species excluded)
    simulation = {
        "species_list": [],
        "daily_series": [],
        "biodiversity_real": {},
        "biodiversity_sim": {},
        "delta": {},
        "excluded_species": None,
    }
    try:
        conn = db_service.get_connection()
        try:
            simulation = fetch_simulation_data(conn)
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Error fetching simulation data: {e}")

    return render_template(
        "analytics.html",
        summary=summary,
        time_of_day=time_of_day,
        daily_activity=daily_activity,
        activity_granularity=activity_granularity,
        activity_controls=activity_controls,
        species_activity=species_activity,
        weather=weather,
        weather_correlation=weather_correlation,
        simulation=simulation,
        current_path="/analytics",
    )
