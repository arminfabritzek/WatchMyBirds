"""
Analytics Blueprint.

Handles analytics routes:
- GET /api/analytics/summary - Summary statistics
- GET /api/analytics/time-of-day - Time distribution KDE
- GET /api/analytics/species-activity - Per-species activity
- GET /analytics - Svelte-rendered dashboard
- GET /analytics-pure - Pure Jinja2/CSS dashboard
"""

import numpy as np
from flask import Blueprint, jsonify, render_template

from logging_config import get_logger
from web.services import db_service

logger = get_logger(__name__)

analytics_bp = Blueprint("analytics", __name__)


@analytics_bp.route("/api/analytics/summary", methods=["GET"])
def analytics_summary():
    with db_service.get_connection() as conn:
        summary = db_service.fetch_analytics_summary(conn)
    return jsonify(summary)


@analytics_bp.route("/api/analytics/time-of-day", methods=["GET"])
def analytics_time_of_day():
    with db_service.get_connection() as conn:
        rows = db_service.fetch_all_detection_times(conn)

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
    with db_service.get_connection() as conn:
        rows = db_service.fetch_species_timestamps(conn)

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


@analytics_bp.route("/analytics", methods=["GET"])
def analytics_page():
    """Serves the analytics dashboard page with minimal HTML for Svelte mount."""
    return render_template("analytics.html")


@analytics_bp.route("/analytics-pure", methods=["GET"])
def analytics_pure():
    """Server-rendered analytics dashboard without Svelte - pure Jinja2/CSS."""
    # 1. Summary Stats
    summary = {
        "total_detections": 0,
        "total_species": 0,
        "date_range": {"first": None, "last": None},
    }
    try:
        with db_service.get_connection() as conn:
            summary = db_service.fetch_analytics_summary(conn)
    except Exception as e:
        logger.error(f"Error fetching analytics summary: {e}")

    # 2. Time of Day Histogram (24 hourly bins)
    time_of_day = {
        "histogram": [],
        "peak_hour": None,
        "peak_hour_formatted": "—",
    }
    try:
        with db_service.get_connection() as conn:
            rows = db_service.fetch_all_detection_times(conn)

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

    # 3. Species Activity with Sparklines
    species_activity = []
    try:
        with db_service.get_connection() as conn:
            rows = db_service.fetch_species_timestamps(conn)

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

    return render_template(
        "analytics_pure.html",
        summary=summary,
        time_of_day=time_of_day,
        species_activity=species_activity,
        current_path="/analytics-pure",
    )
