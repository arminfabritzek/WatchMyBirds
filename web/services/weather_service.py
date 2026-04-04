"""
Weather Service - Fetches and caches weather data from Open-Meteo API.

Runs as a background daemon thread, polling every 30 minutes.
Data is cached in-memory for fast UI access and persisted to weather_logs table.
"""

import json
import threading
import time
import urllib.request
from datetime import UTC, datetime

from config import get_config
from logging_config import get_logger
from web.services import db_service

logger = get_logger(__name__)

# WMO Weather Interpretation Codes -> human-readable + emoji
# https://open-meteo.com/en/docs#weathervariables
WMO_CODES = {
    0: ("Clear sky", "☀️"),
    1: ("Mainly clear", "🌤️"),
    2: ("Partly cloudy", "⛅"),
    3: ("Overcast", "☁️"),
    45: ("Fog", "🌫️"),
    48: ("Depositing rime fog", "🌫️"),
    51: ("Light drizzle", "🌦️"),
    53: ("Moderate drizzle", "🌦️"),
    55: ("Dense drizzle", "🌧️"),
    56: ("Light freezing drizzle", "🌧️"),
    57: ("Dense freezing drizzle", "🌧️"),
    61: ("Slight rain", "🌦️"),
    63: ("Moderate rain", "🌧️"),
    65: ("Heavy rain", "🌧️"),
    66: ("Light freezing rain", "🌧️"),
    67: ("Heavy freezing rain", "🌧️"),
    71: ("Slight snowfall", "🌨️"),
    73: ("Moderate snowfall", "🌨️"),
    75: ("Heavy snowfall", "❄️"),
    77: ("Snow grains", "🌨️"),
    80: ("Slight rain showers", "🌦️"),
    81: ("Moderate rain showers", "🌧️"),
    82: ("Violent rain showers", "⛈️"),
    85: ("Slight snow showers", "🌨️"),
    86: ("Heavy snow showers", "❄️"),
    95: ("Thunderstorm", "⛈️"),
    96: ("Thunderstorm with slight hail", "⛈️"),
    99: ("Thunderstorm with heavy hail", "⛈️"),
}

# Global cache for the latest weather
_current_weather_cache = {
    "temp_c": None,
    "relative_humidity_pct": None,
    "precip_mm": None,
    "wind_kph": None,
    "condition_code": None,
    "condition_text": None,
    "condition_emoji": None,
    "is_day": None,
    "timestamp": None,
}


def get_current_weather():
    """Returns the cached current weather dict."""
    return dict(_current_weather_cache)


def get_condition_info(code):
    """Returns (text, emoji) for a WMO weather code."""
    return WMO_CODES.get(code, ("Unknown", "❓"))


def fetch_weather_data():
    """Fetches current weather from Open-Meteo and stores in DB + cache."""
    cfg = get_config()
    location = cfg.get("LOCATION_DATA", {})

    # LOCATION_DATA is a dict with 'latitude' / 'longitude'
    if isinstance(location, dict):
        lat = location.get("latitude", 52.52)
        lon = location.get("longitude", 13.41)
    else:
        lat, lon = 52.52, 13.41

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&"
        f"current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code,is_day&"
        f"timezone=auto"
    )

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "WatchMyBirds/1.0")
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.load(response)

        current = data.get("current", {})

        temp_c = current.get("temperature_2m")
        relative_humidity_pct = current.get("relative_humidity_2m")
        wind_kph = current.get("wind_speed_10m")
        condition_code = current.get("weather_code")
        is_day = current.get("is_day", 1)
        precip_mm = current.get("precipitation", 0.0)

        condition_text, condition_emoji = get_condition_info(condition_code)

        # Night override for clear/partly cloudy
        if not is_day and condition_code in (0, 1):
            condition_emoji = "🌙"

        now_iso = datetime.now(UTC).isoformat()

        # Update in-memory cache
        global _current_weather_cache
        _current_weather_cache = {
            "temp_c": temp_c,
            "relative_humidity_pct": relative_humidity_pct,
            "precip_mm": precip_mm,
            "wind_kph": wind_kph,
            "condition_code": condition_code,
            "condition_text": condition_text,
            "condition_emoji": condition_emoji,
            "is_day": is_day,
            "timestamp": now_iso,
        }

        # Persist to DB
        try:
            with db_service.closing_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO weather_logs (timestamp, temp_c, precip_mm, wind_kph, condition_code, is_day)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        now_iso,
                        temp_c,
                        precip_mm,
                        wind_kph,
                        condition_code,
                        is_day,
                    ),
                )
                conn.commit()
            logger.info(
                "Weather updated: %.1f°C, %s %s, Wind %.0f km/h",
                temp_c or 0,
                condition_emoji,
                condition_text,
                wind_kph or 0,
            )
        except Exception as e:
            logger.error(f"Database error saving weather: {e}")

    except Exception as e:
        logger.error(f"Failed to fetch weather from Open-Meteo: {e}")


def get_weather_history(hours=24):
    """Returns weather history from DB for the last N hours."""
    try:
        with db_service.closing_connection() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, temp_c, precip_mm, wind_kph, condition_code, is_day
                FROM weather_logs
                WHERE timestamp >= datetime('now', ?)
                ORDER BY timestamp ASC
            """,
                (f"-{hours} hours",),
            ).fetchall()

            result = []
            for row in rows:
                code = row[4]
                text, emoji = get_condition_info(code)
                result.append(
                    {
                        "timestamp": row[0],
                        "temp_c": row[1],
                        "precip_mm": row[2],
                        "wind_kph": row[3],
                        "condition_code": code,
                        "condition_text": text,
                        "condition_emoji": emoji,
                        "is_day": row[5],
                    }
                )
            return result
    except Exception as e:
        logger.error(f"Failed to fetch weather history: {e}")
        return []


def start_weather_loop(interval=1800):
    """Starts the background thread to fetch weather every `interval` seconds (default 30min)."""

    def loop():
        logger.info("Weather service started (interval=%ds).", interval)
        # Initial fetch
        fetch_weather_data()
        while True:
            time.sleep(interval)
            fetch_weather_data()

    t = threading.Thread(target=loop, daemon=True, name="WeatherWorker")
    t.start()
