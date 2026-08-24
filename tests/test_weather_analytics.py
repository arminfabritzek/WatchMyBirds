import datetime
import sqlite3
from types import SimpleNamespace

from utils.db.analytics import fetch_weather_analytics, match_events_to_weather_exposure


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE weather_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            temp_c REAL,
            precip_mm REAL,
            wind_kph REAL,
            condition_code INTEGER,
            is_day INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    return conn


def test_fetch_weather_analytics_sets_temp_bounds_for_single_timeline_reading():
    conn = _make_conn()
    try:
        # Within last 24h so it appears in timeline_24h (n==1).
        ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            """
            INSERT INTO weather_logs(timestamp, temp_c, precip_mm, wind_kph, condition_code, is_day)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (ts, 10.0, 0.0, 1.0, 0, 1),
        )
        conn.commit()

        weather = fetch_weather_analytics(conn)
        assert weather["has_data"] is True
        assert weather["timeline_24h"]
        assert "temp_min" in weather
        assert "temp_max" in weather
        assert weather["temp_min"] == 10.0
        assert weather["temp_max"] == 10.0
    finally:
        conn.close()


def test_fetch_weather_analytics_sets_temp_bounds_from_weekly_when_no_timeline():
    conn = _make_conn()
    try:
        # Older than 24h but within last 7 days.
        ts = (datetime.datetime.utcnow() - datetime.timedelta(days=2)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        conn.execute(
            """
            INSERT INTO weather_logs(timestamp, temp_c, precip_mm, wind_kph, condition_code, is_day)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (ts, 12.3, 0.0, 3.0, 3, 1),
        )
        conn.commit()

        weather = fetch_weather_analytics(conn)
        assert weather["has_data"] is True
        assert weather["weekly_summary"]
        assert weather["timeline_24h"] == []
        assert "temp_min" in weather
        assert "temp_max" in weather
        assert weather["temp_min"] == 12.3
        assert weather["temp_max"] == 12.3
    finally:
        conn.close()


def test_weekly_temperature_bounds_do_not_overwrite_24h_chart_bounds():
    conn = _make_conn()
    try:
        now = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
        recent = now.strftime("%Y-%m-%d %H:%M:%S")
        older = (now - datetime.timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S")
        conn.executemany(
            "INSERT INTO weather_logs(timestamp, temp_c, precip_mm, wind_kph, condition_code, is_day) "
            "VALUES (?, ?, 0, 2, 0, 1)",
            [(recent, 18.0), (older, 35.0)],
        )
        conn.commit()

        weather = fetch_weather_analytics(conn)

        assert weather["temp_min"] == 18.0
        assert weather["temp_max"] == 18.0
        assert weather["weekly_temp_max"] == 35.0
    finally:
        conn.close()


def test_weather_event_activity_is_exposure_normalized_and_uses_nearest_reading():
    rows = [
        {
            "timestamp": "2026-04-25T10:00:00",
            "condition_code": 0,
            "temp_c": 12.0,
            "wind_kph": 3.0,
        },
        {
            "timestamp": "2026-04-25T11:00:00",
            "condition_code": 0,
            "temp_c": 13.0,
            "wind_kph": 4.0,
        },
        {
            "timestamp": "2026-04-25T12:00:00",
            "condition_code": 3,
            "temp_c": 14.0,
            "wind_kph": 5.0,
        },
    ]
    events = [
        SimpleNamespace(start_time="20260425_100500"),
        SimpleNamespace(start_time="20260425_120500"),
    ]

    result = match_events_to_weather_exposure(events, rows)
    by_code = {row["condition_code"]: row for row in result["conditions"]}

    assert result["matched_events"] == 2
    assert by_code[0]["event_count"] == 1
    assert by_code[0]["exposure_hours"] == 2.0
    assert by_code[0]["events_per_100_hours"] == 50.0
    assert by_code[3]["event_count"] == 1
    assert by_code[3]["events_per_100_hours"] == 100.0


def test_weather_event_matching_converts_utc_reading_to_station_local_time():
    weather_timestamp = "2026-07-10T10:00:00+00:00"
    local_timestamp = datetime.datetime.fromisoformat(weather_timestamp).astimezone()
    events = [SimpleNamespace(start_time=local_timestamp.strftime("%Y%m%d_%H%M%S"))]
    rows = [
        {
            "timestamp": weather_timestamp,
            "condition_code": 1,
            "temp_c": 20.0,
            "wind_kph": 4.0,
        }
    ]

    result = match_events_to_weather_exposure(events, rows)

    assert result["matched_events"] == 1
    assert result["unmatched_events"] == 0
