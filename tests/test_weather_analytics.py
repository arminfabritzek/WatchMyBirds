import datetime
import sqlite3

from utils.db.analytics import fetch_weather_analytics


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
