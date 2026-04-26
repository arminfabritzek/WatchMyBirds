from unittest.mock import MagicMock

import pytest
from flask import Flask

from web.blueprints import analytics as analytics_module
from web.blueprints.analytics import _sort_species_activity_by_peak_hour, analytics_bp


@pytest.fixture
def client():
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(analytics_bp)

    with app.test_client() as client:
        yield client


def test_sort_species_activity_by_peak_hour_orders_by_peak_then_species():
    items = [
        {"species": "Phoenicurus_ochruros", "peak_hour_formatted": "08:00"},
        {"species": "Dendrocopos_major", "peak_hour_formatted": "05:00"},
        {"species": "Passer_domesticus", "peak_hour_formatted": "08:00"},
        {"species": "Erithacus_rubecula", "peak_hour": 7.5},
    ]

    sorted_items = _sort_species_activity_by_peak_hour(items)

    assert [item["species"] for item in sorted_items] == [
        "Dendrocopos_major",
        "Erithacus_rubecula",
        "Passer_domesticus",
        "Phoenicurus_ochruros",
    ]


def test_species_activity_api_sorts_by_peak_hour(monkeypatch, client):
    mock_conn = MagicMock()
    rows = [
        {"species": "Phoenicurus_ochruros", "image_timestamp": "20260311_081500"},
        {"species": "Phoenicurus_ochruros", "image_timestamp": "20260311_082500"},
        {"species": "Phoenicurus_ochruros", "image_timestamp": "20260311_084500"},
        {"species": "Dendrocopos_major", "image_timestamp": "20260311_050500"},
        {"species": "Dendrocopos_major", "image_timestamp": "20260311_051000"},
        {"species": "Dendrocopos_major", "image_timestamp": "20260311_053500"},
        {"species": "Erithacus_rubecula", "image_timestamp": "20260311_071000"},
        {"species": "Erithacus_rubecula", "image_timestamp": "20260311_072000"},
        {"species": "Erithacus_rubecula", "image_timestamp": "20260311_075500"},
        {"species": "Passer_domesticus", "image_timestamp": "20260311_080500"},
        {"species": "Passer_domesticus", "image_timestamp": "20260311_081000"},
        {"species": "Passer_domesticus", "image_timestamp": "20260311_085500"},
    ]

    monkeypatch.setattr(
        analytics_module.db_service, "get_connection", lambda: mock_conn
    )
    monkeypatch.setattr(
        analytics_module.db_service, "fetch_species_timestamps", lambda conn, **kw: rows
    )

    response = client.get("/api/analytics/species-activity")

    assert response.status_code == 200
    assert [item["species"] for item in response.get_json()] == [
        "Dendrocopos_major",
        "Erithacus_rubecula",
        "Passer_domesticus",
        "Phoenicurus_ochruros",
    ]
    mock_conn.close.assert_called_once()


def test_event_intelligence_api_returns_summary(monkeypatch, client):
    mock_conn = MagicMock()
    payload = {
        "summary": {
            "event_count": 2,
            "detection_count": 12,
            "representative_image_count": 7,
            "reducible_image_count": 5,
            "retention_savings_pct": 41.7,
            "avg_photos_per_event": 6.0,
            "compression_ratio": 1.7,
            "largest_event_photo_count": 10,
        },
        "largest_events": [],
        "species_pressure": [],
        "profile_distribution": [],
        "retention_formula": "min(Kmax, 3 + ceil(log2(photo_count)) + bonuses)",
    }

    monkeypatch.setattr(
        analytics_module, "get_config", lambda: {"GALLERY_DISPLAY_THRESHOLD": 0.85}
    )
    monkeypatch.setattr(
        analytics_module.db_service, "get_connection", lambda: mock_conn
    )

    def fake_fetch(conn, min_score: float, *, event_limit: int, species_limit: int):
        assert conn is mock_conn
        assert min_score == 0.85
        assert event_limit == 3
        assert species_limit == 4
        return payload

    monkeypatch.setattr(
        analytics_module, "fetch_event_intelligence_summary", fake_fetch
    )

    response = client.get(
        "/api/analytics/event-intelligence?event_limit=3&species_limit=4"
    )

    assert response.status_code == 200
    assert response.get_json()["summary"]["event_count"] == 2
    assert response.get_json()["summary"]["retention_savings_pct"] == 41.7
    mock_conn.close.assert_called_once()


# ============================================================
# Biological insights endpoints — diversity, species PCA,
# species table, quality-metrics. Use real in-memory DB so the
# event aggregation, BirdEvent dataclass, and biodiversity
# metric module are exercised end-to-end.
# ============================================================

import sqlite3
from datetime import datetime, timedelta

from utils.db.events import clear_events_cache


def _make_bio_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE images (
            filename TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            source_id INTEGER,
            review_status TEXT DEFAULT 'untagged'
        );
        CREATE TABLE detections (
            detection_id INTEGER PRIMARY KEY,
            image_filename TEXT NOT NULL,
            bbox_x REAL, bbox_y REAL, bbox_w REAL, bbox_h REAL,
            od_class_name TEXT,
            score REAL,
            status TEXT NOT NULL DEFAULT 'active',
            decision_state TEXT,
            manual_species_override TEXT,
            species_source TEXT
        );
        CREATE TABLE classifications (
            classification_id INTEGER PRIMARY KEY,
            detection_id INTEGER NOT NULL,
            cls_class_name TEXT,
            cls_confidence REAL,
            rank INTEGER DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'active'
        );
        """
    )
    return conn


def _insert_bio(conn, det_id, ts, species):
    fn = f"{det_id:04d}.webp"
    conn.execute(
        "INSERT INTO images(filename, timestamp, source_id, review_status) "
        "VALUES (?, ?, 1, 'confirmed_bird')",
        (fn, ts),
    )
    conn.execute(
        "INSERT INTO detections(detection_id, image_filename, bbox_x, bbox_y, "
        "bbox_w, bbox_h, od_class_name, score, status, decision_state) "
        "VALUES (?, ?, 0.1, 0.1, 0.2, 0.2, 'bird', 0.95, 'active', 'confirmed')",
        (det_id, fn),
    )
    conn.execute(
        "INSERT INTO classifications(detection_id, cls_class_name, "
        "cls_confidence, rank, status) VALUES (?, ?, 0.91, 1, 'active')",
        (det_id, species),
    )


def _seed_bio(conn):
    """Three species with disjoint diel windows so PCA returns ok=True."""
    base = datetime.strptime("20260420_060000", "%Y%m%d_%H%M%S")
    for i in range(3):
        ts = (base + timedelta(days=i)).strftime("%Y%m%d_%H%M%S")
        _insert_bio(conn, i + 1, ts, "Erithacus_rubecula")
    base_mid = datetime.strptime("20260420_120000", "%Y%m%d_%H%M%S")
    for i in range(3):
        ts = (base_mid + timedelta(days=i)).strftime("%Y%m%d_%H%M%S")
        _insert_bio(conn, i + 10, ts, "Cyanistes_caeruleus")
    base_eve = datetime.strptime("20260420_200000", "%Y%m%d_%H%M%S")
    for i in range(3):
        ts = (base_eve + timedelta(days=i)).strftime("%Y%m%d_%H%M%S")
        _insert_bio(conn, i + 20, ts, "Turdus_merula")


class _NoCloseConn:
    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self):
        pass


@pytest.fixture
def bio_client(monkeypatch):
    clear_events_cache()
    conn = _make_bio_conn()
    _seed_bio(conn)
    wrapped = _NoCloseConn(conn)
    monkeypatch.setattr(analytics_module.db_service, "get_connection", lambda: wrapped)
    monkeypatch.setattr(
        analytics_module, "get_config", lambda: {"GALLERY_DISPLAY_THRESHOLD": 0.0}
    )

    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(analytics_bp)
    with app.test_client() as c:
        yield c


@pytest.fixture
def bio_empty_client(monkeypatch):
    clear_events_cache()
    conn = _make_bio_conn()
    wrapped = _NoCloseConn(conn)
    monkeypatch.setattr(analytics_module.db_service, "get_connection", lambda: wrapped)
    monkeypatch.setattr(
        analytics_module, "get_config", lambda: {"GALLERY_DISPLAY_THRESHOLD": 0.0}
    )

    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(analytics_bp)
    with app.test_client() as c:
        yield c


def test_diversity_api_empty_returns_zero_richness(bio_empty_client):
    r = bio_empty_client.get("/api/analytics/diversity")
    assert r.status_code == 200
    body = r.get_json()
    assert body["richness"] == 0


def test_diversity_api_with_data_returns_hill_numbers(bio_client):
    r = bio_client.get("/api/analytics/diversity")
    assert r.status_code == 200
    body = r.get_json()
    assert body["richness"] == 3
    assert "hill_q1" in body
    assert "hill_q2" in body
    assert "sample_coverage" in body
    assert "chao1_richness" in body


def test_species_pca_api_empty_returns_not_ok(bio_empty_client):
    r = bio_empty_client.get("/api/analytics/species-pca")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is False
    assert body["points"] == []


def test_species_pca_api_with_three_species_returns_ok(bio_client):
    r = bio_client.get("/api/analytics/species-pca")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert len(body["points"]) == 3
    pc1, pc2 = body["variance_pct"]
    assert 0.0 <= pc1 <= 100.0
    assert 0.0 <= pc2 <= 100.0
    assert pc1 + pc2 <= 100.001


def test_species_table_api_empty(bio_empty_client):
    r = bio_empty_client.get("/api/analytics/species-table")
    assert r.status_code == 200
    assert r.get_json()["rows"] == []


def test_species_table_api_sorted_by_events_desc(bio_client):
    r = bio_client.get("/api/analytics/species-table")
    body = r.get_json()
    rows = body["rows"]
    assert len(rows) == 3
    counts = [row["events"] for row in rows]
    assert counts == sorted(counts, reverse=True)
    for row in rows:
        assert "rai_per_100_days" in row
        assert "peak_hour" in row
        assert "share_pct" in row


def test_quality_metrics_api(bio_client):
    r = bio_client.get("/api/analytics/quality-metrics")
    assert r.status_code == 200
    body = r.get_json()
    assert body["review_status"].get("confirmed_bird", 0) == 9
    assert "decision_state" in body
    assert "override_rate" in body


def test_quality_metrics_api_empty(bio_empty_client):
    r = bio_empty_client.get("/api/analytics/quality-metrics")
    assert r.status_code == 200
    body = r.get_json()
    assert body["review_status"] == {}
    assert body["override_rate"] == 0.0
