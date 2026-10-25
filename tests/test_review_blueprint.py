"""Tests for review blueprint decision actions."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask
from flask.testing import FlaskClient

from web.blueprints.review import (
    _build_review_event_member,
)


def _sql_result(*, fetchone=None, fetchall=None):
    result = MagicMock()
    if fetchone is not None:
        result.fetchone.return_value = fetchone
    if fetchall is not None:
        result.fetchall.return_value = fetchall
    return result


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def app():
    project_root = _project_root()
    app = Flask(
        __name__,
        template_folder=str(project_root / "templates"),
        static_folder=str(project_root / "assets"),
    )
    app.config["TESTING"] = True
    app.secret_key = "test-secret-key"

    from web.blueprints.auth import auth_bp
    from web.blueprints.review import review_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(review_bp)
    return app


@pytest.fixture
def client(app):
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["authenticated"] = True
        yield client


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/api/review/approve"),
        ("POST", "/api/review/bbox-review"),
        ("POST", "/api/review/event-approve"),
        ("POST", "/api/review/event-resolve"),
        ("POST", "/api/review/event-trash"),
        ("POST", "/api/review/event-unresolved"),
        ("POST", "/api/review/quick-species"),
        ("GET", "/api/review/cleanup/preview"),
        ("POST", "/api/review/cleanup/run"),
    ],
)
def test_removed_review_routes_return_not_found(client, method, path) -> None:
    assert client.open(path, method=method).status_code == 404


def test_build_review_event_member_labels_context_frame_as_in_gallery():
    row = {
        "timestamp": "20260408_155825",
        "human_species_key": "Columba_palumbus",
        "cls_class_name": "Pica_pica",
        "context_only": True,
    }

    with patch("web.blueprints.review._build_review_item") as mock_build_item:
        mock_build_item.return_value = {
            "review_reason": "low_score",
            "reason_label": "Low Score (87%)",
        }

        member = _build_review_event_member(
            row,
            conn=None,
            species_locale="de",
            output_dir="/tmp",
            common_names={"Columba_palumbus": "Ringeltaube"},
            recent_species=[],
        )

    assert member["context_only"] is True
    assert member["candidate_species"] == "Columba_palumbus"
    assert member["candidate_species_common"] == "Ringeltaube"
    assert member["review_reason"] == "context"
    assert member["reason_label"] == "In Gallery"


def test_quick_species_entries_include_server_resolved_thumb_urls():
    from web.blueprints.review import _build_review_quick_species

    picker_entries = [
        {
            "scientific": "Sitta_europaea",
            "common": "Kleiber",
            "source": "prediction",
            "score": 0.73,
        }
    ]

    with patch("web.blueprints.review.resolve_species_thumbnail_url") as mock_thumb:
        mock_thumb.return_value = (
            "/uploads/derivatives/optimized/2026-04-02/nuthatch.webp"
        )
        quick_species = _build_review_quick_species(
            "Sitta_europaea",
            picker_entries,
            [],
            {"Sitta_europaea": "Kleiber"},
            species_thumbnail_map={
                "Sitta_europaea": "/uploads/derivatives/optimized/2026-04-02/nuthatch.webp"
            },
            thumbnail_cache_key="review:DE",
        )

    assert quick_species[0]["scientific"] == "Sitta_europaea"
    assert (
        quick_species[0]["thumb_url"]
        == "/uploads/derivatives/optimized/2026-04-02/nuthatch.webp"
    )


def test_load_recent_review_species_filters_non_bird():
    """The ``fetch_recent_review_species`` aggregate sums
    ``effective_species_sql`` which coalesces ``manual_species_override``
    first. A single prior accidental relabel to ``non_bird`` would
    otherwise pollute the "Recent days" pill for 7 days.
    """
    from web.blueprints.review import _load_recent_review_species

    fake_rows = [
        {"species_key": "Parus_major", "hit_count": 12, "last_seen": "20260522_120000"},
        {"species_key": "non_bird", "hit_count": 1, "last_seen": "20260522_210000"},
        {
            "species_key": "Cyanistes_caeruleus",
            "hit_count": 5,
            "last_seen": "20260522_110000",
        },
    ]

    def fake_is_known(key, locale="DE"):
        return key in {"Parus_major", "Cyanistes_caeruleus"}

    with (
        patch(
            "web.blueprints.review.db_service.fetch_recent_review_species",
            return_value=fake_rows,
        ),
        patch("web.blueprints.review.is_known_species", side_effect=fake_is_known),
    ):
        recent = _load_recent_review_species(
            conn=MagicMock(),
            common_names={
                "Parus_major": "Kohlmeise",
                "Cyanistes_caeruleus": "Blaumeise",
            },
        )

    assert [r["scientific"] for r in recent] == ["Parus_major", "Cyanistes_caeruleus"]


# ---------------------------------------------------------------------------
# Continuity-batch approval path (no event_key)
# ---------------------------------------------------------------------------


def _detection_dict(
    detection_id,
    timestamp,
    species,
    *,
    context_only=False,
    bbox=(0.20, 0.20, 0.18, 0.18),
):
    return {
        "detection_id": detection_id,
        "active_detection_id": detection_id,
        "filename": f"frame-{detection_id}.jpg",
        "timestamp": timestamp,
        "bbox_x": bbox[0],
        "bbox_y": bbox[1],
        "bbox_w": bbox[2],
        "bbox_h": bbox[3],
        "cls_class_name": species,
        "human_species_key": None,
        "species_key": species,
        "species_source": "classifier",
        "sibling_detection_count": 1,
        "context_only": context_only,
    }


def _make_resolve_conn(rows):
    """Build a MagicMock conn whose SELECT on detections returns `rows`."""
    mock_conn = MagicMock()

    def execute_side_effect(query, params=None):
        normalized = " ".join(query.split())
        if "SELECT d.detection_id, d.image_filename" in normalized:
            return _sql_result(fetchall=rows)
        if "COALESCE(d.decision_state" in query:
            return _sql_result(fetchone=[0])
        if "SELECT COUNT(*)" in query:
            return _sql_result(fetchone=[1])
        return _sql_result()

    mock_conn.execute.side_effect = execute_side_effect
    return mock_conn


def test_review_thumb_rejects_out_of_root_path(client):
    """A filename resolving outside the media roots must 404, never serve."""
    escaping = Path("/etc/passwd")

    with patch("web.blueprints.review.gallery_service") as mock_gallery:
        mock_gallery.get_image_paths.return_value = {
            "original": escaping,
            "preview": escaping,
        }
        resp = client.get("/api/review-thumb/anything.jpg")

    assert resp.status_code == 404
    mock_gallery.generate_preview_thumbnail.assert_not_called()


def test_review_thumb_serves_cached_preview_inside_root(client, tmp_path):
    """A contained preview that already exists is served without regeneration."""
    from utils.path_manager import PathManager

    pm = PathManager(str(tmp_path))
    preview = pm.thumbs_dir / "2026-01-01" / "20260101_120000_preview.webp"
    preview.parent.mkdir(parents=True, exist_ok=True)
    preview.write_bytes(b"webp-bytes")
    original = pm.originals_dir / "2026-01-01" / "20260101_120000.jpg"
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes(b"jpg-bytes")

    with (
        patch("web.blueprints.review.path_service") as mock_path_service,
        patch("web.blueprints.review.gallery_service") as mock_gallery,
    ):
        mock_path_service.get_path_manager.return_value = pm
        mock_gallery.get_image_paths.return_value = {
            "original": original,
            "preview": preview,
        }
        resp = client.get("/api/review-thumb/20260101_120000.jpg")

    assert resp.status_code == 200
    mock_gallery.generate_preview_thumbnail.assert_not_called()


def test_removed_analysis_route_returns_404(client: FlaskClient) -> None:
    assert client.post("/api/review/analyze/test.jpg").status_code == 404
