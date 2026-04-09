"""Tests for review blueprint decision actions."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from core.events import build_bird_events
from web.blueprints.review import _build_review_event_member, build_review_continuity_batches


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


def test_review_decision_accepts_trash_alias(client):
    mock_conn = MagicMock()

    with patch("web.blueprints.review.db_service") as mock_db:
        mock_db.get_connection.return_value = mock_conn
        mock_db.update_review_status.return_value = 1

        response = client.post(
            "/api/review/decision",
            json={"filenames": ["review-item.jpg"], "action": "trash"},
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["action"] == "trash"
    assert data["review_status"] == "no_bird"
    mock_db.update_review_status.assert_called_once_with(
        mock_conn, ["review-item.jpg"], "no_bird"
    )
    mock_conn.close.assert_called_once()


def test_review_approve_requires_manual_species_and_bbox(client):
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchone.return_value = {
        "manual_bbox_review": None,
        "manual_species_override": None,
        "species_source": None,
    }

    with patch("web.blueprints.review.db_service") as mock_db:
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/approve",
            json={"filename": "review-item.jpg", "detection_id": 17},
        )

    assert response.status_code == 409
    data = response.get_json()
    assert data["status"] == "error"
    assert "species selection is required" in data["message"]


def test_build_review_event_member_labels_context_frame_as_in_gallery():
    row = {
        "timestamp": "20260408_155825",
        "manual_species_override": "Columba_palumbus",
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


def test_review_approve_requires_bbox_after_species_selection(client):
    mock_conn = MagicMock()

    with patch("web.blueprints.review.db_service") as mock_db:
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/approve",
            json={
                "filename": "review-item.jpg",
                "detection_id": 17,
                "species": "Parus_major",
            },
        )

    assert response.status_code == 409
    data = response.get_json()
    assert data["status"] == "error"
    assert "Bounding box review is required" in data["message"]


def test_review_approve_confirms_after_manual_species_and_bbox(client):
    mock_conn = MagicMock()
    fetchone_results = iter(
        [
            {
                "manual_bbox_review": "correct",
                "manual_species_override": "Parus_major",
                "species_source": "manual",
            },
            [1],
            [0],
        ]
    )
    mock_conn.execute.return_value.fetchone.side_effect = lambda: next(fetchone_results)

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._get_allowed_review_species") as mock_allowed,
        patch("web.blueprints.review.gallery_service.invalidate_cache") as mock_invalidate,
    ):
        mock_allowed.return_value = {"Parus_major"}
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/approve",
            json={
                "filename": "review-item.jpg",
                "detection_id": 17,
                "species": "Parus_major",
                "bbox_review": "correct",
            },
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["review_status"] == "confirmed_bird"
    assert data["gallery_visible"] is True
    assert "now visible in the gallery" in data["message"]
    mock_db.update_review_status.assert_called_once_with(
        mock_conn, ["review-item.jpg"], "confirmed_bird"
    )
    mock_invalidate.assert_called_once()


def test_review_approve_keeps_image_untagged_when_unresolved_siblings_remain(client):
    mock_conn = MagicMock()
    fetchone_results = iter(
        [
            {
                "manual_bbox_review": "correct",
                "manual_species_override": "Parus_major",
                "species_source": "manual",
            },
            [3],
            [2],
        ]
    )
    mock_conn.execute.return_value.fetchone.side_effect = lambda: next(fetchone_results)

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._get_allowed_review_species") as mock_allowed,
        patch("web.blueprints.review.gallery_service.invalidate_cache"),
    ):
        mock_allowed.return_value = {"Parus_major"}
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/approve",
            json={
                "filename": "review-item.jpg",
                "detection_id": 17,
                "species": "Parus_major",
                "bbox_review": "correct",
            },
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["review_status"] == "untagged"
    assert data["gallery_visible"] is False
    assert "remains out of the gallery" in data["message"]
    mock_db.update_review_status.assert_not_called()
    assert mock_conn.execute.call_count >= 3


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
        mock_thumb.return_value = "/uploads/derivatives/optimized/2026-04-02/nuthatch.webp"
        quick_species = _build_review_quick_species(
            "Sitta_europaea",
            picker_entries,
            [],
            {"Sitta_europaea": "Kleiber"},
            species_thumbnail_map={"Sitta_europaea": "/uploads/derivatives/optimized/2026-04-02/nuthatch.webp"},
            thumbnail_cache_key="review:DE",
        )

    assert quick_species[0]["scientific"] == "Sitta_europaea"
    assert quick_species[0]["thumb_url"] == "/uploads/derivatives/optimized/2026-04-02/nuthatch.webp"


def test_stamp_species_display_adds_ref_image_urls_to_event_quick_species():
    from web.blueprints.review import _stamp_species_display_on_event

    event = {
        "candidate_species": "Parus_major",
        "members": [],
        "quick_species": [
            {"scientific": "Parus_major"},
            {"scientific": "Cyanistes_caeruleus"},
        ],
    }

    with patch("web.blueprints.review.resolve_species_ref_image_url") as mock_ref:
        mock_ref.side_effect = (
            lambda species_key: f"/assets/review_species/{species_key}.png"
        )
        _stamp_species_display_on_event(
            event,
            {"Parus_major": 1, "Cyanistes_caeruleus": 2},
        )

    assert (
        event["quick_species"][0]["species_ref_image_url"]
        == "/assets/review_species/Parus_major.png"
    )
    assert (
        event["quick_species"][1]["species_ref_image_url"]
        == "/assets/review_species/Cyanistes_caeruleus.png"
    )


def test_review_event_approve_confirms_event_and_recomputes_gallery_visibility(client):
    mock_conn = MagicMock()

    def execute_side_effect(query, params=None):
        if "SELECT d.detection_id, d.image_filename" in query:
            return _sql_result(fetchall=[{
                "detection_id": 17,
                "image_filename": "review-item.jpg",
                "review_status": "untagged",
            }])
        if "COALESCE(d.decision_state" in query:
            return _sql_result(fetchone=[0])
        if "SELECT COUNT(*)" in query:
            return _sql_result(fetchone=[1])
        return _sql_result()

    mock_conn.execute.side_effect = execute_side_effect

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._get_allowed_review_species") as mock_allowed,
        patch("web.blueprints.review._load_single_review_event") as mock_event,
        patch("web.blueprints.review.gallery_service.invalidate_cache") as mock_invalidate,
    ):
        mock_allowed.return_value = {"Parus_major"}
        mock_event.return_value = {
            "event_key": "bird-event-abc123",
            "detection_ids": [17],
            "candidate_species": "Parus_major",
            "eligibility": "event_eligible",
        }
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/event-approve",
            json={
                "event_key": "bird-event-abc123",
                "detection_ids": [17],
                "species": "Parus_major",
                "bbox_review": "correct",
            },
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["gallery_visible_filenames"] == ["review-item.jpg"]
    mock_db.apply_species_override_many.assert_called_once_with(
        mock_conn, [17], "Parus_major", "manual"
    )
    mock_db.set_manual_bbox_review.assert_called_once_with(mock_conn, 17, "correct")
    mock_db.update_review_status.assert_called_once_with(
        mock_conn, ["review-item.jpg"], "confirmed_bird"
    )
    mock_invalidate.assert_called_once()


def test_review_event_approve_returns_conflict_when_event_disappears(client):
    mock_conn = MagicMock()

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._get_allowed_review_species") as mock_allowed,
        patch("web.blueprints.review._load_single_review_event") as mock_event,
    ):
        mock_allowed.return_value = {"Parus_major"}
        mock_event.return_value = None
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/event-approve",
            json={
                "event_key": "bird-event-abc123",
                "detection_ids": [17],
                "species": "Parus_major",
                "bbox_review": "correct",
            },
        )

    assert response.status_code == 409
    data = response.get_json()
    assert data["status"] == "error"
    assert "no longer exists" in data["message"]


def test_compute_queue_orphans_drops_detection_orphans_inside_events():
    """Detection orphans already in an event are removed from the queue rail."""
    from web.blueprints.review import _compute_queue_orphans

    orphans = [
        {"item_kind": "detection", "item_id": "17", "filename": "a.jpg"},
        {"item_kind": "detection", "item_id": "18", "filename": "b.jpg"},
        {"item_kind": "image", "item_id": "img-99", "filename": "c.jpg"},
    ]
    review_events = [
        {"event_key": "bird-event-x", "detection_ids": [17]},
    ]

    queue_orphans = _compute_queue_orphans(orphans, review_events)

    assert [item["item_id"] for item in queue_orphans] == ["18", "img-99"]


def test_compute_queue_orphans_keeps_everything_when_no_events():
    from web.blueprints.review import _compute_queue_orphans

    orphans = [
        {"item_kind": "detection", "item_id": "17", "filename": "a.jpg"},
        {"item_kind": "image", "item_id": "img-99", "filename": "b.jpg"},
    ]
    queue_orphans = _compute_queue_orphans(orphans, [])
    assert len(queue_orphans) == 2
    assert queue_orphans[0]["item_id"] == "17"


def test_compute_queue_orphans_handles_invalid_detection_ids():
    from web.blueprints.review import _compute_queue_orphans

    orphans = [
        {"item_kind": "detection", "item_id": "abc", "filename": "a.jpg"},
        {"item_kind": "detection", "item_id": None, "filename": "b.jpg"},
    ]
    review_events = [{"event_key": "bird-event-x", "detection_ids": [17]}]

    queue_orphans = _compute_queue_orphans(orphans, review_events)
    # Defensive leftovers (zero/invalid detection_id) stay in the queue
    # rail so they remain reachable.
    assert len(queue_orphans) == 2


def test_review_page_renders_workspace_for_orphan_only_state(client):
    """Regression: orphan-only pages must render the workspace, not the empty state."""
    mock_conn = MagicMock()
    orphan_payload = {
        "item_kind": "image",
        "item_id": "img-1",
        "item_key": "image:img-1",
        "filename": "orphan-1.jpg",
        "review_reason": "no_detection",
        "reason_label": "No detection",
        "thumb_url": "/thumb/orphan-1.jpg",
        "current_species_common": "Unknown",
        "source_image_filename": "orphan-1.jpg",
    }

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._load_review_items") as mock_load_items,
        patch("web.blueprints.review._load_review_events") as mock_load_events,
        patch("web.blueprints.review.load_common_names") as mock_common_names,
    ):
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_load_items.return_value = ([orphan_payload], [])
        mock_load_events.return_value = ([], [], False, set())
        mock_common_names.return_value = {}

        response = client.get("/admin/review")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    # The orphan-only page must render the workspace, not the empty state.
    assert "Review Queue Empty" not in body
    assert 'id="reviewWorkspace"' in body
    assert 'id="reviewQueueBrowser"' in body
    # Event rail must NOT be present (no events).
    assert 'id="reviewEventBrowser"' not in body
    # The default panel type for an orphan-only page is the queue panel.
    assert 'data-panel-type="queue"' in body


def test_review_page_hides_queue_rail_while_event_workspace_is_active(client):
    """Events keep the left rail focused even when queue leftovers exist."""
    mock_conn = MagicMock()
    orphan_payload = {
        "item_kind": "image",
        "item_id": "img-2",
        "item_key": "image:img-2",
        "filename": "orphan-2.jpg",
        "review_reason": "no_detection",
        "reason_label": "No detection",
        "thumb_url": "/thumb/orphan-2.jpg",
        "current_species_common": "Unknown",
        "source_image_filename": "orphan-2.jpg",
    }
    event_payload = {
        "event_key": "bird-event-1",
        "cover_detection_id": 17,
        "eligibility": "event_eligible",
        "eligibility_label": "Event ready",
        "photo_count": 3,
        "window_time": "07:00:00 - 07:12:00",
        "window_date": "07.04.2026",
        "candidate_species": "Parus major",
        "candidate_species_common": "Kohlmeise",
        "detection_ids": [17],
        "members": [],
        "bbox_trail": [],
    }

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._load_review_items") as mock_load_items,
        patch("web.blueprints.review._load_review_events") as mock_load_events,
        patch("web.blueprints.review.load_common_names") as mock_common_names,
    ):
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_load_items.return_value = ([orphan_payload], [])
        mock_load_events.return_value = ([event_payload], [], False, {"Parus major"})
        mock_common_names.return_value = {}

        response = client.get("/admin/review")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert 'id="reviewWorkspace"' in body
    assert 'id="reviewEventBrowser"' in body
    assert 'id="reviewQueueBrowser"' not in body
    assert 'data-panel-type="event"' in body


def test_review_event_approve_keeps_images_hidden_when_open_detections_remain(client):
    mock_conn = MagicMock()

    def execute_side_effect(query, params=None):
        if "SELECT d.detection_id, d.image_filename" in query:
            return _sql_result(fetchall=[{
                "detection_id": 17,
                "image_filename": "review-item.jpg",
                "review_status": "untagged",
            }])
        if "COALESCE(d.decision_state" in query:
            return _sql_result(fetchone=[2])
        if "SELECT COUNT(*)" in query:
            return _sql_result(fetchone=[3])
        return _sql_result()

    mock_conn.execute.side_effect = execute_side_effect

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._get_allowed_review_species") as mock_allowed,
        patch("web.blueprints.review._load_single_review_event") as mock_event,
        patch("web.blueprints.review.gallery_service.invalidate_cache"),
    ):
        mock_allowed.return_value = {"Parus_major"}
        mock_event.return_value = {
            "event_key": "bird-event-abc123",
            "detection_ids": [17],
            "candidate_species": "Parus_major",
            "eligibility": "event_eligible",
        }
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/event-approve",
            json={
                "event_key": "bird-event-abc123",
                "detection_ids": [17],
                "species": "Parus_major",
                "bbox_review": "correct",
            },
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["gallery_visible_filenames"] == []
    assert data["review_status_by_filename"]["review-item.jpg"] == "untagged"
    mock_db.update_review_status.assert_not_called()


def test_review_event_approve_ignores_gallery_anchors_for_event_payload(client):
    """Anchored events may show Gallery frames, but approval only touches review frames."""
    mock_conn = MagicMock()

    def execute_side_effect(query, params=None):
        if "SELECT d.detection_id, d.image_filename" in query:
            assert params == [17]
            return _sql_result(fetchall=[{
                "detection_id": 17,
                "image_filename": "review-item.jpg",
                "review_status": "untagged",
            }])
        if "COALESCE(d.decision_state" in query:
            return _sql_result(fetchone=[0])
        if "SELECT COUNT(*)" in query:
            return _sql_result(fetchone=[1])
        return _sql_result()

    mock_conn.execute.side_effect = execute_side_effect

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._get_allowed_review_species") as mock_allowed,
        patch("web.blueprints.review._load_single_review_event") as mock_event,
        patch("web.blueprints.review.gallery_service.invalidate_cache"),
    ):
        mock_allowed.return_value = {"Parus_major"}
        mock_event.return_value = {
            "event_key": "bird-event-abc123",
            "detection_ids": [17, 42],
            "candidate_species": "Parus_major",
            "eligibility": "event_eligible",
            "members": [
                {"best_detection_id": 17, "context_only": False},
                {"best_detection_id": 42, "context_only": True},
            ],
        }
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/event-approve",
            json={
                "event_key": "bird-event-abc123",
                "detection_ids": [17, 42],
                "species": "Parus_major",
                "bbox_review": "correct",
            },
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["detection_ids"] == [17]
    mock_db.apply_species_override_many.assert_called_once_with(
        mock_conn, [17], "Parus_major", "manual"
    )
    mock_db.set_manual_bbox_review.assert_called_once_with(mock_conn, 17, "correct")


def test_review_event_approve_allows_user_selected_species_override(client):
    """Direct event-surface species correction must survive approval.

    The event payload may still carry the old candidate species from the
    server-side grouping pass, but the operator is allowed to choose a
    different valid species locally before clicking `Approve Event`.
    """
    mock_conn = MagicMock()

    def execute_side_effect(query, params=None):
        if "SELECT d.detection_id, d.image_filename" in query:
            return _sql_result(fetchall=[{
                "detection_id": 17,
                "image_filename": "review-item.jpg",
                "review_status": "untagged",
            }])
        if "COALESCE(d.decision_state" in query:
            return _sql_result(fetchone=[0])
        if "SELECT COUNT(*)" in query:
            return _sql_result(fetchone=[1])
        return _sql_result()

    mock_conn.execute.side_effect = execute_side_effect

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._get_allowed_review_species") as mock_allowed,
        patch("web.blueprints.review._load_single_review_event") as mock_event,
        patch("web.blueprints.review.gallery_service.invalidate_cache"),
    ):
        mock_allowed.return_value = {"Parus_major"}
        mock_event.return_value = {
            "event_key": "bird-event-abc123",
            "detection_ids": [17],
            "candidate_species": "Cyanistes_caeruleus",
            "eligibility": "event_eligible",
        }
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/event-approve",
            json={
                "event_key": "bird-event-abc123",
                "detection_ids": [17],
                "species": "Parus_major",
                "bbox_review": "correct",
            },
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    mock_db.apply_species_override_many.assert_called_once_with(
        mock_conn, [17], "Parus_major", "manual"
    )


def test_review_event_trash_moves_images_without_active_detections_to_trash(client):
    mock_conn = MagicMock()

    def execute_side_effect(query, params=None):
        if "SELECT d.detection_id, d.image_filename" in query:
            return _sql_result(fetchall=[{
                "detection_id": 17,
                "image_filename": "review-item.jpg",
            }])
        if "SELECT COUNT(*)" in query:
            return _sql_result(fetchone=[0])
        return _sql_result()

    mock_conn.execute.side_effect = execute_side_effect

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._load_single_review_event") as mock_event,
        patch("web.blueprints.review.gallery_service.invalidate_cache") as mock_invalidate,
    ):
        mock_event.return_value = {
            "event_key": "bird-event-abc123",
            "detection_ids": [17],
        }
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/event-trash",
            json={
                "event_key": "bird-event-abc123",
                "detection_ids": [17],
            },
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["review_status_by_filename"]["review-item.jpg"] == "no_bird"
    assert data["trash_filenames"] == ["review-item.jpg"]
    assert "no active detections left" in data["message"]
    mock_db.reject_detections.assert_called_once_with(mock_conn, [17])
    mock_db.update_review_status.assert_called_once_with(
        mock_conn, ["review-item.jpg"], "no_bird"
    )
    mock_invalidate.assert_called_once()


def test_review_event_trash_keeps_images_in_review_when_active_detections_remain(client):
    mock_conn = MagicMock()

    def execute_side_effect(query, params=None):
        if "SELECT d.detection_id, d.image_filename" in query:
            return _sql_result(fetchall=[{
                "detection_id": 17,
                "image_filename": "review-item.jpg",
            }])
        if "COALESCE(d.decision_state" in query:
            return _sql_result(fetchone=[1])
        if "SELECT COUNT(*)" in query:
            return _sql_result(fetchone=[2])
        return _sql_result()

    mock_conn.execute.side_effect = execute_side_effect

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._load_single_review_event") as mock_event,
        patch("web.blueprints.review.gallery_service.invalidate_cache"),
    ):
        mock_event.return_value = {
            "event_key": "bird-event-abc123",
            "detection_ids": [17],
        }
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/event-trash",
            json={
                "event_key": "bird-event-abc123",
                "detection_ids": [17],
            },
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["review_status_by_filename"]["review-item.jpg"] == "untagged"
    assert data["trash_filenames"] == []
    assert data["review_filenames"] == ["review-item.jpg"]
    mock_db.reject_detections.assert_called_once_with(mock_conn, [17])
    mock_db.update_review_status.assert_not_called()


# ---------------------------------------------------------------------------
# Continuity-batch approval path (no event_key)
# ---------------------------------------------------------------------------


def test_review_event_approve_batch_refuses_gallery_anchor_detection_ids(client):
    """Batch approvals must refuse anchor detections.

    A continuity batch approval POSTs ``detection_ids`` without an
    ``event_key`` because the ids span sibling events. The backend must
    refuse outright if any of those detections belong to an image
    already confirmed in the gallery (``review_status='confirmed_bird'``),
    since those are read-only context anchors.
    """
    mock_conn = MagicMock()

    def execute_side_effect(query, params=None):
        if "SELECT d.detection_id, d.image_filename" in query:
            return _sql_result(fetchall=[
                {
                    "detection_id": 17,
                    "image_filename": "review-item.jpg",
                    "review_status": "untagged",
                },
                {
                    "detection_id": 42,
                    "image_filename": "gallery-anchor.jpg",
                    "review_status": "confirmed_bird",
                },
            ])
        return _sql_result()

    mock_conn.execute.side_effect = execute_side_effect

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._get_allowed_review_species") as mock_allowed,
    ):
        mock_allowed.return_value = {"Pica_pica"}
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/event-approve",
            json={
                # Batch path: no event_key.
                "detection_ids": [17, 42],
                "species": "Pica_pica",
                "bbox_review": "correct",
            },
        )

    assert response.status_code == 409
    data = response.get_json()
    assert data["status"] == "error"
    assert "Gallery anchors" in data["message"]
    assert data["anchor_filenames"] == ["gallery-anchor.jpg"]
    mock_db.apply_species_override_many.assert_not_called()
    mock_db.set_manual_bbox_review.assert_not_called()


def test_review_event_approve_batch_confirms_actionable_ids_without_event_key(client):
    """Submitting only review detection ids works for batch approval.

    When the operator clicks `Approve Batch` the client POSTs
    ``detection_ids`` that span multiple sibling events. The backend
    skips the strict ``event_key``/``detection_ids`` parity check and
    confirms every submitted id, as long as none of them belong to a
    Gallery anchor.
    """
    mock_conn = MagicMock()

    def execute_side_effect(query, params=None):
        if "SELECT d.detection_id, d.image_filename" in query:
            return _sql_result(fetchall=[
                {
                    "detection_id": 17,
                    "image_filename": "spatz-frame.jpg",
                    "review_status": "untagged",
                },
                {
                    "detection_id": 18,
                    "image_filename": "kleiber-frame.jpg",
                    "review_status": "untagged",
                },
            ])
        if "COALESCE(d.decision_state" in query:
            return _sql_result(fetchone=[0])
        if "SELECT COUNT(*)" in query:
            return _sql_result(fetchone=[1])
        return _sql_result()

    mock_conn.execute.side_effect = execute_side_effect

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._get_allowed_review_species") as mock_allowed,
        patch("web.blueprints.review.gallery_service.invalidate_cache"),
    ):
        mock_allowed.return_value = {"Pica_pica"}
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/event-approve",
            json={
                "detection_ids": [17, 18],
                "species": "Pica_pica",
                "bbox_review": "correct",
            },
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["event_key"] is None
    assert sorted(data["detection_ids"]) == [17, 18]
    mock_db.apply_species_override_many.assert_called_once_with(
        mock_conn, [17, 18], "Pica_pica", "manual"
    )
    assert mock_db.set_manual_bbox_review.call_count == 2


# ---------------------------------------------------------------------------
# Review continuity-batch builder
#
# These tests exercise build_review_continuity_batches directly. The
# batch builder is a pure function over BirdEvent objects, so we feed it
# real events constructed via build_bird_events instead of mocking the
# Flask request lifecycle.
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
        "manual_species_override": None,
        "species_key": species,
        "species_source": "classifier",
        "sibling_detection_count": 1,
        "context_only": context_only,
    }


def test_continuity_batch_groups_split_events_around_a_single_anchor():
    detections = [
        # Confirmed Pica anchor (read-only Gallery context).
        _detection_dict(101, "20260407_120000", "Pica_pica", context_only=True),
        _detection_dict(102, "20260407_120020", "Pica_pica", context_only=True),
        # Two actionable Review events the AI predicted as different species
        # but that should be reviewed together against the Pica anchor.
        _detection_dict(201, "20260407_121000", "Passer_domesticus"),
        _detection_dict(202, "20260407_121010", "Passer_domesticus"),
        _detection_dict(301, "20260407_122000", "Sitta_europaea"),
    ]
    raw_events = build_bird_events(detections)
    batches = build_review_continuity_batches(raw_events)

    assert len(batches) == 1
    batch = batches[0]
    # The two split actionable events both belong to the batch.
    assert len(batch["event_keys"]) == 2
    # Single confirmed-species anchor → recommended species fires.
    assert batch["recommended_species"] == "Pica_pica"
    # Action targets must only include actionable detections.
    assert sorted(batch["review_detection_ids"]) == [201, 202, 301]
    assert 101 in batch["context_detection_ids"]
    assert 102 in batch["context_detection_ids"]
    assert 101 not in batch["review_detection_ids"]
    assert 102 not in batch["review_detection_ids"]
    # Anchor species summary reports the Pica context.
    assert batch["context_species_summary"].get("Pica_pica") == 2
    # batch_bbox_map carries both context and actionable bbox entries
    # with the context_only flag and an event_key tag.
    assert batch["batch_bbox_map"], "batch_bbox_map must not be empty"
    assert any(point["context_only"] for point in batch["batch_bbox_map"])
    assert any(not point["context_only"] for point in batch["batch_bbox_map"])
    assert all("event_key" in point for point in batch["batch_bbox_map"])


def test_continuity_batch_drops_recommendation_when_anchors_disagree():
    detections = [
        _detection_dict(101, "20260407_120000", "Pica_pica", context_only=True),
        _detection_dict(102, "20260407_120030", "Parus_major", context_only=True),
        _detection_dict(201, "20260407_121000", "Passer_domesticus"),
    ]
    raw_events = build_bird_events(detections)
    batches = build_review_continuity_batches(raw_events)

    assert len(batches) == 1
    batch = batches[0]
    assert batch["recommended_species"] is None
    # Both confirmed anchors must show up in the species summary.
    assert set(batch["context_species_summary"].keys()) == {
        "Pica_pica",
        "Parus_major",
    }


def test_no_continuity_batch_without_a_confirmed_anchor():
    """Pure unanchored time coincidence is not enough to form a batch."""
    detections = [
        _detection_dict(201, "20260407_121000", "Passer_domesticus"),
        _detection_dict(202, "20260407_121005", "Passer_domesticus"),
        _detection_dict(301, "20260407_122000", "Sitta_europaea"),
    ]
    raw_events = build_bird_events(detections)
    batches = build_review_continuity_batches(raw_events)
    assert batches == []


def test_no_continuity_batch_when_anchor_has_no_actionable_neighbour():
    """A confirmed Gallery anchor without nearby Review work emits nothing."""
    detections = [
        _detection_dict(101, "20260407_120000", "Pica_pica", context_only=True),
        _detection_dict(102, "20260407_120020", "Pica_pica", context_only=True),
        # Actionable Review event hours later — outside the ±30 min window.
        _detection_dict(201, "20260407_180000", "Passer_domesticus"),
    ]
    raw_events = build_bird_events(detections)
    batches = build_review_continuity_batches(raw_events)
    assert batches == []


# ─────────────────────────────────────────────────────────────────────
# Mixed-event resolve endpoint
# ─────────────────────────────────────────────────────────────────────


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


def test_review_event_resolve_commits_keep_and_trash_in_one_call(client):
    rows = [
        {
            "detection_id": 17,
            "image_filename": "frame-a.jpg",
            "review_status": "untagged",
            "manual_species_override": None,
            "species_source": None,
        },
        {
            "detection_id": 18,
            "image_filename": "frame-b.jpg",
            "review_status": "untagged",
            "manual_species_override": None,
            "species_source": None,
        },
    ]
    mock_conn = _make_resolve_conn(rows)

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._get_allowed_review_species") as mock_allowed,
        patch("web.blueprints.review._load_single_review_event") as mock_event,
        patch("web.blueprints.review.gallery_service.invalidate_cache") as mock_invalidate,
    ):
        mock_allowed.return_value = {"Columba_palumbus"}
        mock_event.return_value = {
            "event_key": "bird-event-abc",
            "detection_ids": [17, 18],
            "candidate_species": "Columba_palumbus",
            "eligibility": "event_eligible",
        }
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/event-resolve",
            json={
                "event_key": "bird-event-abc",
                "keep_detection_ids": [17],
                "trash_detection_ids": [18],
                "species": "Columba_palumbus",
                "bbox_review": "correct",
            },
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["keep_detection_ids"] == [17]
    assert data["trash_detection_ids"] == [18]

    # Keep ids get species override; trash ids go through reject_detections.
    mock_db.apply_species_override_many.assert_called_once_with(
        mock_conn, [17], "Columba_palumbus", "manual"
    )
    mock_db.set_manual_bbox_review.assert_called_once_with(mock_conn, 17, "correct")
    mock_db.reject_detections.assert_called_once_with(mock_conn, [18])
    mock_invalidate.assert_called_once()


def test_review_event_resolve_rejects_empty_keep_list(client):
    response = client.post(
        "/api/review/event-resolve",
        json={
            "event_key": "bird-event-abc",
            "keep_detection_ids": [],
            "trash_detection_ids": [18],
            "species": "Columba_palumbus",
            "bbox_review": "correct",
        },
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert "keep_detection_ids must not be empty" in data["message"]


def test_review_event_resolve_rejects_overlapping_keep_and_trash(client):
    response = client.post(
        "/api/review/event-resolve",
        json={
            "event_key": "bird-event-abc",
            "keep_detection_ids": [17, 18],
            "trash_detection_ids": [18],
            "species": "Columba_palumbus",
            "bbox_review": "correct",
        },
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "disjoint" in data["message"]
    assert data["overlap"] == [18]


def test_review_event_resolve_rejects_partial_event_coverage(client):
    """keep+trash must cover every detection of the loaded event."""
    mock_conn = _make_resolve_conn([])

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._get_allowed_review_species") as mock_allowed,
        patch("web.blueprints.review._load_single_review_event") as mock_event,
    ):
        mock_allowed.return_value = {"Columba_palumbus"}
        # Event actually has three detections; the caller only covered two.
        mock_event.return_value = {
            "event_key": "bird-event-abc",
            "detection_ids": [17, 18, 19],
            "candidate_species": "Columba_palumbus",
            "eligibility": "event_eligible",
        }
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/event-resolve",
            json={
                "event_key": "bird-event-abc",
                "keep_detection_ids": [17],
                "trash_detection_ids": [18],
                "species": "Columba_palumbus",
                "bbox_review": "correct",
            },
        )

    assert response.status_code == 409
    data = response.get_json()
    assert "changed and must be reloaded" in data["message"]


def test_review_event_resolve_refuses_gallery_anchors(client):
    rows = [
        {
            "detection_id": 17,
            "image_filename": "frame-a.jpg",
            "review_status": "untagged",
            "manual_species_override": None,
            "species_source": None,
        },
        {
            "detection_id": 18,
            "image_filename": "anchor.jpg",
            "review_status": "confirmed_bird",
            "manual_species_override": None,
            "species_source": None,
        },
    ]
    mock_conn = _make_resolve_conn(rows)

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._get_allowed_review_species") as mock_allowed,
        patch("web.blueprints.review._load_single_review_event") as mock_event,
    ):
        mock_allowed.return_value = {"Columba_palumbus"}
        mock_event.return_value = {
            "event_key": "bird-event-abc",
            "detection_ids": [17, 18],
            "candidate_species": "Columba_palumbus",
            "eligibility": "event_eligible",
        }
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/event-resolve",
            json={
                "event_key": "bird-event-abc",
                "keep_detection_ids": [17],
                "trash_detection_ids": [18],
                "species": "Columba_palumbus",
                "bbox_review": "correct",
            },
        )

    assert response.status_code == 409
    data = response.get_json()
    assert "Gallery anchors" in data["message"]
    assert data["anchor_filenames"] == ["anchor.jpg"]
    mock_db.reject_detections.assert_not_called()


def test_review_event_resolve_preserves_per_frame_manual_override(client):
    """Per-frame wins: a Keep frame already relabelled via bulk/relabel
    must keep its manual species override. The event-level species from
    the right control rail only stamps frames that have no override yet.
    """
    rows = [
        # Frame 17 already relabelled to Columba_palumbus (Tauben) via
        # the per-cell WmSpeciesPicker path.
        {
            "detection_id": 17,
            "image_filename": "frame-a.jpg",
            "review_status": "untagged",
            "manual_species_override": "Columba_palumbus",
            "species_source": "manual",
        },
        # Frame 18 is still untouched — should receive the event species.
        {
            "detection_id": 18,
            "image_filename": "frame-b.jpg",
            "review_status": "untagged",
            "manual_species_override": None,
            "species_source": None,
        },
    ]
    mock_conn = _make_resolve_conn(rows)

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._get_allowed_review_species") as mock_allowed,
        patch("web.blueprints.review._load_single_review_event") as mock_event,
        patch("web.blueprints.review.gallery_service.invalidate_cache"),
    ):
        mock_allowed.return_value = {"Pica_pica", "Columba_palumbus"}
        mock_event.return_value = {
            "event_key": "bird-event-abc",
            "detection_ids": [17, 18],
            "candidate_species": "Pica_pica",
            "eligibility": "event_eligible",
        }
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/event-resolve",
            json={
                "event_key": "bird-event-abc",
                "keep_detection_ids": [17, 18],
                "trash_detection_ids": [],
                "species": "Pica_pica",
                "bbox_review": "correct",
            },
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"

    # Only frame 18 (no prior manual override) should be stamped with
    # the event-level species. Frame 17 keeps Columba_palumbus.
    mock_db.apply_species_override_many.assert_called_once_with(
        mock_conn, [18], "Pica_pica", "manual"
    )
    # bbox_review still applies to every keep frame regardless.
    bbox_calls = mock_db.set_manual_bbox_review.call_args_list
    bbox_ids = sorted(call.args[1] for call in bbox_calls)
    assert bbox_ids == [17, 18]


def test_review_event_resolve_preserves_override_when_all_keeps_are_manual(client):
    """Edge case: every Keep frame is already hand-relabelled. The
    bulk `apply_species_override_many` call must not fire at all."""
    rows = [
        {
            "detection_id": 17,
            "image_filename": "frame-a.jpg",
            "review_status": "untagged",
            "manual_species_override": "Columba_palumbus",
            "species_source": "manual",
        },
        {
            "detection_id": 18,
            "image_filename": "frame-b.jpg",
            "review_status": "untagged",
            "manual_species_override": "Columba_palumbus",
            "species_source": "manual",
        },
    ]
    mock_conn = _make_resolve_conn(rows)

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._get_allowed_review_species") as mock_allowed,
        patch("web.blueprints.review._load_single_review_event") as mock_event,
        patch("web.blueprints.review.gallery_service.invalidate_cache"),
    ):
        mock_allowed.return_value = {"Pica_pica", "Columba_palumbus"}
        mock_event.return_value = {
            "event_key": "bird-event-abc",
            "detection_ids": [17, 18],
            "candidate_species": "Pica_pica",
            "eligibility": "event_eligible",
        }
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/event-resolve",
            json={
                "event_key": "bird-event-abc",
                "keep_detection_ids": [17, 18],
                "trash_detection_ids": [],
                "species": "Pica_pica",
                "bbox_review": "correct",
            },
        )

    assert response.status_code == 200
    mock_db.apply_species_override_many.assert_not_called()


def test_review_event_resolve_requires_species_and_bbox(client):
    # Missing species.
    r1 = client.post(
        "/api/review/event-resolve",
        json={
            "keep_detection_ids": [17],
            "trash_detection_ids": [18],
            "bbox_review": "correct",
        },
    )
    assert r1.status_code == 409
    assert "species" in r1.get_json()["message"].lower()

    # Missing bbox_review.
    r2 = client.post(
        "/api/review/event-resolve",
        json={
            "keep_detection_ids": [17],
            "trash_detection_ids": [18],
            "species": "Columba_palumbus",
        },
    )
    assert r2.status_code == 409
    assert "bounding box" in r2.get_json()["message"].lower()
