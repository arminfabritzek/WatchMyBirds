"""Tests for the Rescan Safety Layer.

Verifies:
- Moderation rescan writes to rescan_proposals, NEVER to detections.
- Orphan path still writes directly to detections (unchanged behavior).
- Proposal apply is idempotent.
- submit_moderation_rescan bypasses orphan eligibility.
- Proposal status flow works correctly.
"""

import sqlite3
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_test_db(tmp_path) -> sqlite3.Connection:
    """Create an in-memory-like SQLite DB with full schema for testing."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")

    conn.execute("""
        CREATE TABLE images (
            filename TEXT PRIMARY KEY,
            timestamp TEXT,
            review_status TEXT DEFAULT 'untagged',
            deep_scan_last_result TEXT,
            deep_scan_attempt_count INTEGER DEFAULT 0,
            deep_scan_last_attempt_at TEXT,
            content_hash TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE detections (
            detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_filename TEXT NOT NULL,
            od_class_name TEXT,
            od_confidence REAL,
            status TEXT DEFAULT 'active',
            score REAL,
            FOREIGN KEY(image_filename) REFERENCES images(filename) ON DELETE CASCADE
        )
    """)
    conn.execute("""
        CREATE TABLE classifications (
            classification_id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id INTEGER NOT NULL,
            cls_class_name TEXT,
            cls_confidence REAL,
            FOREIGN KEY(detection_id) REFERENCES detections(detection_id) ON DELETE CASCADE
        )
    """)
    conn.execute("""
        CREATE TABLE rescan_proposals (
            proposal_id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            target_detection_id INTEGER,
            image_filename TEXT NOT NULL,
            suggested_species TEXT,
            suggested_confidence REAL,
            suggested_score REAL,
            bbox_x REAL, bbox_y REAL, bbox_w REAL, bbox_h REAL,
            topk_json TEXT,
            source_model TEXT,
            status TEXT DEFAULT 'queued',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            applied_at TEXT,
            FOREIGN KEY(image_filename) REFERENCES images(filename) ON DELETE CASCADE
        )
    """)
    conn.commit()
    return conn


@dataclass
class _FakePayload:
    bbox: tuple = (10, 10, 100, 100)
    confidence: float = 0.9
    class_name: str = "bird"
    cls_class_name: str = "Parus_major"
    cls_confidence: float = 0.85
    score: float = 0.8
    agreement_score: float = 0.7
    decision_state: str = "confirmed"
    bbox_quality: float = 0.9
    unknown_score: float = 0.1
    decision_reasons: str = '{"top_k": ["Parus_major", "Cyanistes_caeruleus"]}'
    policy_version: str = "v1"


# ---------------------------------------------------------------------------
# _save_rescan_proposals: writes to proposals, NOT detections
# ---------------------------------------------------------------------------


class TestSaveRescanProposals:
    def test_writes_to_proposals_not_detections(self, tmp_path):
        conn = _create_test_db(tmp_path)

        # Insert image + detection
        conn.execute(
            "INSERT INTO images (filename, timestamp) VALUES ('img_a.jpg', '20260306')"
        )
        conn.execute(
            "INSERT INTO detections (image_filename, od_class_name) VALUES ('img_a.jpg', 'Unknown')"
        )
        conn.commit()

        det_row = conn.execute("SELECT detection_id FROM detections").fetchone()
        det_id = det_row["detection_id"]

        # Count before
        det_count_before = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]

        # Save proposals using the real function
        from web.services.analysis_service import _save_rescan_proposals

        payloads = [_FakePayload()]
        with patch("web.services.analysis_service.db_service") as mock_db:
            # Make closing_connection return our test conn
            mock_db.closing_connection.return_value.__enter__ = MagicMock(
                return_value=conn
            )
            mock_db.closing_connection.return_value.__exit__ = MagicMock(
                return_value=False
            )

            saved = _save_rescan_proposals(
                filename="img_a.jpg",
                job_id="test-job-1",
                target_detection_ids=[det_id],
                detection_payloads=payloads,
                source_models=["deep_scan_yolo:cls_v1"],
            )

        assert saved == 1

        # Detections table must be UNCHANGED
        det_count_after = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        assert det_count_after == det_count_before, (
            "Rescan must NOT write to detections!"
        )

        # Check proposals table
        proposals = conn.execute("SELECT * FROM rescan_proposals").fetchall()
        assert len(proposals) == 1
        p = dict(proposals[0])
        assert p["job_id"] == "test-job-1"
        assert p["target_detection_id"] == det_id
        assert p["suggested_species"] == "Parus_major"
        assert p["status"] == "ready"

        conn.close()


# ---------------------------------------------------------------------------
# process_deep_analysis_job: mode routing
# ---------------------------------------------------------------------------


class TestProcessDeepAnalysisJobModeRouting:
    """Verify that moderation_rescan mode writes proposals, not detections."""

    @patch("web.services.analysis_service.cv2")
    @patch("web.services.analysis_service.gallery_service")
    @patch("web.services.analysis_service.db_service")
    @patch("web.services.analysis_service._save_rescan_proposals")
    @patch("web.services.analysis_service._build_detection_payload")
    def test_moderation_rescan_calls_save_proposals(
        self, mock_build, mock_save_proposals, mock_db, mock_gallery, mock_cv2
    ):
        """mode=moderation_rescan → _save_rescan_proposals, NOT persistence_service."""
        from web.services.analysis_service import process_deep_analysis_job

        # Setup mocks
        mock_cv2.imread.return_value = MagicMock()  # fake frame
        mock_gallery.get_image_paths.return_value = {"original": "/tmp/test.jpg"}
        mock_build.return_value = (_FakePayload(), "cls_model_v1")

        dm = MagicMock()
        dm.run_exhaustive_scan.return_value = [
            {
                "x1": 10,
                "y1": 10,
                "x2": 100,
                "y2": 100,
                "confidence": 0.9,
                "method": "yolo",
            }
        ]

        job_data = {
            "filename": "img_test.jpg",
            "mode": "moderation_rescan",
            "job_id": "test-job-2",
            "target_detection_ids": [42],
        }

        process_deep_analysis_job(dm, job_data)

        # _save_rescan_proposals MUST be called
        mock_save_proposals.assert_called_once()
        call_kwargs = mock_save_proposals.call_args
        assert call_kwargs[1].get("job_id", call_kwargs[0][1]) == "test-job-2"

        # persistence_service.save_detection must NOT be called
        dm.persistence_service.save_detection.assert_not_called()

    @patch("web.services.analysis_service.cv2")
    @patch("web.services.analysis_service.gallery_service")
    @patch("web.services.analysis_service.db_service")
    @patch("web.services.analysis_service.check_deep_analysis_eligibility")
    @patch("web.services.analysis_service._build_detection_payload")
    def test_orphan_mode_saves_to_detections(
        self, mock_build, mock_eligible, mock_db, mock_gallery, mock_cv2
    ):
        """mode=orphan → persistence_service.save_detection (existing behavior)."""
        from web.services.analysis_service import process_deep_analysis_job

        mock_eligible.return_value = (True, "")
        mock_cv2.imread.return_value = MagicMock()
        mock_gallery.get_image_paths.return_value = {"original": "/tmp/test.jpg"}
        mock_build.return_value = (_FakePayload(), "cls_model_v1")

        mock_conn = MagicMock()
        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, key: 0  # existing_count = 0
        mock_conn.execute.return_value.fetchone.return_value = mock_row
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        dm = MagicMock()
        dm.run_exhaustive_scan.return_value = [
            {
                "x1": 10,
                "y1": 10,
                "x2": 100,
                "y2": 100,
                "confidence": 0.9,
                "method": "yolo",
            }
        ]
        save_result = MagicMock()
        save_result.success = True
        dm.persistence_service.save_detection.return_value = save_result

        job_data = {"filename": "orphan.jpg"}  # no mode = defaults to orphan

        process_deep_analysis_job(dm, job_data)

        # persistence_service.save_detection MUST be called
        dm.persistence_service.save_detection.assert_called_once()


# ---------------------------------------------------------------------------
# submit_moderation_rescan: bypasses orphan eligibility
# ---------------------------------------------------------------------------


class TestSubmitModerationRescan:
    @patch("web.services.analysis_service.analysis_queue")
    @patch("web.services.analysis_service.db_service")
    def test_submits_even_with_existing_detections(self, mock_db, mock_queue):
        from web.services.analysis_service import submit_moderation_rescan

        # Image exists
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = {"1": 1}
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        mock_queue.enqueue.return_value = True

        result = submit_moderation_rescan(
            "img.jpg", job_id="j1", target_detection_ids=[1, 2]
        )
        assert result is True

        # Check enqueued data includes mode
        enqueued_data = mock_queue.enqueue.call_args[0][0]
        assert enqueued_data["mode"] == "moderation_rescan"
        assert enqueued_data["job_id"] == "j1"
        assert enqueued_data["target_detection_ids"] == [1, 2]

    @patch("web.services.analysis_service.db_service")
    def test_returns_false_if_image_not_found(self, mock_db):
        from web.services.analysis_service import submit_moderation_rescan

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = None
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        result = submit_moderation_rescan("nonexistent.jpg", job_id="j2")
        assert result is False


# ---------------------------------------------------------------------------
# Proposal apply: idempotent + status flow
# ---------------------------------------------------------------------------


@pytest.fixture
def proposal_app():
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.secret_key = "test-secret-key"

    from web.blueprints.auth import auth_bp
    from web.blueprints.moderation import moderation_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(moderation_bp)
    return app


@pytest.fixture
def proposal_client(proposal_app):
    with proposal_app.test_client() as c:
        with c.session_transaction() as sess:
            sess["authenticated"] = True
        yield c


class _DictRow(dict):
    """Mimics sqlite3.Row for dict() conversion in blueprint code."""

    def __getitem__(self, key):
        return super().__getitem__(key)


class TestProposalApply:
    @patch("web.blueprints.moderation.gallery_service")
    @patch("web.blueprints.moderation.db_service")
    def test_apply_ready_proposal_succeeds(
        self, mock_db, mock_gallery, proposal_client
    ):
        mock_conn = MagicMock()
        mock_row = _DictRow(
            proposal_id=1,
            status="ready",
            target_detection_id=42,
            suggested_species="Parus_major",
        )
        mock_conn.execute.return_value.fetchone.return_value = mock_row
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        resp = proposal_client.post("/api/moderation/rescan-proposals/1/apply")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert data["applied_species"] == "Parus_major"

    @patch("web.blueprints.moderation.db_service")
    def test_apply_already_applied_is_idempotent(self, mock_db, proposal_client):
        mock_conn = MagicMock()
        mock_row = _DictRow(
            proposal_id=1, status="applied", suggested_species="Parus_major"
        )
        mock_conn.execute.return_value.fetchone.return_value = mock_row
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        resp = proposal_client.post("/api/moderation/rescan-proposals/1/apply")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert data.get("already_applied") is True

    @patch("web.blueprints.moderation.db_service")
    def test_apply_discarded_returns_409(self, mock_db, proposal_client):
        mock_conn = MagicMock()
        mock_row = _DictRow(proposal_id=1, status="discarded")
        mock_conn.execute.return_value.fetchone.return_value = mock_row
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        resp = proposal_client.post("/api/moderation/rescan-proposals/1/apply")
        assert resp.status_code == 409
