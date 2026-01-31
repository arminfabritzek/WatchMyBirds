# tests/test_file_gc.py
"""
Unit tests for the file_gc module (file garbage collection).
Tests both hard_delete_detections() and hard_delete_images().
"""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory with standard structure."""
    output_dir = tmp_path / "output"

    # Create standard directory structure
    (output_dir / "originals" / "2026-01-31").mkdir(parents=True)
    (output_dir / "derivatives" / "optimized" / "2026-01-31").mkdir(parents=True)
    (output_dir / "derivatives" / "thumbs" / "2026-01-31").mkdir(parents=True)

    return output_dir


@pytest.fixture
def mock_db_connection():
    """Create an in-memory SQLite database with test schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    # Create minimal schema
    conn.executescript("""
        CREATE TABLE images (
            filename TEXT PRIMARY KEY,
            timestamp TEXT,
            review_status TEXT DEFAULT 'untagged'
        );

        CREATE TABLE detections (
            detection_id INTEGER PRIMARY KEY,
            image_filename TEXT,
            thumbnail_path TEXT,
            status TEXT DEFAULT 'active',
            FOREIGN KEY (image_filename) REFERENCES images(filename)
        );
    """)

    return conn


class MockPathManager:
    """Mock PathManager that uses the provided output_dir."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.originals_dir = self.output_dir / "originals"
        self.optimized_dir = self.output_dir / "derivatives" / "optimized"
        self.thumbs_dir = self.output_dir / "derivatives" / "thumbs"

    def get_original_path(self, filename: str) -> Path:
        date_folder = self._extract_date_folder(filename)
        return self.originals_dir / date_folder / filename

    def get_derivative_path(self, filename: str, type: str = "thumb") -> Path:
        date_folder = self._extract_date_folder(filename)
        stem = Path(filename).stem
        derivative_name = f"{stem}.webp"

        if type == "thumb":
            return self.thumbs_dir / date_folder / derivative_name
        elif type == "optimized":
            return self.optimized_dir / date_folder / derivative_name
        else:
            raise ValueError(f"Unknown derivative type: {type}")

    def get_preview_thumb_path(self, filename: str) -> Path:
        date_folder = self._extract_date_folder(filename)
        stem = Path(filename).stem
        return self.thumbs_dir / date_folder / f"{stem}_preview.webp"

    def _extract_date_folder(self, filename: str) -> str:
        # Assumes YYYYMMDD_HHMMSS_ format
        if len(filename) >= 8:
            return f"{filename[:4]}-{filename[4:6]}-{filename[6:8]}"
        return "unknown"


def create_sample_files(temp_output_dir, test_images):
    """Helper to create sample files on disk."""
    date_folder = "2026-01-31"
    for filename, _, status in test_images:
        if status == "no_bird":
            stem = Path(filename).stem

            # Original
            orig = temp_output_dir / "originals" / date_folder / filename
            orig.write_text("fake original")

            # Optimized
            opt = (
                temp_output_dir
                / "derivatives"
                / "optimized"
                / date_folder
                / f"{stem}.webp"
            )
            opt.write_text("fake optimized")

            # Preview thumb
            preview = (
                temp_output_dir
                / "derivatives"
                / "thumbs"
                / date_folder
                / f"{stem}_preview.webp"
            )
            preview.write_text("fake preview")


# ---------------------------------------------------------------------------
# Tests for hard_delete_images()
# ---------------------------------------------------------------------------


class TestHardDeleteImages:
    """Tests for the hard_delete_images() function."""

    def test_delete_specific_files(self, mock_db_connection, temp_output_dir):
        """Delete specific no_bird images with full file cleanup."""
        from utils.file_gc import hard_delete_images

        # Insert test images
        test_images = [
            ("20260131_120000_test1.jpg", "20260131_120000", "no_bird"),
            ("20260131_120100_test2.jpg", "20260131_120100", "no_bird"),
        ]
        for filename, timestamp, status in test_images:
            mock_db_connection.execute(
                "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
                (filename, timestamp, status),
            )
        mock_db_connection.commit()

        # Create files
        create_sample_files(temp_output_dir, test_images)

        mock_pm = MockPathManager(temp_output_dir)

        with (
            patch("utils.file_gc.get_config") as mock_config,
            patch("utils.file_gc.get_path_manager") as mock_get_pm,
        ):
            mock_config.return_value = {"OUTPUT_DIR": str(temp_output_dir)}
            mock_get_pm.return_value = mock_pm

            # Delete only the first image
            result = hard_delete_images(
                mock_db_connection, filenames=["20260131_120000_test1.jpg"]
            )

        assert result["purged"] is True
        assert result["rows_deleted"] == 1
        assert result["files_deleted"] == 3  # original + optimized + preview

        # Verify file is gone
        assert not (
            temp_output_dir / "originals" / "2026-01-31" / "20260131_120000_test1.jpg"
        ).exists()

        # Verify other no_bird image still exists
        assert (
            temp_output_dir / "originals" / "2026-01-31" / "20260131_120100_test2.jpg"
        ).exists()

        # Verify DB row is deleted
        cur = mock_db_connection.execute(
            "SELECT * FROM images WHERE filename = '20260131_120000_test1.jpg'"
        )
        assert cur.fetchone() is None

    def test_delete_all_no_bird_images(self, mock_db_connection, temp_output_dir):
        """Delete all no_bird images at once."""
        from utils.file_gc import hard_delete_images

        # Insert test images
        test_images = [
            ("20260131_120000_test1.jpg", "20260131_120000", "no_bird"),
            ("20260131_120100_test2.jpg", "20260131_120100", "no_bird"),
            ("20260131_120200_test3.jpg", "20260131_120200", "untagged"),
        ]
        for filename, timestamp, status in test_images:
            mock_db_connection.execute(
                "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
                (filename, timestamp, status),
            )
        mock_db_connection.commit()

        # Create files for no_bird images only
        create_sample_files(temp_output_dir, test_images)

        mock_pm = MockPathManager(temp_output_dir)

        with (
            patch("utils.file_gc.get_config") as mock_config,
            patch("utils.file_gc.get_path_manager") as mock_get_pm,
        ):
            mock_config.return_value = {"OUTPUT_DIR": str(temp_output_dir)}
            mock_get_pm.return_value = mock_pm

            result = hard_delete_images(mock_db_connection, delete_all=True)

        assert result["purged"] is True
        assert result["rows_deleted"] == 2  # Only no_bird images, not untagged
        assert result["files_deleted"] == 6  # 2 images * 3 files each

    def test_safeguard_only_no_bird_status(self, mock_db_connection, temp_output_dir):
        """Safeguard: Only images with review_status='no_bird' can be deleted."""
        from utils.file_gc import hard_delete_images

        # Insert only untagged image
        mock_db_connection.execute(
            "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
            ("20260131_120200_test3.jpg", "20260131_120200", "untagged"),
        )
        mock_db_connection.commit()

        mock_pm = MockPathManager(temp_output_dir)

        with (
            patch("utils.file_gc.get_config") as mock_config,
            patch("utils.file_gc.get_path_manager") as mock_get_pm,
        ):
            mock_config.return_value = {"OUTPUT_DIR": str(temp_output_dir)}
            mock_get_pm.return_value = mock_pm

            # Try to delete an 'untagged' image - should be ignored
            result = hard_delete_images(
                mock_db_connection, filenames=["20260131_120200_test3.jpg"]
            )

        # Nothing should be deleted
        assert result["rows_deleted"] == 0

        # Verify untagged image is still in DB
        cur = mock_db_connection.execute(
            "SELECT * FROM images WHERE filename = '20260131_120200_test3.jpg'"
        )
        assert cur.fetchone() is not None

    def test_dry_run_mode(self, mock_db_connection, temp_output_dir):
        """Dry-run mode returns what would be deleted without deleting."""
        from utils.file_gc import hard_delete_images

        # Insert test images
        test_images = [
            ("20260131_120000_test1.jpg", "20260131_120000", "no_bird"),
            ("20260131_120100_test2.jpg", "20260131_120100", "no_bird"),
        ]
        for filename, timestamp, status in test_images:
            mock_db_connection.execute(
                "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
                (filename, timestamp, status),
            )
        mock_db_connection.commit()

        # Create files
        create_sample_files(temp_output_dir, test_images)

        result = hard_delete_images(mock_db_connection, delete_all=True, dry_run=True)

        assert result["purged"] is False
        assert result["would_purge"] == 2
        assert "20260131_120000_test1.jpg" in result["filenames"]

        # Files should still exist
        assert (
            temp_output_dir / "originals" / "2026-01-31" / "20260131_120000_test1.jpg"
        ).exists()

        # DB should still have all records
        cur = mock_db_connection.execute(
            "SELECT COUNT(*) as cnt FROM images WHERE review_status = 'no_bird'"
        )
        assert cur.fetchone()["cnt"] == 2

    def test_missing_files_handled_gracefully(
        self, mock_db_connection, temp_output_dir
    ):
        """Gracefully handle case where files are already missing."""
        from utils.file_gc import hard_delete_images

        # Insert image in DB but don't create files
        mock_db_connection.execute(
            "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
            ("20260131_130000_missing.jpg", "20260131_130000", "no_bird"),
        )
        mock_db_connection.commit()

        mock_pm = MockPathManager(temp_output_dir)

        with (
            patch("utils.file_gc.get_config") as mock_config,
            patch("utils.file_gc.get_path_manager") as mock_get_pm,
        ):
            mock_config.return_value = {"OUTPUT_DIR": str(temp_output_dir)}
            mock_get_pm.return_value = mock_pm

            result = hard_delete_images(
                mock_db_connection, filenames=["20260131_130000_missing.jpg"]
            )

        assert result["purged"] is True
        assert result["rows_deleted"] == 1
        assert result["files_missing"] == 3  # All 3 expected files are missing
        assert result["files_failed"] == 0

    def test_no_action_without_args(self, mock_db_connection):
        """No deletion happens if neither filenames nor delete_all is provided."""
        from utils.file_gc import hard_delete_images

        result = hard_delete_images(mock_db_connection)

        assert result["purged"] is False
        assert result["rows_deleted"] == 0

    def test_detection_thumbs_cleaned_up(self, mock_db_connection, temp_output_dir):
        """Detection thumbnails tied to no_bird images are also cleaned up."""
        from utils.file_gc import hard_delete_images

        # Insert image with associated detection
        mock_db_connection.execute(
            "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
            ("20260131_140000_withdet.jpg", "20260131_140000", "no_bird"),
        )
        mock_db_connection.execute(
            "INSERT INTO detections (image_filename, thumbnail_path, status) VALUES (?, ?, ?)",
            (
                "20260131_140000_withdet.jpg",
                "20260131_140000_withdet_crop_1.webp",
                "rejected",
            ),
        )
        mock_db_connection.commit()

        # Create all files
        date_folder = "2026-01-31"
        stem = "20260131_140000_withdet"

        (temp_output_dir / "originals" / date_folder / f"{stem}.jpg").write_text("orig")
        (
            temp_output_dir / "derivatives" / "optimized" / date_folder / f"{stem}.webp"
        ).write_text("opt")
        (
            temp_output_dir
            / "derivatives"
            / "thumbs"
            / date_folder
            / f"{stem}_preview.webp"
        ).write_text("preview")
        (
            temp_output_dir
            / "derivatives"
            / "thumbs"
            / date_folder
            / f"{stem}_crop_1.webp"
        ).write_text("det_thumb")

        mock_pm = MockPathManager(temp_output_dir)

        with (
            patch("utils.file_gc.get_config") as mock_config,
            patch("utils.file_gc.get_path_manager") as mock_get_pm,
        ):
            mock_config.return_value = {"OUTPUT_DIR": str(temp_output_dir)}
            mock_get_pm.return_value = mock_pm

            result = hard_delete_images(
                mock_db_connection, filenames=["20260131_140000_withdet.jpg"]
            )

        assert result["purged"] is True
        assert (
            result["files_deleted"] == 4
        )  # original + optimized + preview + detection thumb

        # All files should be gone
        assert not (
            temp_output_dir
            / "derivatives"
            / "thumbs"
            / date_folder
            / f"{stem}_crop_1.webp"
        ).exists()


# ---------------------------------------------------------------------------
# Tests for _safe_delete()
# ---------------------------------------------------------------------------


class TestSafeDelete:
    """Tests for the _safe_delete() helper function."""

    def test_delete_existing_file(self, tmp_path):
        """Successfully delete an existing file."""
        from utils.file_gc import _safe_delete

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        result = _safe_delete(test_file, tmp_path)

        assert result == "deleted"
        assert not test_file.exists()

    def test_missing_file_returns_missing(self, tmp_path):
        """Non-existent file returns 'missing'."""
        from utils.file_gc import _safe_delete

        result = _safe_delete(tmp_path / "nonexistent.txt", tmp_path)

        assert result == "missing"

    def test_refuses_outside_output_dir(self, tmp_path):
        """Refuses to delete files outside output_dir."""
        from utils.file_gc import _safe_delete

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        outside_file = tmp_path / "outside.txt"
        outside_file.write_text("should not be deleted")

        result = _safe_delete(outside_file, output_dir)

        assert result == "error"
        assert outside_file.exists()  # File should still exist

    def test_none_path_returns_skipped(self, tmp_path):
        """None path returns 'skipped'."""
        from utils.file_gc import _safe_delete

        result = _safe_delete(None, tmp_path)

        assert result == "skipped"
