# tests/test_file_gc.py
"""
Unit tests for the file_gc module (file garbage collection).
Tests hard_delete_detections().
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
# Detection deletion contracts
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests for hard_delete_detections()
# ---------------------------------------------------------------------------


class TestHardDeleteDetections:
    """Regression tests for thumbnail reference counting."""

    def test_shared_thumbnail_path_preserved_for_surviving_detection(
        self, mock_db_connection, temp_output_dir
    ):
        """When two detections share a thumbnail_path and only the rejected
        one is hard-deleted, the surviving detection's thumb file must
        stay on disk."""
        from utils.file_gc import hard_delete_detections

        mock_db_connection.execute(
            "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
            ("20260131_150000_shared.jpg", "20260131_150000", "untagged"),
        )
        # Surviving (active) detection — references crop_1.webp
        mock_db_connection.execute(
            "INSERT INTO detections (detection_id, image_filename, thumbnail_path, status) "
            "VALUES (?, ?, ?, ?)",
            (
                1,
                "20260131_150000_shared.jpg",
                "20260131_150000_shared_crop_1.webp",
                "active",
            ),
        )
        # Rejected detection — same thumbnail_path (legacy / re-import case)
        mock_db_connection.execute(
            "INSERT INTO detections (detection_id, image_filename, thumbnail_path, status) "
            "VALUES (?, ?, ?, ?)",
            (
                2,
                "20260131_150000_shared.jpg",
                "20260131_150000_shared_crop_1.webp",
                "rejected",
            ),
        )
        mock_db_connection.commit()

        date_folder = "2026-01-31"
        stem = "20260131_150000_shared"
        thumb_file = (
            temp_output_dir
            / "derivatives"
            / "thumbs"
            / date_folder
            / f"{stem}_crop_1.webp"
        )
        thumb_file.write_text("shared thumb")
        (temp_output_dir / "originals" / date_folder / f"{stem}.jpg").write_text("orig")
        (
            temp_output_dir / "derivatives" / "optimized" / date_folder / f"{stem}.webp"
        ).write_text("opt")

        mock_pm = MockPathManager(temp_output_dir)

        with (
            patch("utils.file_gc.get_config") as mock_config,
            patch("utils.file_gc.get_path_manager") as mock_get_pm,
        ):
            mock_config.return_value = {"OUTPUT_DIR": str(temp_output_dir)}
            mock_get_pm.return_value = mock_pm

            result = hard_delete_detections(mock_db_connection, detection_ids=[2])

        assert result["purged"] is True
        # Thumb must survive — detection 1 still references it.
        assert thumb_file.exists()
        # Original & optimized also survive (detection 1 still references the image).
        assert (temp_output_dir / "originals" / date_folder / f"{stem}.jpg").exists()

    def test_unshared_thumbnail_deleted(self, mock_db_connection, temp_output_dir):
        """When the rejected detection is the sole owner of its thumb file,
        the file is deleted as before."""
        from utils.file_gc import hard_delete_detections

        mock_db_connection.execute(
            "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
            ("20260131_160000_solo.jpg", "20260131_160000", "untagged"),
        )
        mock_db_connection.execute(
            "INSERT INTO detections (detection_id, image_filename, thumbnail_path, status) "
            "VALUES (?, ?, ?, ?)",
            (
                10,
                "20260131_160000_solo.jpg",
                "20260131_160000_solo_crop_1.webp",
                "rejected",
            ),
        )
        mock_db_connection.commit()

        date_folder = "2026-01-31"
        stem = "20260131_160000_solo"
        thumb_file = (
            temp_output_dir
            / "derivatives"
            / "thumbs"
            / date_folder
            / f"{stem}_crop_1.webp"
        )
        thumb_file.write_text("solo thumb")
        (temp_output_dir / "originals" / date_folder / f"{stem}.jpg").write_text("orig")
        (
            temp_output_dir / "derivatives" / "optimized" / date_folder / f"{stem}.webp"
        ).write_text("opt")

        mock_pm = MockPathManager(temp_output_dir)

        with (
            patch("utils.file_gc.get_config") as mock_config,
            patch("utils.file_gc.get_path_manager") as mock_get_pm,
        ):
            mock_config.return_value = {"OUTPUT_DIR": str(temp_output_dir)}
            mock_get_pm.return_value = mock_pm

            result = hard_delete_detections(mock_db_connection, detection_ids=[10])

        assert result["purged"] is True
        assert not thumb_file.exists()


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

    def test_refuses_symlink_escaping_output_dir(self, tmp_path):
        """A symlink inside OUTPUT_DIR pointing outside must not be followed
        into deleting the external target."""
        from utils.file_gc import _safe_delete

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        external = tmp_path / "external_secret.txt"
        external.write_text("must survive")

        link = output_dir / "link.txt"
        try:
            link.symlink_to(external)
        except (OSError, NotImplementedError):
            import pytest

            pytest.skip("symlinks not supported on this platform")

        result = _safe_delete(link, output_dir)

        # The external target must still exist regardless of how the link
        # itself is treated — the escape must not delete outside OUTPUT_DIR.
        assert external.exists()
        assert result in ("error", "deleted", "missing")
        if result == "deleted":
            # If anything was removed it must have been the link, not the target.
            assert not link.is_symlink() or not link.exists()

    def test_refuses_sibling_dir_with_shared_prefix(self, tmp_path):
        """OUTPUT_DIR='/x/out' must not authorise deletion in '/x/out_evil'.

        A naive string-prefix check would pass the sibling; the guard must
        treat path components, not raw string prefixes.
        """
        from utils.file_gc import _safe_delete

        output_dir = tmp_path / "out"
        output_dir.mkdir()
        sibling = tmp_path / "out_evil"
        sibling.mkdir()
        victim = sibling / "victim.txt"
        victim.write_text("must survive")

        result = _safe_delete(victim, output_dir)

        assert result == "error"
        assert victim.exists()

    def test_unlink_failure_returns_error(self, tmp_path, monkeypatch):
        """A real unlink failure (e.g. permission) returns 'error', not a crash."""
        from pathlib import Path

        from utils.file_gc import _safe_delete

        test_file = tmp_path / "locked.txt"
        test_file.write_text("content")

        def boom(self):
            raise PermissionError("locked")

        monkeypatch.setattr(Path, "unlink", boom)
        result = _safe_delete(test_file, tmp_path)

        assert result == "error"

    def test_concurrent_disappearance_returns_missing(self, tmp_path, monkeypatch):
        """If the file vanishes between resolve and unlink, report 'missing'.

        This pins the race-safe behaviour the syscall-lean path relies on:
        deletion is attempted directly and a FileNotFoundError is mapped to
        'missing' instead of pre-checking with exists().
        """
        from pathlib import Path

        from utils.file_gc import _safe_delete

        test_file = tmp_path / "gone.txt"
        test_file.write_text("content")

        def vanish(self):
            raise FileNotFoundError("already gone")

        monkeypatch.setattr(Path, "unlink", vanish)
        result = _safe_delete(test_file, tmp_path)

        assert result == "missing"
