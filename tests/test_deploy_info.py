"""Tests for utils/deploy_info.py — build metadata reader and deploy type detection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from utils.deploy_info import (
    _UNKNOWN,
    _read_first,
    detect_deploy_type,
    read_build_metadata,
)

# ---------------------------------------------------------------------------
# Unit tests for _read_first
# ---------------------------------------------------------------------------


class TestReadFirst:
    """Low-level file-read helper."""

    def test_returns_unknown_when_no_paths_exist(self, tmp_path: Path) -> None:
        result = _read_first([tmp_path / "no_such_file.txt"])
        assert result == _UNKNOWN

    def test_reads_first_existing_file(self, tmp_path: Path) -> None:
        f1 = tmp_path / "first.txt"
        f2 = tmp_path / "second.txt"
        f1.write_text("preferred\n")
        f2.write_text("fallback\n")
        assert _read_first([f1, f2]) == "preferred"

    def test_falls_through_to_second_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.txt"
        f2 = tmp_path / "second.txt"
        f2.write_text("  fallback  \n")
        assert _read_first([missing, f2]) == "fallback"

    def test_skips_empty_files(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.txt"
        empty.write_text("   \n")
        good = tmp_path / "good.txt"
        good.write_text("value")
        assert _read_first([empty, good]) == "value"


# ---------------------------------------------------------------------------
# Unit tests for detect_deploy_type
# ---------------------------------------------------------------------------


class TestDetectDeployType:
    """Deploy type detection logic."""

    def test_docker_via_dockerenv(self, tmp_path: Path) -> None:
        with patch("utils.deploy_info.Path") as MockPath:
            # /.dockerenv exists → docker
            def path_side_effect(p: str) -> Path:
                if p == "/.dockerenv":
                    mock = type("PathMock", (), {"exists": lambda self: True})()
                    return mock  # type: ignore[return-value]
                return Path(p)

            MockPath.side_effect = path_side_effect
            assert detect_deploy_type() == "docker"

    def test_dev_when_no_markers(self) -> None:
        """On a dev machine neither /.dockerenv nor /opt/app exist → dev."""
        with (
            patch("utils.deploy_info.Path") as MockPath,
        ):

            def path_side_effect(p: str) -> Path:
                mock_obj = type(
                    "PathMock",
                    (),
                    {
                        "exists": lambda self: False,
                        "is_dir": lambda self: False,
                        "is_file": lambda self: False,
                        "read_text": lambda self, **kw: (_ for _ in ()).throw(
                            FileNotFoundError
                        ),
                    },
                )()
                return mock_obj  # type: ignore[return-value]

            MockPath.side_effect = path_side_effect
            assert detect_deploy_type() == "dev"


# ---------------------------------------------------------------------------
# Integration-style tests for read_build_metadata
# ---------------------------------------------------------------------------


class TestReadBuildMetadata:
    """Full metadata reader with file-system fixtures."""

    def test_docker_style_file_set(self, tmp_path: Path) -> None:
        """Docker paths preferred: /app/APP_VERSION etc."""
        (tmp_path / "APP_VERSION").write_text("1.2.3\n")
        (tmp_path / "BUILD_COMMIT").write_text("abcdef1234567890\n")
        (tmp_path / "BUILD_DATE").write_text("2026-03-11T10:00:00Z\n")

        docker_sources = {
            "app_version": [tmp_path / "APP_VERSION"],
            "git_commit": [tmp_path / "BUILD_COMMIT"],
            "build_date": [tmp_path / "BUILD_DATE"],
        }
        with (
            patch("utils.deploy_info._FILE_SOURCES", docker_sources),
            patch("utils.deploy_info.detect_deploy_type", return_value="docker"),
        ):
            meta = read_build_metadata()

        assert meta["app_version"] == "1.2.3"
        assert meta["git_commit"] == "abcdef1"  # trimmed to 7 chars
        assert meta["build_date"] == "2026-03-11T10:00:00Z"
        assert meta["deploy_type"] == "docker"

    def test_rpi_style_file_set(self, tmp_path: Path) -> None:
        """RPi compatibility paths: /opt/app/* files."""
        (tmp_path / "APP_VERSION").write_text("0.1.0\n")
        (tmp_path / "commit.txt").write_text("fedcba9876543210\n")
        (tmp_path / "build_date.txt").write_text("2026-03-10T08:00:00Z\n")

        rpi_sources = {
            "app_version": [tmp_path / "APP_VERSION"],
            "git_commit": [tmp_path / "commit.txt"],
            "build_date": [tmp_path / "build_date.txt"],
        }
        with (
            patch("utils.deploy_info._FILE_SOURCES", rpi_sources),
            patch("utils.deploy_info.detect_deploy_type", return_value="rpi"),
        ):
            meta = read_build_metadata()

        assert meta["app_version"] == "0.1.0"
        assert meta["git_commit"] == "fedcba9"
        assert meta["build_date"] == "2026-03-10T08:00:00Z"
        assert meta["deploy_type"] == "rpi"

    def test_no_files_returns_safe_defaults(self, tmp_path: Path) -> None:
        """No metadata files → Unknown values and deploy_type 'dev'."""
        empty_sources = {
            "app_version": [tmp_path / "nope"],
            "git_commit": [tmp_path / "nope2"],
            "build_date": [tmp_path / "nope3"],
        }
        with (
            patch("utils.deploy_info._FILE_SOURCES", empty_sources),
            patch("utils.deploy_info._try_local_app_version", return_value=_UNKNOWN),
            patch("utils.deploy_info.detect_deploy_type", return_value="dev"),
        ):
            meta = read_build_metadata()

        assert meta["app_version"] == _UNKNOWN
        assert meta["git_commit"] == _UNKNOWN
        assert meta["build_date"] == _UNKNOWN
        assert meta["deploy_type"] == "dev"

    def test_local_app_version_fallback(self, tmp_path: Path) -> None:
        """When file sources miss app_version, fall back to local APP_VERSION."""
        empty_sources = {
            "app_version": [tmp_path / "nope"],
            "git_commit": [tmp_path / "nope2"],
            "build_date": [tmp_path / "nope3"],
        }
        with (
            patch("utils.deploy_info._FILE_SOURCES", empty_sources),
            patch(
                "utils.deploy_info._try_local_app_version", return_value="0.1.0-local"
            ),
            patch("utils.deploy_info.detect_deploy_type", return_value="dev"),
        ):
            meta = read_build_metadata()

        assert meta["app_version"] == "0.1.0-local"
