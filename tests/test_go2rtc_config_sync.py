"""
Tests for utils/go2rtc_config.py – YAML config sync for go2rtc.

Covers:
- Create / update / read streams.camera
- Preserve unrelated config keys
- Atomic write + .bak backup
- Graceful handling of missing files
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_config_dir(tmp_path):
    """Returns a temporary directory for go2rtc config files."""
    return tmp_path


@pytest.fixture
def sample_config(tmp_config_dir):
    """Creates a sample go2rtc.yaml and returns its path."""
    config_path = tmp_config_dir / "go2rtc.yaml"
    config_data = {
        "streams": {
            "test": ["ffmpeg:http://example.com/video.mp4#video=h264"],
            "camera": ["rtsp://admin:instar@192.168.178.92:554/11?transport=tcp"],
        },
        "api": {"listen": ":1984", "origin": "*"},
        "rtsp": {"listen": ":8554"},
        "webrtc": {
            "listen": ":8555",
            "candidates": ["192.168.178.129:8555", "stun:8555"],
        },
        "log": {"level": "info", "format": "text"},
    }
    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config_data, fh, default_flow_style=False, sort_keys=False)
    return str(config_path)


# ---------------------------------------------------------------------------
# read_camera_stream_source
# ---------------------------------------------------------------------------


class TestReadCameraStreamSource:
    """Tests for reading stream sources from go2rtc config."""

    def test_reads_existing_stream(self, sample_config):
        from utils.go2rtc_config import read_camera_stream_source

        url = read_camera_stream_source(sample_config, "camera")
        assert url == "rtsp://admin:instar@192.168.178.92:554/11?transport=tcp"

    def test_reads_test_stream(self, sample_config):
        from utils.go2rtc_config import read_camera_stream_source

        url = read_camera_stream_source(sample_config, "test")
        assert url == "ffmpeg:http://example.com/video.mp4#video=h264"

    def test_returns_empty_for_missing_stream(self, sample_config):
        from utils.go2rtc_config import read_camera_stream_source

        url = read_camera_stream_source(sample_config, "nonexistent")
        assert url == ""

    def test_returns_empty_for_missing_file(self):
        from utils.go2rtc_config import read_camera_stream_source

        url = read_camera_stream_source("/nonexistent/go2rtc.yaml")
        assert url == ""

    def test_returns_empty_for_empty_file(self, tmp_config_dir):
        from utils.go2rtc_config import read_camera_stream_source

        empty_path = tmp_config_dir / "empty.yaml"
        empty_path.write_text("")
        url = read_camera_stream_source(str(empty_path))
        assert url == ""


# ---------------------------------------------------------------------------
# set_camera_stream_source
# ---------------------------------------------------------------------------


class TestSetCameraStreamSource:
    """Tests for updating stream sources in go2rtc config."""

    def test_updates_camera_stream(self, sample_config):
        from utils.go2rtc_config import (
            read_camera_stream_source,
            set_camera_stream_source,
        )

        new_url = "rtsp://new-cam:554/stream1"
        ok = set_camera_stream_source(sample_config, new_url, "camera")
        assert ok is True

        # Verify
        url = read_camera_stream_source(sample_config, "camera")
        assert url == new_url

    def test_preserves_other_streams(self, sample_config):
        from utils.go2rtc_config import (
            read_camera_stream_source,
            set_camera_stream_source,
        )

        set_camera_stream_source(sample_config, "rtsp://new:554/stream", "camera")

        # 'test' stream must still be there
        test_url = read_camera_stream_source(sample_config, "test")
        assert test_url == "ffmpeg:http://example.com/video.mp4#video=h264"

    def test_preserves_non_stream_config(self, sample_config):
        from utils.go2rtc_config import set_camera_stream_source

        set_camera_stream_source(sample_config, "rtsp://new:554/stream", "camera")

        with open(sample_config, encoding="utf-8") as fh:
            config = yaml.safe_load(fh)

        # API, RTSP, WebRTC, Log sections must be intact
        assert config["api"]["listen"] == ":1984"
        assert config["rtsp"]["listen"] == ":8554"
        assert config["webrtc"]["listen"] == ":8555"
        assert config["log"]["level"] == "info"

    def test_creates_backup(self, sample_config):
        from utils.go2rtc_config import set_camera_stream_source

        set_camera_stream_source(sample_config, "rtsp://new:554/stream", "camera")

        bak_path = Path(sample_config + ".bak")
        assert bak_path.exists()

        # Backup should contain the old URL
        with open(bak_path, encoding="utf-8") as fh:
            backup = yaml.safe_load(fh)
        assert backup["streams"]["camera"] == [
            "rtsp://admin:instar@192.168.178.92:554/11?transport=tcp"
        ]

    def test_returns_false_for_missing_file(self):
        from utils.go2rtc_config import set_camera_stream_source

        ok = set_camera_stream_source(
            "/nonexistent/go2rtc.yaml", "rtsp://cam:554/stream"
        )
        assert ok is False

    def test_creates_new_stream(self, sample_config):
        from utils.go2rtc_config import (
            read_camera_stream_source,
            set_camera_stream_source,
        )

        ok = set_camera_stream_source(
            sample_config, "rtsp://front:554/stream", "front_yard"
        )
        assert ok is True

        url = read_camera_stream_source(sample_config, "front_yard")
        assert url == "rtsp://front:554/stream"


# ---------------------------------------------------------------------------
# sync_camera_stream_source
# ---------------------------------------------------------------------------


class TestSyncCameraStreamSource:
    """Tests for one-shot ensure+update helper."""

    def test_creates_missing_file_and_sets_stream(self, tmp_config_dir):
        from utils.go2rtc_config import (
            read_camera_stream_source,
            sync_camera_stream_source,
        )

        config_path = tmp_config_dir / "missing" / "go2rtc.yaml"
        camera_url = "rtsp://user:pass@192.168.1.10:554/stream"

        ok = sync_camera_stream_source(str(config_path), camera_url, "camera")
        assert ok is True
        assert config_path.exists()
        assert read_camera_stream_source(str(config_path), "camera") == camera_url

    def test_uses_template_when_provided(self, tmp_config_dir):
        from utils.go2rtc_config import sync_camera_stream_source

        template_path = tmp_config_dir / "go2rtc.template.yaml"
        template_path.write_text(
            yaml.safe_dump(
                {
                    "api": {"listen": ":1984", "origin": "*"},
                    "webrtc": {"listen": ":8555"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        config_path = tmp_config_dir / "new" / "go2rtc.yaml"
        ok = sync_camera_stream_source(
            str(config_path),
            "rtsp://admin:secret@192.168.1.50:554/live",
            "camera",
            template_path=str(template_path),
        )
        assert ok is True

        with open(config_path, encoding="utf-8") as fh:
            config = yaml.safe_load(fh)

        assert config["api"]["origin"] == "*"
        assert config["webrtc"]["listen"] == ":8555"
        assert config["streams"]["camera"] == [
            "rtsp://admin:secret@192.168.1.50:554/live"
        ]

    def test_returns_false_if_ensure_fails(self):
        from utils.go2rtc_config import sync_camera_stream_source

        with patch(
            "utils.go2rtc_config.ensure_go2rtc_config_exists", return_value=False
        ):
            assert (
                sync_camera_stream_source(
                    "/nonexistent/go2rtc.yaml",
                    "rtsp://cam:554/stream",
                    "camera",
                )
                is False
            )


# ---------------------------------------------------------------------------
# ensure_go2rtc_config_exists
# ---------------------------------------------------------------------------


class TestEnsureGo2rtcConfigExists:
    """Tests for config file creation/existence checks."""

    def test_returns_true_if_exists(self, sample_config):
        from utils.go2rtc_config import ensure_go2rtc_config_exists

        assert ensure_go2rtc_config_exists(sample_config) is True

    def test_creates_from_template(self, tmp_config_dir, sample_config):
        from utils.go2rtc_config import ensure_go2rtc_config_exists

        new_path = str(tmp_config_dir / "subdir" / "go2rtc.yaml")
        ok = ensure_go2rtc_config_exists(new_path, template_path=sample_config)
        assert ok is True
        assert Path(new_path).exists()

        with open(new_path, encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        assert "camera" in config["streams"]

    def test_creates_default_if_no_template(self, tmp_config_dir):
        from utils.go2rtc_config import ensure_go2rtc_config_exists

        new_path = str(tmp_config_dir / "new" / "go2rtc.yaml")
        ok = ensure_go2rtc_config_exists(new_path)
        assert ok is True
        assert Path(new_path).exists()

        with open(new_path, encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        assert "streams" in config
        assert "camera" in config["streams"]

    def test_idempotent(self, sample_config):
        from utils.go2rtc_config import ensure_go2rtc_config_exists

        # Should not modify existing config
        with open(sample_config) as fh:
            original = fh.read()

        ensure_go2rtc_config_exists(sample_config)

        with open(sample_config) as fh:
            after = fh.read()

        assert original == after


# ---------------------------------------------------------------------------
# Edge cases & credential masking
# ---------------------------------------------------------------------------


class TestCredentialMasking:
    """Tests for the _mask_credentials helper."""

    def test_masks_rtsp_password(self):
        from utils.go2rtc_config import _mask_credentials

        masked = _mask_credentials("rtsp://admin:secret@192.168.1.100:554/stream")
        assert "secret" not in masked
        assert "*****" in masked
        assert "admin" in masked

    def test_no_password_unchanged(self):
        from utils.go2rtc_config import _mask_credentials

        url = "rtsp://192.168.1.100:554/stream"
        assert _mask_credentials(url) == url

    def test_non_url_unchanged(self):
        from utils.go2rtc_config import _mask_credentials

        val = "webcam:0"
        assert _mask_credentials(val) == val


# ---------------------------------------------------------------------------
# reload_go2rtc_stream – runtime reload via REST API
# ---------------------------------------------------------------------------


class TestReloadGo2rtcStream:
    """Tests for the fixed go2rtc runtime reload API call."""

    def test_skips_empty_camera_url(self):
        from utils.go2rtc_config import reload_go2rtc_stream

        assert reload_go2rtc_stream(camera_url="") is False

    def test_constructs_correct_url_with_name_and_src_params(self):
        """Verifies the fixed API format: ?name=<stream>&src=<url>."""
        from utils.go2rtc_config import reload_go2rtc_stream

        captured_requests = []

        def mock_urlopen(req, **kwargs):
            captured_requests.append(req)
            # Simulate success response
            from unittest.mock import MagicMock

            resp = MagicMock()
            resp.status = 200
            resp.__enter__ = lambda s: s
            resp.__exit__ = lambda s, *a: None
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            result = reload_go2rtc_stream(
                api_base="http://127.0.0.1:1984",
                stream_name="camera",
                camera_url="rtsp://admin:pass@192.168.1.100:554/stream",
            )

        assert result is True
        assert len(captured_requests) == 1
        req = captured_requests[0]
        url = req.full_url

        # Must have name=camera AND src=<camera_url> as query params
        assert "name=camera" in url
        assert "src=" in url
        # Must NOT have a JSON body
        assert req.data is None
        # Must be PUT
        assert req.get_method() == "PUT"

    def test_url_encodes_special_characters_in_camera_url(self):
        """Camera URLs with special chars must be properly URL-encoded."""
        from utils.go2rtc_config import reload_go2rtc_stream

        captured_requests = []

        def mock_urlopen(req, **kwargs):
            captured_requests.append(req)
            from unittest.mock import MagicMock

            resp = MagicMock()
            resp.status = 200
            resp.__enter__ = lambda s: s
            resp.__exit__ = lambda s, *a: None
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            reload_go2rtc_stream(
                stream_name="camera",
                camera_url="rtsp://admin:p@ss@192.168.1.100:554/stream?transport=tcp",
            )

        url = captured_requests[0].full_url
        # The @ and ? in the camera URL should be encoded in query params
        assert "name=camera" in url
        # The raw password should not appear unencoded
        assert "p@ss@192" not in url

    def test_returns_false_on_http_400(self):
        """HTTP 400 (the bug we fixed) should return False."""
        import urllib.error

        from utils.go2rtc_config import reload_go2rtc_stream

        def mock_urlopen(req, **kwargs):
            raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, None)

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            result = reload_go2rtc_stream(
                stream_name="camera",
                camera_url="rtsp://192.168.1.100:554/stream",
            )

        assert result is False

    def test_returns_false_on_connection_error(self):
        """Connection refused (go2rtc not running) should return False."""
        from utils.go2rtc_config import reload_go2rtc_stream

        def mock_urlopen(req, **kwargs):
            raise ConnectionRefusedError("Connection refused")

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            result = reload_go2rtc_stream(
                stream_name="camera",
                camera_url="rtsp://192.168.1.100:554/stream",
            )

        assert result is False


# ---------------------------------------------------------------------------
# reload_go2rtc_stream_with_retry – startup retry wrapper
# ---------------------------------------------------------------------------


class TestReloadGo2rtcStreamWithRetry:
    """Tests for the bounded retry/backoff wrapper."""

    def test_skips_empty_camera_url(self):
        from utils.go2rtc_config import reload_go2rtc_stream_with_retry

        assert reload_go2rtc_stream_with_retry(camera_url="") is False

    def test_succeeds_on_first_attempt(self):
        from utils.go2rtc_config import reload_go2rtc_stream_with_retry

        with patch(
            "utils.go2rtc_config.reload_go2rtc_stream", return_value=True
        ) as mock_reload:
            result = reload_go2rtc_stream_with_retry(
                camera_url="rtsp://192.168.1.100:554/stream",
                max_attempts=3,
            )

        assert result is True
        assert mock_reload.call_count == 1

    def test_retries_on_failure_then_succeeds(self):
        from utils.go2rtc_config import reload_go2rtc_stream_with_retry

        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            return call_count >= 2  # Fail first, succeed second

        with patch("utils.go2rtc_config.reload_go2rtc_stream", side_effect=side_effect):
            with patch("time.sleep") as mock_sleep:
                result = reload_go2rtc_stream_with_retry(
                    camera_url="rtsp://192.168.1.100:554/stream",
                    max_attempts=3,
                    backoff_steps=(0.1, 0.2, 0.4),
                )

        assert result is True
        assert call_count == 2
        # Should have slept once (after first failure)
        assert mock_sleep.call_count == 1
        mock_sleep.assert_called_with(0.1)

    def test_exhausts_all_attempts(self):
        from utils.go2rtc_config import reload_go2rtc_stream_with_retry

        with patch(
            "utils.go2rtc_config.reload_go2rtc_stream", return_value=False
        ) as mock_reload:
            with patch("time.sleep"):
                result = reload_go2rtc_stream_with_retry(
                    camera_url="rtsp://192.168.1.100:554/stream",
                    max_attempts=3,
                )

        assert result is False
        assert mock_reload.call_count == 3

    def test_backoff_uses_last_step_for_overflow(self):
        """If more attempts than backoff steps, uses last step value."""
        from utils.go2rtc_config import reload_go2rtc_stream_with_retry

        sleep_calls = []

        def track_sleep(duration):
            sleep_calls.append(duration)

        with patch("utils.go2rtc_config.reload_go2rtc_stream", return_value=False):
            with patch("time.sleep", side_effect=track_sleep):
                reload_go2rtc_stream_with_retry(
                    camera_url="rtsp://192.168.1.100:554/stream",
                    max_attempts=5,
                    backoff_steps=(1.0, 2.0),  # Only 2 steps for 5 attempts
                )

        # Sleeps between attempts: after 1st, 2nd, 3rd, 4th (not after 5th)
        assert len(sleep_calls) == 4
        assert sleep_calls == [1.0, 2.0, 2.0, 2.0]  # Last step reused
