"""
Tests for the stream source resolution algorithm (§5) and migration logic (§8).

Covers:
- mode=auto with relay up/down
- mode=relay and mode=direct (forced)
- Validation errors (missing CAMERA_URL in direct mode)
- Migration from legacy VIDEO_SOURCE to CAMERA_URL
- CAMERA_URL null/None/"null" coercion
"""

from unittest.mock import patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_config(**overrides):
    """Returns a minimal config dict suitable for resolve_effective_sources."""
    cfg = {
        "CAMERA_URL": "rtsp://admin:pass@192.168.1.100:554/stream",
        "STREAM_SOURCE_MODE": "auto",
        "GO2RTC_STREAM_NAME": "camera",
        "GO2RTC_API_BASE": "http://127.0.0.1:1984",
        "GO2RTC_CONFIG_PATH": "./go2rtc.yaml",
        "VIDEO_SOURCE": "0",
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# resolve_effective_sources
# ---------------------------------------------------------------------------


class TestResolveEffectiveSources:
    """Tests for the central source resolver."""

    def test_auto_mode_go2rtc_healthy(self):
        """auto + go2rtc healthy + stream ready → relay."""
        from config import resolve_effective_sources

        cfg = _base_config()
        with (
            patch("config.probe_go2rtc", return_value=True),
            patch("config.verify_go2rtc_stream_ready", return_value=True),
        ):
            result = resolve_effective_sources(cfg)

        assert result["effective_mode"] == "relay"
        assert result["video_source"] == "rtsp://127.0.0.1:8554/camera"
        assert "configured" in result["reason"] or "relay" in result["reason"]

    def test_auto_mode_go2rtc_down(self):
        """auto + go2rtc unreachable → direct."""
        from config import resolve_effective_sources

        cfg = _base_config()
        with patch("config.probe_go2rtc", return_value=False):
            result = resolve_effective_sources(cfg)

        assert result["effective_mode"] == "direct"
        assert result["video_source"] == cfg["CAMERA_URL"]
        assert "unavailable" in result["reason"]

    def test_auto_mode_no_camera_url(self):
        """auto + empty CAMERA_URL → direct with explanatory reason."""
        from config import resolve_effective_sources

        cfg = _base_config(CAMERA_URL="")
        with (
            patch("config.probe_go2rtc", return_value=True),
            patch("config.verify_go2rtc_stream_ready", return_value=True),
        ):
            result = resolve_effective_sources(cfg)

        assert result["effective_mode"] == "direct"
        assert result["video_source"] == ""
        assert "empty" in result["reason"]

    def test_auto_mode_go2rtc_healthy_stream_not_ready(self):
        """auto + go2rtc healthy but stream not ready → direct."""
        from config import resolve_effective_sources

        cfg = _base_config()
        with (
            patch("config.probe_go2rtc", return_value=True),
            patch("config.verify_go2rtc_stream_ready", return_value=False),
        ):
            result = resolve_effective_sources(cfg)

        assert result["effective_mode"] == "direct"
        assert result["video_source"] == cfg["CAMERA_URL"]
        assert "not configured" in result["reason"]

    def test_relay_mode_forced(self):
        """relay mode always uses relay URL, regardless of probe."""
        from config import resolve_effective_sources

        cfg = _base_config(STREAM_SOURCE_MODE="relay")
        # probe is never called in relay mode
        result = resolve_effective_sources(cfg)

        assert result["effective_mode"] == "relay"
        assert result["video_source"] == "rtsp://127.0.0.1:8554/camera"

    def test_direct_mode_forced(self):
        """direct mode always uses CAMERA_URL."""
        from config import resolve_effective_sources

        cfg = _base_config(STREAM_SOURCE_MODE="direct")
        result = resolve_effective_sources(cfg)

        assert result["effective_mode"] == "direct"
        assert result["video_source"] == cfg["CAMERA_URL"]

    def test_direct_mode_empty_camera_url(self):
        """direct + empty CAMERA_URL → still works, returns empty source."""
        from config import resolve_effective_sources

        cfg = _base_config(STREAM_SOURCE_MODE="direct", CAMERA_URL="")
        result = resolve_effective_sources(cfg)

        assert result["effective_mode"] == "direct"
        assert result["video_source"] == ""

    def test_custom_stream_name(self):
        """Custom GO2RTC_STREAM_NAME changes relay URL."""
        from config import resolve_effective_sources

        cfg = _base_config(
            STREAM_SOURCE_MODE="relay",
            GO2RTC_STREAM_NAME="front_yard",
        )
        result = resolve_effective_sources(cfg)

        assert result["video_source"] == "rtsp://127.0.0.1:8554/front_yard"

    def test_relay_url_uses_api_base_hostname(self):
        """Relay URL derives hostname from GO2RTC_API_BASE (bridge-network)."""
        from config import resolve_effective_sources

        cfg = _base_config(
            STREAM_SOURCE_MODE="relay",
            GO2RTC_API_BASE="http://go2rtc:1984",
        )
        result = resolve_effective_sources(cfg)

        assert result["video_source"] == "rtsp://go2rtc:8554/camera"
        assert result["effective_mode"] == "relay"


# ---------------------------------------------------------------------------
# probe_go2rtc
# ---------------------------------------------------------------------------


class TestProbeGo2rtc:
    """Tests for the go2rtc health probe."""

    def test_returns_false_when_unreachable(self):
        """probe_go2rtc returns False when endpoint is unreachable."""
        from config import probe_go2rtc

        # Use a port that's almost certainly not listening
        result = probe_go2rtc("http://127.0.0.1:19999", timeout_sec=0.1)
        assert result is False

    def test_returns_false_on_invalid_url(self):
        """probe_go2rtc handles malformed URLs gracefully."""
        from config import probe_go2rtc

        result = probe_go2rtc("not-a-url", timeout_sec=0.1)
        assert result is False


# ---------------------------------------------------------------------------
# verify_go2rtc_stream_ready
# ---------------------------------------------------------------------------


class TestVerifyGo2rtcStreamReady:
    """Tests for the stream-level readiness check."""

    def _mock_urlopen(self, response_data, status=200):
        """Creates a mock urlopen that returns the given JSON data."""
        import json
        from unittest.mock import MagicMock

        def mock_urlopen(req, **kwargs):
            resp = MagicMock()
            resp.status = status
            resp.read.return_value = json.dumps(response_data).encode()
            resp.__enter__ = lambda s: s
            resp.__exit__ = lambda s, *a: None
            return resp

        return mock_urlopen

    def test_returns_true_when_stream_has_producers(self):
        """Stream with active producers → ready."""
        from config import verify_go2rtc_stream_ready

        data = {
            "camera": {
                "producers": [{"url": "rtsp://192.168.1.100:554/stream"}],
                "consumers": [],
            }
        }
        with patch("urllib.request.urlopen", side_effect=self._mock_urlopen(data)):
            assert verify_go2rtc_stream_ready(stream_name="camera") is True

    def test_returns_false_when_stream_missing(self):
        """Stream not found in go2rtc → not ready."""
        from config import verify_go2rtc_stream_ready

        data = {}
        with patch("urllib.request.urlopen", side_effect=self._mock_urlopen(data)):
            assert verify_go2rtc_stream_ready(stream_name="camera") is False

    def test_returns_true_when_no_producers(self):
        """Stream exists but has empty producers list → still ready (on-demand OK)."""
        from config import verify_go2rtc_stream_ready

        data = {
            "camera": {
                "producers": [],
                "consumers": [],
            }
        }
        with patch("urllib.request.urlopen", side_effect=self._mock_urlopen(data)):
            assert verify_go2rtc_stream_ready(stream_name="camera") is True

    def test_returns_true_when_producers_is_none(self):
        """Stream exists but producers is null/None → still ready (lazy/on-demand)."""
        from config import verify_go2rtc_stream_ready

        data = {
            "camera": {
                "producers": None,
                "consumers": [],
            }
        }
        with patch("urllib.request.urlopen", side_effect=self._mock_urlopen(data)):
            assert verify_go2rtc_stream_ready(stream_name="camera") is True

    def test_returns_false_on_connection_error(self):
        """API unreachable → not ready."""
        from config import verify_go2rtc_stream_ready

        def mock_urlopen(req, **kwargs):
            raise ConnectionRefusedError("Connection refused")

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            assert verify_go2rtc_stream_ready(stream_name="camera") is False

    def test_queries_all_streams_endpoint(self):
        """Verify the request URL is /api/streams (no ?src= filter)."""
        from config import verify_go2rtc_stream_ready

        captured = []

        def mock_urlopen(req, **kwargs):
            captured.append(req.full_url)
            raise ConnectionRefusedError("test")

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            verify_go2rtc_stream_ready(stream_name="front_yard")

        assert len(captured) == 1
        assert captured[0].endswith("/api/streams")
        assert "src=" not in captured[0]


# ---------------------------------------------------------------------------
# Migration logic (§8)
# ---------------------------------------------------------------------------


class TestMigrateCameraUrl:
    """Tests for _migrate_camera_url."""

    def test_no_migration_when_camera_url_set(self):
        """If CAMERA_URL is already set, migration is skipped."""
        from config import _migrate_camera_url

        cfg = {"CAMERA_URL": "rtsp://existing", "VIDEO_SOURCE": "rtsp://old"}
        _migrate_camera_url(cfg)
        assert cfg["CAMERA_URL"] == "rtsp://existing"

    def test_no_migration_for_default_webcam(self):
        """Default webcam VIDEO_SOURCE=0 should not trigger migration."""
        from config import _migrate_camera_url

        cfg = {"CAMERA_URL": "", "VIDEO_SOURCE": "0"}
        _migrate_camera_url(cfg)
        assert cfg["CAMERA_URL"] == ""

    def test_no_migration_for_int_webcam(self):
        """Integer webcam VIDEO_SOURCE=0 should not trigger migration."""
        from config import _migrate_camera_url

        cfg = {"CAMERA_URL": "", "VIDEO_SOURCE": 0}
        _migrate_camera_url(cfg)
        assert cfg["CAMERA_URL"] == ""

    def test_direct_camera_url_migrated(self):
        """Direct camera URL in VIDEO_SOURCE is copied to CAMERA_URL."""
        from config import _migrate_camera_url

        cfg = {
            "CAMERA_URL": "",
            "VIDEO_SOURCE": "rtsp://admin:pass@192.168.1.100:554/stream",
        }
        _migrate_camera_url(cfg)
        assert cfg["CAMERA_URL"] == "rtsp://admin:pass@192.168.1.100:554/stream"

    def test_webcam_index_gt0_migrated(self):
        """Webcam index > 0 is migrated as string."""
        from config import _migrate_camera_url

        cfg = {"CAMERA_URL": "", "VIDEO_SOURCE": 2}
        _migrate_camera_url(cfg)
        assert cfg["CAMERA_URL"] == "2"

    def test_relay_url_reads_go2rtc_config(self):
        """Relay URL triggers go2rtc config read for the real camera URL."""
        from config import _migrate_camera_url

        cfg = {
            "CAMERA_URL": "",
            "VIDEO_SOURCE": "rtsp://127.0.0.1:8554/camera",
            "GO2RTC_CONFIG_PATH": "./go2rtc.yaml",
            "GO2RTC_STREAM_NAME": "camera",
        }
        with patch(
            "utils.go2rtc_config.read_camera_stream_source",
            return_value="rtsp://admin:pass@192.168.1.100:554/stream",
        ):
            _migrate_camera_url(cfg)

        assert cfg["CAMERA_URL"] == "rtsp://admin:pass@192.168.1.100:554/stream"

    def test_relay_url_go2rtc_config_missing(self):
        """Relay URL with no go2rtc config leaves CAMERA_URL empty."""
        from config import _migrate_camera_url

        cfg = {
            "CAMERA_URL": "",
            "VIDEO_SOURCE": "rtsp://127.0.0.1:8554/camera",
            "GO2RTC_CONFIG_PATH": "/nonexistent/go2rtc.yaml",
            "GO2RTC_STREAM_NAME": "camera",
        }
        _migrate_camera_url(cfg)
        # CAMERA_URL stays empty because go2rtc config can't be read
        assert cfg["CAMERA_URL"] == ""


# ---------------------------------------------------------------------------
# Validation (_validate_value)
# ---------------------------------------------------------------------------


class TestValidateNewKeys:
    """Tests for _validate_value with new keys."""

    def test_camera_url_valid_string(self):
        from config import _validate_value

        ok, val = _validate_value("CAMERA_URL", "rtsp://cam:554/stream")
        assert ok is True
        assert val == "rtsp://cam:554/stream"

    def test_camera_url_empty_string(self):
        from config import _validate_value

        ok, val = _validate_value("CAMERA_URL", "")
        assert ok is True
        assert val == ""

    def test_camera_url_none_coerced(self):
        from config import _validate_value

        ok, val = _validate_value("CAMERA_URL", None)
        assert ok is True
        assert val == ""

    def test_camera_url_null_string_coerced(self):
        from config import _validate_value

        ok, val = _validate_value("CAMERA_URL", "null")
        assert ok is True
        assert val == ""

    def test_stream_source_mode_auto(self):
        from config import _validate_value

        ok, val = _validate_value("STREAM_SOURCE_MODE", "auto")
        assert ok is True
        assert val == "auto"

    def test_stream_source_mode_relay(self):
        from config import _validate_value

        ok, val = _validate_value("STREAM_SOURCE_MODE", "relay")
        assert ok is True
        assert val == "relay"

    def test_stream_source_mode_direct(self):
        from config import _validate_value

        ok, val = _validate_value("STREAM_SOURCE_MODE", "direct")
        assert ok is True
        assert val == "direct"

    def test_stream_source_mode_invalid(self):
        from config import _validate_value

        ok, val = _validate_value("STREAM_SOURCE_MODE", "turbo")
        assert ok is False

    def test_stream_source_mode_case_insensitive(self):
        from config import _validate_value

        ok, val = _validate_value("STREAM_SOURCE_MODE", "AUTO")
        assert ok is True
        assert val == "auto"

    def test_telegram_report_time_valid(self):
        from config import _validate_value

        ok, val = _validate_value("TELEGRAM_REPORT_TIME", "09:45")
        assert ok is True
        assert val == "09:45"

    def test_telegram_report_time_empty_uses_default(self):
        from config import DEFAULTS, _validate_value

        ok, val = _validate_value("TELEGRAM_REPORT_TIME", "")
        assert ok is True
        assert val == DEFAULTS["TELEGRAM_REPORT_TIME"]

    def test_telegram_report_time_invalid_rejected(self):
        from config import _validate_value

        ok, _val = _validate_value("TELEGRAM_REPORT_TIME", "25:99")
        assert ok is False


# ---------------------------------------------------------------------------
# Coercion (_coerce_config_types)
# ---------------------------------------------------------------------------


class TestCoercionNewKeys:
    """Ensures coercion handles the new keys correctly."""

    def test_camera_url_null_coerced_to_empty(self):
        from config import DEFAULTS, _coerce_config_types

        cfg = dict(DEFAULTS, CAMERA_URL=None)
        _coerce_config_types(cfg)
        assert cfg["CAMERA_URL"] == ""

    def test_camera_url_none_string_coerced(self):
        from config import DEFAULTS, _coerce_config_types

        cfg = dict(DEFAULTS, CAMERA_URL="None")
        _coerce_config_types(cfg)
        assert cfg["CAMERA_URL"] == ""

    def test_stream_source_mode_invalid_fallback(self):
        from config import DEFAULTS, _coerce_config_types

        cfg = dict(DEFAULTS, STREAM_SOURCE_MODE="invalid")
        _coerce_config_types(cfg)
        assert cfg["STREAM_SOURCE_MODE"] == "auto"

    def test_go2rtc_api_base_trailing_slash_stripped(self):
        from config import DEFAULTS, _coerce_config_types

        cfg = dict(DEFAULTS, GO2RTC_API_BASE="http://127.0.0.1:1984/")
        _coerce_config_types(cfg)
        assert cfg["GO2RTC_API_BASE"] == "http://127.0.0.1:1984"

    def test_telegram_report_time_invalid_fallback(self):
        from config import DEFAULTS, _coerce_config_types

        cfg = dict(DEFAULTS, TELEGRAM_REPORT_TIME="invalid")
        _coerce_config_types(cfg)
        assert cfg["TELEGRAM_REPORT_TIME"] == DEFAULTS["TELEGRAM_REPORT_TIME"]


# ---------------------------------------------------------------------------
# get_settings_payload – VIDEO_SOURCE internal flag
# ---------------------------------------------------------------------------


class TestSettingsPayload:
    """VIDEO_SOURCE must be marked as internal/non-editable."""

    def test_video_source_not_editable(self):
        from config import get_settings_payload

        payload = get_settings_payload()
        vs = payload.get("VIDEO_SOURCE", {})
        assert vs.get("editable") is False
        assert vs.get("internal") is True

    def test_camera_url_is_editable(self):
        from config import get_settings_payload

        payload = get_settings_payload()
        cu = payload.get("CAMERA_URL", {})
        assert cu.get("editable") is True
        assert cu.get("internal") is False

    def test_stream_source_mode_is_editable(self):
        from config import get_settings_payload

        payload = get_settings_payload()
        ssm = payload.get("STREAM_SOURCE_MODE", {})
        assert ssm.get("editable") is True
        assert ssm.get("internal") is False

    def test_stream_source_effective_mode_present(self):
        from config import get_settings_payload

        payload = get_settings_payload()
        sem = payload.get("STREAM_SOURCE_EFFECTIVE_MODE", {})
        assert sem.get("editable") is False
        assert sem.get("internal") is True
        assert sem.get("value") in ("direct", "relay")

    def test_runtime_video_source_present(self):
        """Runtime source fields must exist and reflect actual VIDEO_SOURCE."""
        from config import get_settings_payload

        payload = get_settings_payload()
        rt_video = payload.get("STREAM_SOURCE_RUNTIME_VIDEO", {})
        assert rt_video.get("editable") is False
        assert rt_video.get("internal") is True
        assert rt_video.get("source") == "runtime"
        # Value should match what get_config() returns for VIDEO_SOURCE
        assert rt_video.get("value") is not None

    def test_runtime_mode_present(self):
        """Runtime mode field must exist and be relay or direct."""
        from config import get_settings_payload

        payload = get_settings_payload()
        rt_mode = payload.get("STREAM_SOURCE_RUNTIME_MODE", {})
        assert rt_mode.get("editable") is False
        assert rt_mode.get("internal") is True
        assert rt_mode.get("source") == "runtime"
        assert rt_mode.get("value") in ("direct", "relay")
