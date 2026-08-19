import json
from pathlib import Path

import pytest

from camera import video_capture as vc_module
from camera.video_capture import VideoCapture


class _SizedStream:
    def __init__(self) -> None:
        self.read_sizes: list[int] = []

    def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        return bytes(size)


class _RunningProcess:
    def __init__(self, stdout: _SizedStream) -> None:
        self.stdout = stdout

    def poll(self) -> None:
        return None


def _build_cached_capture(monkeypatch: pytest.MonkeyPatch) -> VideoCapture:
    monkeypatch.setattr(
        VideoCapture, "_register_instance_for_shutdown", lambda self: None
    )
    monkeypatch.setattr(
        VideoCapture, "_prime_stream_settings_from_cache", lambda self: None
    )
    capture = VideoCapture("rtsp://example.local/stream", debug=False, auto_start=False)
    capture.stream_width = 2560
    capture.stream_height = 1920
    capture.stream_settings_loaded = True
    monkeypatch.setattr(capture, "_start_reader_thread", lambda: None)
    monkeypatch.setattr(capture, "_start_health_check_thread", lambda: None)
    return capture


def test_stale_cache_is_replaced_before_raw_frame_size_is_calculated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _build_cached_capture(monkeypatch)
    stream = _SizedStream()
    process = _RunningProcess(stream)
    events: list[str] = []

    def probe() -> None:
        events.append("probe")
        capture.stream_width = 2560
        capture.stream_height = 1440

    def setup_capture() -> None:
        events.append("setup")
        capture.backend = VideoCapture.BACKEND_FFMPEG
        capture.ffmpeg_process = process
        frame = capture._read_ffmpeg_frame()
        assert frame is not None
        assert frame.shape == (1440, 2560, 3)

    monkeypatch.setattr(capture, "_get_stream_resolution_ffprobe", probe)
    monkeypatch.setattr(capture, "_setup_capture", setup_capture)

    capture.start()

    assert events == ["probe", "setup"]
    assert stream.read_sizes == [2560 * 1440 * 3]


def test_matching_cache_is_validated_before_ffmpeg_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _build_cached_capture(monkeypatch)
    events: list[str] = []

    def probe() -> None:
        events.append("probe")
        capture.stream_width = 2560
        capture.stream_height = 1920

    monkeypatch.setattr(capture, "_get_stream_resolution_ffprobe", probe)
    monkeypatch.setattr(capture, "_setup_capture", lambda: events.append("setup"))

    capture.start()

    assert events == ["probe", "setup"]
    assert capture.resolution == (2560, 1920)


def test_probe_failure_does_not_start_ffmpeg_with_cached_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _build_cached_capture(monkeypatch)
    setup_calls: list[str] = []

    def fail_probe() -> None:
        raise RuntimeError("camera unreachable")

    monkeypatch.setattr(capture, "_get_stream_resolution_ffprobe", fail_probe)
    monkeypatch.setattr(
        capture, "_setup_capture", lambda: setup_calls.append("unsafe setup")
    )

    with pytest.raises(RuntimeError, match="validate cached RTSP stream settings"):
        capture.start()

    assert setup_calls == []


def test_detected_resolution_change_is_persisted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    capture = _build_cached_capture(monkeypatch)
    cache_path = tmp_path / "stream_settings.json"

    monkeypatch.setattr(capture, "_stream_settings_path", lambda: cache_path)
    monkeypatch.setattr(capture, "_get_ffmpeg_version", lambda: "ffmpeg test")
    monkeypatch.setattr(capture, "_setup_capture", lambda: None)
    monkeypatch.setattr(
        vc_module.subprocess,
        "check_output",
        lambda command, **_kwargs: (
            b"2560\n1440\n"
            if command[0] == "ffprobe"
            else pytest.fail(f"unexpected command: {command}")
        ),
    )

    capture.start()

    persisted = json.loads(cache_path.read_text())[str(capture.source)]
    assert (persisted["width"], persisted["height"]) == (2560, 1440)
    assert capture.resolution == (2560, 1440)


def test_recovery_revalidates_dimensions_before_ffmpeg_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _build_cached_capture(monkeypatch)
    setup_dimensions: list[tuple[int | None, int | None]] = []

    monkeypatch.setattr(capture, "stop", lambda: None)
    monkeypatch.setattr(vc_module.time, "sleep", lambda _seconds: None)

    def probe() -> None:
        capture.stream_width = 2560
        capture.stream_height = 1440

    def setup_capture(**_kwargs) -> None:
        setup_dimensions.append((capture.stream_width, capture.stream_height))

    monkeypatch.setattr(capture, "_get_stream_resolution_ffprobe", probe)
    monkeypatch.setattr(capture, "_setup_capture", setup_capture)

    capture._reinitialize_camera(reason="incomplete frame")

    assert setup_dimensions == [(2560, 1440)]


def test_recovery_probe_failure_does_not_restart_ffmpeg_with_stale_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _build_cached_capture(monkeypatch)
    setup_calls: list[str] = []

    monkeypatch.setattr(capture, "stop", lambda: None)
    monkeypatch.setattr(vc_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        capture,
        "_get_stream_resolution_ffprobe",
        lambda: (_ for _ in ()).throw(RuntimeError("camera unreachable")),
    )
    monkeypatch.setattr(
        capture,
        "_setup_capture",
        lambda **_kwargs: setup_calls.append("unsafe setup"),
    )

    with pytest.raises(RuntimeError, match="validate cached RTSP stream settings"):
        capture._reinitialize_camera(reason="incomplete frame")

    assert setup_calls == []
