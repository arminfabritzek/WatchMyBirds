import numpy as np
import pytest

from camera import video_capture as vc_module
from camera.video_capture import VideoCapture


class _FakeCap:
    def __init__(self, ret=True, frame=None):
        self._ret = ret
        self._frame = frame if frame is not None else np.zeros((1, 1, 3), dtype=np.uint8)

    def isOpened(self):
        return True

    def read(self):
        return self._ret, self._frame

    def release(self):
        return None


class _FakeThread:
    def __init__(self, alive=True):
        self._alive = alive
        self.join_calls = []

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        self.join_calls.append(timeout)
        self._alive = False


def _build_capture(monkeypatch):
    monkeypatch.setattr(VideoCapture, "_register_instance_for_shutdown", lambda self: None)
    monkeypatch.setattr(VideoCapture, "_prime_stream_settings_from_cache", lambda self: None)
    return VideoCapture("rtsp://example.local/stream", debug=False, auto_start=False)


def test_rtsp_ffmpeg_startup_missing_initial_frame_keeps_ffmpeg(monkeypatch):
    cap = _build_capture(monkeypatch)
    ffmpeg_calls = []
    opencv_calls = []
    terminated = []

    monkeypatch.setattr(vc_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(cap, "_setup_ffmpeg", lambda: ffmpeg_calls.append("ffmpeg"))
    monkeypatch.setattr(cap, "_setup_opencv_rtsp", lambda: opencv_calls.append("opencv"))
    monkeypatch.setattr(cap, "_read_ffmpeg_frame", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cap, "_terminate_ffmpeg_process", lambda reason="": terminated.append(reason)
    )

    cap._setup_capture()

    assert ffmpeg_calls == ["ffmpeg"]
    assert opencv_calls == []
    assert terminated == []
    assert cap.backend == VideoCapture.BACKEND_FFMPEG


def test_rtsp_ffmpeg_start_failure_still_falls_back_to_opencv(monkeypatch):
    cap = _build_capture(monkeypatch)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    opencv_calls = []

    monkeypatch.setattr(vc_module.time, "sleep", lambda _seconds: None)

    def _fail_ffmpeg():
        raise RuntimeError("ffmpeg failed")

    def _setup_opencv():
        opencv_calls.append("opencv")
        cap.cap = _FakeCap(ret=True, frame=frame)
        cap.stream_width = 2
        cap.stream_height = 2

    monkeypatch.setattr(cap, "_setup_ffmpeg", _fail_ffmpeg)
    monkeypatch.setattr(cap, "_setup_opencv_rtsp", _setup_opencv)

    cap._setup_capture()

    assert opencv_calls == ["opencv"]
    assert cap.backend == VideoCapture.BACKEND_OPENCV


def test_rtsp_ffmpeg_recovery_strict_requires_initial_frame(monkeypatch):
    cap = _build_capture(monkeypatch)
    ffmpeg_calls = []

    monkeypatch.setattr(vc_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(cap, "_setup_ffmpeg", lambda: ffmpeg_calls.append("ffmpeg"))
    monkeypatch.setattr(cap, "_read_ffmpeg_frame", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="Initial test frame missing"):
        cap._setup_capture(require_initial_frame=True, initial_frame_wait_sec=0.5)

    assert ffmpeg_calls == ["ffmpeg"]


def test_reinitialize_camera_uses_strict_initial_frame_validation(monkeypatch):
    cap = _build_capture(monkeypatch)
    setup_kwargs = {}

    monkeypatch.setattr(cap, "stop", lambda: None)
    monkeypatch.setattr(vc_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(cap, "_start_reader_thread", lambda: None)
    monkeypatch.setattr(cap, "_start_health_check_thread", lambda: None)

    def _setup_capture(**kwargs):
        setup_kwargs.update(kwargs)

    monkeypatch.setattr(cap, "_setup_capture", _setup_capture)

    cap._reinitialize_camera(reason="test")

    assert setup_kwargs["require_initial_frame"] is True
    assert (
        setup_kwargs["initial_frame_wait_sec"]
        == cap._recovery_initial_frame_wait_sec
    )


def test_stop_terminates_ffmpeg_before_reader_join(monkeypatch):
    cap = _build_capture(monkeypatch)
    order = []
    fake_reader = _FakeThread(alive=True)
    fake_health = _FakeThread(alive=False)

    cap.reader_thread = fake_reader
    cap.health_check_thread = fake_health
    cap.cap = None

    monkeypatch.setattr(
        cap, "_terminate_ffmpeg_process", lambda reason="": order.append("terminate")
    )

    original_join = fake_reader.join

    def _join(timeout=None):
        order.append("reader_join")
        original_join(timeout)

    fake_reader.join = _join

    cap.stop()

    assert order[:2] == ["terminate", "reader_join"]
    assert cap.reader_thread is None
    assert cap.health_check_thread is None
