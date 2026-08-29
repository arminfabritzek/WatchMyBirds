from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np
import pytest

from core.ptz_aim_core import AimBusyError, AimController
from core.visual_ptz_core import VisualPtzLimits
from utils.visual_alignment import AlignmentResult


def _result(error_x: float, error_y: float) -> AlignmentResult:
    return AlignmentResult(
        success=True,
        method="orb",
        model="homography",
        reference_keypoints=80,
        current_keypoints=75,
        good_matches=40,
        inliers=34,
        inlier_ratio=0.85,
        coverage=0.4,
        reprojection_rmse=1.0,
        quality=0.88,
        error_x=error_x,
        error_y=error_y,
        pixel_dx=error_x * 640,
        pixel_dy=error_y * 480,
        mapped_x=0.5 + error_x,
        mapped_y=0.5 + error_y,
    )


class _Aligner:
    def __init__(self, results):
        self.results = deque(results)
        self.anchors = []

    def align(self, _reference, _current, *, reference_anchor, debug_path=None):
        self.anchors.append(reference_anchor)
        return self.results.popleft()


class _Pause:
    def __init__(self):
        self.events = []

    def pause_for_external(self, reason):
        self.events.append(("pause", reason))
        return True

    def resume_from_external(self):
        self.events.append(("resume", None))
        return True


def _wait_until_finished(controller: AimController) -> dict:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        status = controller.status()
        if not status["active"]:
            return status
        time.sleep(0.005)
    raise AssertionError("aim worker did not finish")


def test_aim_session_centers_clicked_anchor_and_releases_auto_ptz():
    moves = []
    stops = []
    pause = _Pause()
    aligner = _Aligner([_result(0.16, -0.08), _result(0.01, -0.01)])
    controller = AimController(
        frame_supplier=lambda: np.zeros((120, 160, 3), dtype=np.uint8),
        mover=lambda camera_id, pan, tilt, duration: moves.append(
            (camera_id, pan, tilt, duration)
        ),
        stopper=stops.append,
        external_pause=pause,
        aligner=aligner,
        limits=VisualPtzLimits(max_iterations=5, settle_sec=0.0),
        sleep=lambda _seconds: None,
    )

    started = controller.start(7, 0.78, 0.32)
    finished = _wait_until_finished(controller)

    assert started["state"] in {"matching", "moving", "centered"}
    assert finished["state"] == "centered"
    assert finished["quality"] == 0.88
    assert finished["inliers"] == 34
    assert aligner.anchors == [(0.78, 0.32), (0.78, 0.32)]
    assert moves and moves[0][0] == 7
    assert stops[-1] == 7
    assert pause.events == [("pause", "Click-to-Aim"), ("resume", None)]


def test_aim_rejects_missing_reference_frame_and_releases_pause():
    pause = _Pause()
    controller = AimController(
        frame_supplier=lambda: None,
        mover=lambda *_args: None,
        stopper=lambda _camera_id: None,
        external_pause=pause,
    )

    try:
        controller.start(1, 0.5, 0.5)
    except RuntimeError as exc:
        assert "frame" in str(exc)
    else:
        raise AssertionError("missing frame should fail")

    assert pause.events == [("pause", "Click-to-Aim"), ("resume", None)]


def test_aim_translates_foreign_pause_owner_to_busy_without_resuming():
    class _BusyPause:
        def __init__(self):
            self.resume_called = False

        def pause_for_external(self, _reason):
            raise RuntimeError("camera is already paused by another owner")

        def resume_from_external(self):
            self.resume_called = True
            return True

    pause = _BusyPause()
    controller = AimController(
        frame_supplier=lambda: np.zeros((120, 160, 3), dtype=np.uint8),
        mover=lambda *_args: None,
        stopper=lambda _camera_id: None,
        external_pause=pause,
    )

    with pytest.raises(AimBusyError, match="another owner"):
        controller.start(1, 0.5, 0.5)

    assert pause.resume_called is False
    assert controller.status()["active"] is False


def test_aim_stays_active_until_exclusive_pause_is_released():
    class _BlockingPause(_Pause):
        def __init__(self):
            super().__init__()
            self.resume_entered = threading.Event()
            self.allow_resume = threading.Event()

        def resume_from_external(self):
            self.resume_entered.set()
            self.allow_resume.wait(timeout=1.0)
            return super().resume_from_external()

    pause = _BlockingPause()
    controller = AimController(
        frame_supplier=lambda: np.zeros((120, 160, 3), dtype=np.uint8),
        mover=lambda *_args: None,
        stopper=lambda _camera_id: None,
        external_pause=pause,
        aligner=_Aligner([_result(0.0, 0.0)]),
        limits=VisualPtzLimits(max_iterations=1, settle_sec=0.0),
        sleep=lambda _seconds: None,
    )

    controller.start(2, 0.5, 0.5)

    assert pause.resume_entered.wait(timeout=1.0)
    assert controller.status()["active"] is True
    pause.allow_resume.set()
    assert _wait_until_finished(controller)["state"] == "centered"
