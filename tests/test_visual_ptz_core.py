from __future__ import annotations

from collections import deque
from dataclasses import replace

import numpy as np
import pytest

from core.visual_ptz_core import (
    ClosedLoopVisualPtz,
    VisualKeyframeStore,
    VisualPtzLimits,
    verify_motion_response,
)
from utils.path_manager import PathManager
from utils.visual_alignment import AlignmentResult


def _result(error_x: float, error_y: float, *, quality: float = 0.9):
    return AlignmentResult(
        success=True,
        method="sift",
        model="homography",
        reference_keypoints=100,
        current_keypoints=100,
        good_matches=50,
        inliers=45,
        inlier_ratio=0.9,
        coverage=0.4,
        reprojection_rmse=1.0,
        quality=quality,
        error_x=error_x,
        error_y=error_y,
        pixel_dx=error_x * 640,
        pixel_dy=error_y * 480,
        scale=1.0,
    )


class _SequenceAligner:
    def __init__(self, results):
        self.results = deque(results)
        self.reference_anchors = []

    def align(
        self,
        _reference,
        _current,
        *,
        reference_anchor=(0.5, 0.5),
        debug_path=None,
    ):
        self.reference_anchors.append(reference_anchor)
        return self.results.popleft()


def test_keyframe_store_round_trip(tmp_path):
    store = VisualKeyframeStore(PathManager(str(tmp_path)))
    frame = np.full((90, 160, 3), (20, 80, 140), dtype=np.uint8)

    metadata = store.save(3, "feeder-left", frame)
    restored = store.load(3, "feeder-left")

    assert metadata["camera_id"] == 3
    assert restored.shape == frame.shape
    assert store.list(3)[0]["keyframe_id"] == "feeder-left"


def test_keyframe_store_rejects_path_traversal(tmp_path):
    store = VisualKeyframeStore(PathManager(str(tmp_path)))
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    metadata = store.save(1, "../../feeder", frame)

    assert metadata["keyframe_id"] == "feeder"
    assert ".." not in metadata["image_path"]


def test_closed_loop_reduces_error_and_stops_in_deadband():
    moves = []
    aligner = _SequenceAligner(
        [_result(0.20, -0.12), _result(0.08, -0.05), _result(0.01, -0.01)]
    )
    controller = ClosedLoopVisualPtz(
        aligner=aligner,
        frame_supplier=lambda: np.zeros((10, 10, 3), dtype=np.uint8),
        mover=lambda pan, tilt, duration: moves.append((pan, tilt, duration)),
        stopper=lambda: None,
        limits=VisualPtzLimits(settle_sec=0.0),
        sleep=lambda _seconds: None,
    )

    run = controller.align_to(np.zeros((10, 10, 3), dtype=np.uint8))

    assert run.aligned is True
    assert run.status == "aligned"
    assert len(moves) == 2
    assert moves[0][0] > 0
    assert moves[0][1] > 0  # image target is high, so ONVIF tilt goes up
    assert run.iterations[1].progress == pytest.approx(
        _result(0.20, -0.12).error_magnitude - _result(0.08, -0.05).error_magnitude
    )


def test_closed_loop_uses_coarse_burst_and_preserves_diagonal_ratio():
    moves = []
    aligner = _SequenceAligner([_result(0.40, 0.20), _result(0.01, 0.01)])
    controller = ClosedLoopVisualPtz(
        aligner=aligner,
        frame_supplier=lambda: np.zeros((10, 10, 3), dtype=np.uint8),
        mover=lambda pan, tilt, duration: moves.append((pan, tilt, duration)),
        stopper=lambda: None,
        limits=VisualPtzLimits(
            coarse_threshold=0.15,
            coarse_min_step=0.6,
            coarse_max_step=0.6,
            coarse_move_duration_ms=400,
            coarse_settle_sec=0.0,
            max_total_motion=0.3,
        ),
        sleep=lambda _seconds: None,
    )

    run = controller.align_to(np.zeros((10, 10, 3), dtype=np.uint8))

    assert run.aligned is True
    assert moves == [(pytest.approx(0.6), pytest.approx(-0.3), 400)]
    assert run.total_motion == pytest.approx((0.6**2 + 0.3**2) ** 0.5 * 0.4)


def test_closed_loop_scales_coarse_duration_with_visual_error():
    moves = []
    aligner = _SequenceAligner([_result(0.40, 0.0), _result(0.01, 0.0)])
    controller = ClosedLoopVisualPtz(
        aligner=aligner,
        frame_supplier=lambda: np.zeros((10, 10, 3), dtype=np.uint8),
        mover=lambda pan, tilt, duration: moves.append((pan, tilt, duration)),
        stopper=lambda: None,
        limits=VisualPtzLimits(
            coarse_move_duration_ms=400,
            coarse_max_move_duration_ms=1500,
            coarse_duration_gain_ms=3000,
            coarse_settle_sec=0.0,
        ),
        sleep=lambda _seconds: None,
    )

    run = controller.align_to(np.zeros((10, 10, 3), dtype=np.uint8))

    assert run.aligned is True
    assert moves[0][2] == 1200


def test_closed_loop_serializes_diagonal_axes_for_limited_firmware():
    moves = []
    aligner = _SequenceAligner([_result(0.40, 0.20), _result(0.01, 0.01)])
    controller = ClosedLoopVisualPtz(
        aligner=aligner,
        frame_supplier=lambda: np.zeros((10, 10, 3), dtype=np.uint8),
        mover=lambda pan, tilt, duration: moves.append((pan, tilt, duration)),
        stopper=lambda: None,
        limits=VisualPtzLimits(
            coarse_min_step=0.6,
            coarse_max_step=0.6,
            coarse_move_duration_ms=400,
            coarse_settle_sec=0.0,
            serialize_axes=True,
            max_total_motion=0.4,
        ),
        sleep=lambda _seconds: None,
    )

    run = controller.align_to(np.zeros((10, 10, 3), dtype=np.uint8))

    assert run.aligned is True
    assert moves == [
        (pytest.approx(0.6), 0.0, 400),
        (0.0, pytest.approx(-0.3), 400),
    ]
    assert run.total_motion == pytest.approx((0.6 + 0.3) * 0.4)


def test_closed_loop_uses_fine_burst_below_coarse_threshold():
    moves = []
    aligner = _SequenceAligner([_result(0.10, 0.05), _result(0.01, 0.01)])
    controller = ClosedLoopVisualPtz(
        aligner=aligner,
        frame_supplier=lambda: np.zeros((10, 10, 3), dtype=np.uint8),
        mover=lambda pan, tilt, duration: moves.append((pan, tilt, duration)),
        stopper=lambda: None,
        limits=VisualPtzLimits(
            gain=1.0,
            min_step=0.08,
            max_step=0.22,
            move_duration_ms=180,
            settle_sec=0.0,
            coarse_threshold=0.15,
        ),
        sleep=lambda _seconds: None,
    )

    run = controller.align_to(np.zeros((10, 10, 3), dtype=np.uint8))

    assert run.aligned is True
    assert moves == [(pytest.approx(0.1), pytest.approx(-0.05), 180)]


def test_closed_loop_verifies_result_after_last_allowed_move():
    moves = []
    aligner = _SequenceAligner(
        [_result(0.20, 0.0), _result(0.08, 0.0), _result(0.01, 0.0)]
    )
    controller = ClosedLoopVisualPtz(
        aligner=aligner,
        frame_supplier=lambda: np.zeros((10, 10, 3), dtype=np.uint8),
        mover=lambda pan, tilt, duration: moves.append((pan, tilt, duration)),
        stopper=lambda: None,
        limits=VisualPtzLimits(max_iterations=2, settle_sec=0.0),
        sleep=lambda _seconds: None,
    )

    run = controller.align_to(np.zeros((10, 10, 3), dtype=np.uint8))

    assert run.aligned is True
    assert run.status == "aligned"
    assert len(moves) == 2
    assert run.iterations[-1].index == 2


def test_closed_loop_aborts_when_visual_error_does_not_improve():
    moves = []
    aligner = _SequenceAligner(
        [_result(0.20, 0.10), _result(0.22, 0.11), _result(0.24, 0.12)]
    )
    controller = ClosedLoopVisualPtz(
        aligner=aligner,
        frame_supplier=lambda: np.zeros((10, 10, 3), dtype=np.uint8),
        mover=lambda pan, tilt, duration: moves.append((pan, tilt, duration)),
        stopper=lambda: None,
        limits=VisualPtzLimits(settle_sec=0.0, max_stagnant_iterations=2),
        sleep=lambda _seconds: None,
    )

    run = controller.align_to(np.zeros((10, 10, 3), dtype=np.uint8))

    assert run.aligned is False
    assert run.status == "no_progress"
    assert len(moves) == 2


def test_closed_loop_sends_no_command_below_quality_threshold():
    moves = []
    controller = ClosedLoopVisualPtz(
        aligner=_SequenceAligner([_result(0.2, 0.1, quality=0.2)]),
        frame_supplier=lambda: np.zeros((10, 10, 3), dtype=np.uint8),
        mover=lambda pan, tilt, duration: moves.append((pan, tilt, duration)),
        stopper=lambda: None,
        limits=VisualPtzLimits(min_quality=0.5),
    )

    run = controller.align_to(np.zeros((10, 10, 3), dtype=np.uint8))

    assert run.status == "low_quality"
    assert moves == []


def test_closed_loop_forwards_clicked_anchor_and_emits_visual_events():
    events = []
    aligner = _SequenceAligner([_result(0.12, 0.0), _result(0.01, 0.0)])
    controller = ClosedLoopVisualPtz(
        aligner=aligner,
        frame_supplier=lambda: np.zeros((10, 10, 3), dtype=np.uint8),
        mover=lambda _pan, _tilt, _duration: None,
        stopper=lambda: None,
        limits=VisualPtzLimits(settle_sec=0.0),
        sleep=lambda _seconds: None,
    )

    run = controller.align_to(
        np.zeros((10, 10, 3), dtype=np.uint8),
        reference_anchor=(0.8, 0.3),
        on_event=lambda state, iteration: events.append((state, iteration.index)),
    )

    assert run.aligned is True
    assert aligner.reference_anchors == [(0.8, 0.3), (0.8, 0.3)]
    assert events == [("moving", 1), ("verifying", 2)]


def test_closed_loop_cancels_before_sending_a_move():
    moves = []
    controller = ClosedLoopVisualPtz(
        aligner=_SequenceAligner([_result(0.2, 0.1)]),
        frame_supplier=lambda: np.zeros((10, 10, 3), dtype=np.uint8),
        mover=lambda pan, tilt, duration: moves.append((pan, tilt, duration)),
        stopper=lambda: None,
    )

    run = controller.align_to(
        np.zeros((10, 10, 3), dtype=np.uint8),
        should_cancel=lambda: True,
    )

    assert run.status == "cancelled"
    assert moves == []


@pytest.mark.parametrize(
    ("axis", "direction", "result", "expected_response"),
    [
        ("pan", 1, _result(-0.08, 0.0), 0.08),
        ("pan", -1, _result(0.08, 0.0), -0.08),
        ("tilt", 1, _result(0.0, 0.06), 0.06),
        ("tilt", -1, _result(0.0, -0.06), -0.06),
    ],
)
def test_verify_motion_response_maps_image_motion_to_camera_axis(
    axis, direction, result, expected_response
):
    observation = verify_motion_response(result, axis=axis, direction=direction)

    assert observation.movement_detected is True
    assert observation.direction_correct is True
    assert observation.response == pytest.approx(expected_response)


def test_verify_motion_response_measures_zoom_scale():
    result = replace(_result(0.0, 0.0), scale=1.08)

    observation = verify_motion_response(result, axis="zoom", direction=1)

    assert observation.direction_correct is True
    assert observation.response == pytest.approx(0.08)
