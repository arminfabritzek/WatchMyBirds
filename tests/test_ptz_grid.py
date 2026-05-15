"""Tests for `core.ptz_grid` — pure grid-routing math."""

import pytest

from core.ptz_grid import (
    ALLOWED_GRID_SHAPES,
    cell_for_center,
    cell_preset_name,
    normalize_grid_shape,
    parse_cell_preset_name,
    required_cell_count,
)


class TestNormalizeGridShape:
    @pytest.mark.parametrize("shape", ALLOWED_GRID_SHAPES)
    def test_allowed_shapes_pass_through(self, shape):
        assert normalize_grid_shape(list(shape)) == shape
        assert normalize_grid_shape(shape) == shape

    @pytest.mark.parametrize("bad", [None, "3x3", [3], [3, 3, 3], {}, 9, [99, 99]])
    def test_invalid_falls_back_to_3x3(self, bad):
        assert normalize_grid_shape(bad) == (3, 3)


class TestCellForCenter:
    """Bare cell lookup (no hysteresis)."""

    def test_top_left_corner(self):
        assert cell_for_center(0.0, 0.0, 3, 3) == (0, 0)

    def test_bottom_right_corner(self):
        assert cell_for_center(1.0, 1.0, 3, 3) == (2, 2)

    def test_center_of_frame_3x3_lands_in_middle_cell(self):
        assert cell_for_center(0.5, 0.5, 3, 3) == (1, 1)

    def test_2x2_split_on_axes(self):
        assert cell_for_center(0.25, 0.25, 2, 2) == (0, 0)
        assert cell_for_center(0.75, 0.25, 2, 2) == (0, 1)
        assert cell_for_center(0.25, 0.75, 2, 2) == (1, 0)
        assert cell_for_center(0.75, 0.75, 2, 2) == (1, 1)

    def test_3x4_wide_grid(self):
        # 3 rows, 4 cols → each cell is 0.333h × 0.25w.
        # x=0.1, y=0.1 → row=0, col=0.
        assert cell_for_center(0.1, 0.1, 3, 4) == (0, 0)
        # x=0.6, y=0.5 → col=int(0.6*4)=2, row=int(0.5*3)=1.
        assert cell_for_center(0.6, 0.5, 3, 4) == (1, 2)

    def test_out_of_range_clamps_to_edge(self):
        # Bird detected outside the frame (e.g. negative bbox) clamps.
        assert cell_for_center(-0.5, -0.5, 3, 3) == (0, 0)
        assert cell_for_center(2.0, 2.0, 3, 3) == (2, 2)

    def test_invalid_dims_default_to_single_cell(self):
        # Defensive: 0×0 grid would crash int(0.5 * 0); we clamp to 1×1.
        assert cell_for_center(0.5, 0.5, 0, 0) == (0, 0)


class TestCellHysteresis:
    """Adjacent-cell hysteresis prevents flap on the border."""

    def test_within_hysteresis_keeps_current_cell(self):
        # 3×3 grid, current cell (0, 0), bird hops to x=0.34 (just over
        # the 0.333 boundary). Without hysteresis → (0, 1); with 0.05
        # margin → stay at (0, 0).
        assert cell_for_center(0.34, 0.1, 3, 3, current_cell=(0, 0)) == (0, 0)

    def test_decisive_move_overrides_hysteresis(self):
        # Same setup, but bird has moved well past the margin.
        assert cell_for_center(0.6, 0.5, 3, 3, current_cell=(0, 0)) == (1, 1)

    def test_no_current_cell_no_hysteresis(self):
        # Borderline x=0.34, no current cell → pure cell lookup.
        assert cell_for_center(0.34, 0.1, 3, 3, current_cell=None) == (0, 1)

    def test_stale_current_cell_from_different_grid_discarded(self):
        # current_cell from a 4×4 grid passed to a 3×3 lookup: out of
        # range, should be ignored entirely.
        assert cell_for_center(0.5, 0.5, 3, 3, current_cell=(3, 3)) == (1, 1)

    def test_hysteresis_margin_configurable(self):
        # Larger margin → stickier current cell.
        assert (
            cell_for_center(0.4, 0.5, 3, 3, current_cell=(1, 0), hysteresis=0.1)
            == (1, 0)
        )
        # Same input, default 0.05 margin → cross the boundary.
        assert (
            cell_for_center(0.4, 0.5, 3, 3, current_cell=(1, 0), hysteresis=0.05)
            == (1, 1)
        )


class TestCellPresetNaming:
    def test_round_trip(self):
        for row in range(4):
            for col in range(4):
                assert parse_cell_preset_name(cell_preset_name(row, col)) == (
                    row,
                    col,
                )

    def test_non_grid_name_returns_none(self):
        assert parse_cell_preset_name("overview") is None
        assert parse_cell_preset_name("PresetToken_001") is None
        assert parse_cell_preset_name("") is None
        assert parse_cell_preset_name(None) is None  # type: ignore[arg-type]

    def test_malformed_grid_name_returns_none(self):
        assert parse_cell_preset_name("grid_r") is None
        assert parse_cell_preset_name("grid_rX_c0") is None
        assert parse_cell_preset_name("grid_r0_c") is None


class TestRequiredCellCount:
    @pytest.mark.parametrize(
        "shape,count",
        [((2, 2), 4), ((2, 3), 6), ((3, 3), 9), ((3, 4), 12)],
    )
    def test_counts(self, shape, count):
        assert required_cell_count(shape) == count
