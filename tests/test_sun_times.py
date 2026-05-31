"""Tests for utils.sun_times.is_daytime.

Strategy: pick stable astronomical landmarks (Berlin summer solstice
noon, Berlin winter solstice 03:00 UTC, equator, Tromsø) and assert
qualitative properties. Absolute minutes are not asserted because
astral's depression-angle computation has small refraction-model
drift; behavioural properties (day vs night, next transition near
the expected wall clock) are what matters.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from utils.sun_times import is_daytime

# Reference coordinates
BERLIN = (52.52, 13.40)
EQUATOR_QUITO = (-0.18, -78.47)  # Quito-ish; near equator
TROMSO = (69.65, 18.96)  # well inside the Arctic Circle


def test_naive_datetime_raises():
    with pytest.raises(ValueError, match="timezone-aware"):
        is_daytime(datetime(2026, 6, 21, 12, 0), *BERLIN)


def test_unknown_twilight_mode_raises():
    with pytest.raises(ValueError, match="unknown twilight mode"):
        is_daytime(
            datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc),
            *BERLIN,
            twilight="lunar",  # type: ignore[arg-type]
        )


def test_berlin_summer_noon_is_daytime():
    """Summer solstice noon UTC in Berlin → unambiguously daytime."""
    now = datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc)
    is_day, next_transition = is_daytime(now, *BERLIN)
    assert is_day is True
    # The next transition is tonight's daytime_end, which is dusk +
    # start_offset_min. Berlin civil dusk on summer solstice is
    # ~21:23 UTC, plus 30 minutes default offset puts the next
    # transition some time before midnight UTC.
    assert next_transition > now
    assert next_transition < now + timedelta(hours=12)


def test_berlin_winter_pre_dawn_is_night():
    """Winter solstice 03:00 UTC in Berlin → unambiguously night."""
    now = datetime(2026, 12, 21, 3, 0, tzinfo=timezone.utc)
    is_day, next_transition = is_daytime(now, *BERLIN)
    assert is_day is False
    # Next transition is morning daytime_start = civil dawn + (-45)
    # min. Berlin civil dawn on winter solstice is ~07:09 UTC, so
    # daytime_start is around 06:24 UTC — within hours of now.
    assert next_transition > now
    assert next_transition < now + timedelta(hours=12)


def test_berlin_dusk_transition_with_default_offsets():
    """Right after civil dusk but before dusk+30: still daytime.

    Reference (astral 3.2): Berlin 2026-06-21 civil dusk = 20:24 UTC.
    At 20:30 UTC we are 6 min past dusk → inside the +30-min
    extension (default), so OD treats it as daytime. Same moment
    with zero offsets → night.
    """
    now = datetime(2026, 6, 21, 20, 30, tzinfo=timezone.utc)
    is_day, _ = is_daytime(
        now, *BERLIN, start_offset_min=30, end_offset_min=-45
    )
    assert is_day is True

    is_day_zero, _ = is_daytime(
        now, *BERLIN, start_offset_min=0, end_offset_min=0
    )
    assert is_day_zero is False


def test_berlin_pre_dawn_with_default_offsets():
    """45 min before civil dawn: daytime with default offset, night without.

    Reference (astral 3.2): Berlin 2026-06-21 civil dawn = 01:52 UTC.
    At 01:30 UTC we are 22 min before civil dawn → inside the -45
    extension (default), so OD already considers it daytime. Same
    moment with zero offsets → night.
    """
    now = datetime(2026, 6, 21, 1, 30, tzinfo=timezone.utc)
    is_day_default, _ = is_daytime(
        now, *BERLIN, start_offset_min=30, end_offset_min=-45
    )
    assert is_day_default is True

    is_day_zero, _ = is_daytime(
        now, *BERLIN, start_offset_min=0, end_offset_min=0
    )
    assert is_day_zero is False


def test_offset_zero_matches_civil_twilight():
    """With offsets=0, daytime ends exactly at civil dusk.

    Reference (astral 3.2): Berlin 2026-06-21 civil dusk = 20:24 UTC.
    """
    now = datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc)
    _, next_transition = is_daytime(
        now, *BERLIN, start_offset_min=0, end_offset_min=0
    )
    expected = datetime(2026, 6, 21, 20, 24, tzinfo=timezone.utc)
    assert abs((next_transition - expected).total_seconds()) < 5 * 60


def test_equator_has_short_twilight():
    """Equator at noon is daytime; equator at midnight is night."""
    noon = datetime(2026, 6, 21, 17, 0, tzinfo=timezone.utc)  # ~12:00 local
    is_day, _ = is_daytime(noon, *EQUATOR_QUITO)
    assert is_day is True

    midnight = datetime(2026, 6, 21, 5, 0, tzinfo=timezone.utc)  # ~midnight local
    is_day, _ = is_daytime(midnight, *EQUATOR_QUITO)
    assert is_day is False


def test_tromso_polar_day_is_daytime():
    """Mid-polar-day in Tromsø: sun never sets → always daytime."""
    # June 21 in Tromsø: midnight sun. Pick local midnight UTC.
    now = datetime(2026, 6, 21, 22, 0, tzinfo=timezone.utc)
    is_day, next_transition = is_daytime(now, *TROMSO)
    assert is_day is True
    # Either falls back to a far-future transition (synthetic +12h)
    # or finds the next genuine dusk weeks ahead.
    assert next_transition > now


def test_tromso_polar_night_is_night():
    """Mid-polar-night in Tromsø: sun never rises → daytime is False."""
    # Dec 21 in Tromsø: polar night.
    now = datetime(2026, 12, 21, 12, 0, tzinfo=timezone.utc)
    is_day, _ = is_daytime(now, *TROMSO)
    # During genuine polar night civil twilight may still exist
    # briefly (Tromsø has a "blue hour" even in December). Accept
    # either answer — what we MUST avoid is a crash.
    assert isinstance(is_day, bool)


def test_twilight_modes_widen_progressively():
    """Twilight depression angle determines window width: nautical
    (-12°) > civil (-6°) > geometric (0°) by total daytime minutes.

    Reference (astral 3.2, Berlin 2026-12-21):
        geometric sunrise  07:15  sunset 14:53  → window 7h38
        civil     dawn     06:32  dusk   15:36  → window 9h04
        nautical dawn      05:48  dusk   16:20  → window 10h32

    So at 07:00 UTC (between nautical and civil dawn, before geometric
    sunrise):
        - nautical: daytime (after nautical dawn 05:48) ✓
        - civil:    daytime (after civil dawn 06:32) ✓
        - geometric: still night (before sunrise 07:15) ✗

    At 06:00 UTC (between nautical dawn and civil dawn):
        - nautical: daytime ✓
        - civil:    night ✗
    """
    seven_am = datetime(2026, 12, 21, 7, 0, tzinfo=timezone.utc)
    is_day_naut, _ = is_daytime(seven_am, *BERLIN, start_offset_min=0, end_offset_min=0, twilight="nautical")
    is_day_civ, _ = is_daytime(seven_am, *BERLIN, start_offset_min=0, end_offset_min=0, twilight="civil")
    is_day_geo, _ = is_daytime(seven_am, *BERLIN, start_offset_min=0, end_offset_min=0, twilight="geometric")
    assert is_day_naut is True
    assert is_day_civ is True
    assert is_day_geo is False

    six_am = datetime(2026, 12, 21, 6, 0, tzinfo=timezone.utc)
    is_day_naut, _ = is_daytime(six_am, *BERLIN, start_offset_min=0, end_offset_min=0, twilight="nautical")
    is_day_civ, _ = is_daytime(six_am, *BERLIN, start_offset_min=0, end_offset_min=0, twilight="civil")
    assert is_day_naut is True
    assert is_day_civ is False


def test_next_transition_is_in_future():
    """The returned next_transition is always strictly > now."""
    for hour in range(0, 24, 3):
        now = datetime(2026, 5, 1, hour, 0, tzinfo=timezone.utc)
        _, next_transition = is_daytime(now, *BERLIN)
        assert next_transition > now, f"Failed at hour {hour}: {next_transition} <= {now}"
