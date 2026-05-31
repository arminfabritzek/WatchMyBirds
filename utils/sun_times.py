"""Sun-time helpers for the OD night-pause gate.

Pure functions. No I/O, no caching, no module-level state — callers
cache. This makes the helper trivially unit-testable and lets the
detection_manager TTL cache stay in the manager where it belongs.

Twilight modes
--------------
``civil`` (default): sun centre is between 0° and -6° below the
horizon. "Twilight you can still read in." Matches the empirical
break-point for camera sensors picking up useful bird detail.

``nautical``: sun centre is between -6° and -12°. Significantly
darker. Useful when running the camera with infrared assist.

``geometric``: sunrise/sunset proper (sun centre at the horizon).
The strictest mode — OD would pause earliest in the evening and
resume latest in the morning.

Offsets
-------
``start_offset_min`` is added to dusk (positive minutes = OD keeps
running later into the evening; e.g. +30 means "stop OD 30 min
after civil dusk"). Useful for late-active species (blackbird,
thrush).

``end_offset_min`` is added to dawn (negative minutes = OD starts
earlier in the morning; e.g. -45 means "resume OD 45 min before
civil dawn"). Useful for the dawn chorus (robin, wren, great tit).

Sign convention: both offsets shift the *daytime window* — positive
end-of-day extends, negative start-of-day extends. The defaults
(+30, -45) widen the active window past pure civil twilight on
both ends.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Literal

from astral import LocationInfo
from astral.sun import sun

logger = logging.getLogger(__name__)

TwilightMode = Literal["civil", "nautical", "geometric"]

# Depression angles (degrees below horizon) for each twilight mode.
# astral.sun() takes the `depression` argument for civil/nautical.
# For "geometric" we use astral's default sunrise/sunset (depression=0).
_DEPRESSION = {
    "civil": 6.0,
    "nautical": 12.0,
    "geometric": 0.0,
}


def is_daytime(
    now: datetime,
    lat: float,
    lon: float,
    *,
    start_offset_min: int = 30,
    end_offset_min: int = -45,
    twilight: TwilightMode = "civil",
) -> tuple[bool, datetime]:
    """Return whether ``now`` is within the operator-defined daytime
    window, plus the next transition timestamp.

    Args:
        now: Current time. Must be timezone-aware (UTC preferred).
        lat: Latitude in degrees. Positive north.
        lon: Longitude in degrees. Positive east.
        start_offset_min: Minutes added to dusk. Positive widens the
            window into the evening. Defaults to +30.
        end_offset_min: Minutes added to dawn. Negative widens the
            window into the early morning. Defaults to -45.
        twilight: Twilight mode. ``civil`` (default), ``nautical``,
            or ``geometric``.

    Returns:
        ``(is_currently_daytime, next_transition_utc)``. The
        transition is the timestamp at which ``is_daytime`` will
        flip — either tonight's "end of day" if currently daytime,
        or tomorrow morning's "start of day" if currently night.

    Notes:
        - Polar day / polar night edge cases: if astral cannot find
          a dawn or dusk for the date (sun never rises or never sets),
          the helper falls back to scanning forward up to 14 days.
          On the rare day where neither bound exists in that window,
          returns ``(True, now + 12h)`` defensively — OD keeps running.
        - Timezone safety: ``now`` must be aware. A naive datetime
          raises ``ValueError`` — silent timezone bugs would be much
          worse than a loud crash at startup.
    """
    if now.tzinfo is None:
        raise ValueError(
            "is_daytime: 'now' must be timezone-aware (got naive). "
            "Pass datetime.now(tz=UTC) or equivalent."
        )

    if twilight not in _DEPRESSION:
        raise ValueError(
            f"is_daytime: unknown twilight mode {twilight!r}; "
            f"expected one of {list(_DEPRESSION)}"
        )

    location = LocationInfo(
        name="custom",
        region="custom",
        timezone="UTC",
        latitude=float(lat),
        longitude=float(lon),
    )

    # Compute today's and a few neighbouring days' dawn/dusk so we can
    # find the *next* transition regardless of where in the day we
    # are. astral's sun() returns dict keys 'dawn', 'sunrise', 'noon',
    # 'sunset', 'dusk' — with depression>0 the dawn/dusk keys move to
    # civil/nautical twilight.
    transitions = _compute_transitions(location, now, twilight)

    if not transitions:
        # Polar edge case: nothing in the next 14 days. Fall back to
        # "always daytime" — OD never pauses. Safer than the inverse.
        logger.warning(
            "is_daytime: no twilight transitions found near %s "
            "(polar edge case?); defaulting to daytime.",
            now.isoformat(),
        )
        return True, now + timedelta(hours=12)

    # Apply offsets and figure out where `now` sits.
    intervals = _build_daytime_intervals(
        transitions,
        start_offset_min=start_offset_min,
        end_offset_min=end_offset_min,
    )

    for start, end in intervals:
        if start <= now < end:
            return True, end
        if now < start:
            return False, start

    # All intervals are in the past — this can only happen if the
    # window we computed was too short. Re-extend and recurse once.
    # In practice we always have at least 14 days of intervals, so
    # this is effectively unreachable; we still guard against it.
    return True, now + timedelta(hours=12)


def _compute_transitions(
    location: LocationInfo,
    now: datetime,
    twilight: TwilightMode,
) -> list[tuple[datetime, datetime]]:
    """Return ``(dawn, dusk)`` UTC pairs for the next ~14 days."""
    depression = _DEPRESSION[twilight]
    out: list[tuple[datetime, datetime]] = []
    # Start one day in the past so we can resolve the *previous*
    # daytime window if `now` falls just after midnight.
    start_date = (now - timedelta(days=1)).date()
    for offset in range(0, 16):
        d = start_date + timedelta(days=offset)
        try:
            if depression > 0:
                s = sun(location.observer, date=d, dawn_dusk_depression=depression)
            else:
                s = sun(location.observer, date=d)
            dawn = s["dawn"] if depression > 0 else s["sunrise"]
            dusk = s["dusk"] if depression > 0 else s["sunset"]
            # Force UTC tz on the returned datetimes (astral already
            # returns aware datetimes in UTC, but be defensive).
            if dawn.tzinfo is None:
                dawn = dawn.replace(tzinfo=UTC)
            if dusk.tzinfo is None:
                dusk = dusk.replace(tzinfo=UTC)
            out.append((dawn, dusk))
        except ValueError:
            # Polar day or polar night: no dawn or no dusk for this
            # date. Skip; the caller-side fallback handles 14
            # consecutive misses.
            continue
    return out


def _build_daytime_intervals(
    transitions: list[tuple[datetime, datetime]],
    *,
    start_offset_min: int,
    end_offset_min: int,
) -> list[tuple[datetime, datetime]]:
    """Apply offsets and produce ``(daytime_start, daytime_end)`` pairs.

    The offsets shift the geometric/twilight times — see module
    docstring for sign convention.
    """
    intervals: list[tuple[datetime, datetime]] = []
    for dawn, dusk in transitions:
        # end_offset_min is typically <=0; subtracting that many
        # minutes from dawn means OD starts *earlier*.
        # Sign: dawn + timedelta(end_offset_min) — for default -45,
        # daytime_start = dawn - 45 min.
        daytime_start = dawn + timedelta(minutes=end_offset_min)
        # start_offset_min is typically >=0; adding to dusk extends
        # daytime into the evening.
        daytime_end = dusk + timedelta(minutes=start_offset_min)
        # Sanity: if offsets are pathological (start > end), skip.
        if daytime_end > daytime_start:
            intervals.append((daytime_start, daytime_end))
    return intervals
