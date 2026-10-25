"""Reusing capture seconds preserves local-time and DST conversion semantics."""

import os
import time
from datetime import datetime

import pytest

from core import _geom_helpers as geometry


def test_capture_second_conversion_is_reused_across_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry._timestamp_for_zone.cache_clear()
    conversions = 0

    class CountedDatetime(datetime):
        def timestamp(self) -> float:
            nonlocal conversions
            conversions += 1
            return super().timestamp()

    monkeypatch.setattr(geometry, "datetime", CountedDatetime)
    expected = datetime(2026, 10, 14, 12, 30, 15).timestamp()
    for index in range(1000):
        assert geometry.ts_to_epoch(f"20261014_123015_{index:06}") == expected
    assert conversions == 1
    geometry._timestamp_for_zone.cache_clear()


@pytest.mark.skipif(
    not hasattr(time, "tzset"), reason="Host has no process timezone switch"
)
def test_cached_timestamp_follows_timezone_switches_and_dst_gaps() -> None:
    old = os.environ.get("TZ")
    try:
        for zone in ("UTC", "Europe/Berlin", "America/New_York", "UTC"):
            os.environ["TZ"] = zone
            time.tzset()
            for stamp in (
                "20260329_023000",
                "20261025_023000",
                "19691231_233000",
                "20261014_123015",
            ):
                expected = datetime.strptime(stamp, "%Y%m%d_%H%M%S").timestamp()
                assert geometry.ts_to_epoch(stamp) == expected, (zone, stamp)
            assert geometry.ts_to_epoch("20260230_120000") == 0
            assert geometry.ts_to_epoch("") == 0
    finally:
        if old is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old
        time.tzset()
        geometry._timestamp_for_zone.cache_clear()
