from pathlib import Path

import cv2
import numpy as np
import piexif

from utils.ingest import _check_inbox_exif_requirements


def _write_minimal_jpeg(path: Path) -> None:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), img), "cv2 failed to write test JPEG"


def _degrees_to_dms_rational(degrees_float: float):
    degrees_float = abs(degrees_float)
    d = int(degrees_float)
    m_float = (degrees_float - d) * 60
    m = int(m_float)
    s_int = max(0, int((m_float - m) * 60 * 1000))
    return [(d, 1), (m, 1), (s_int, 1000)]


def _insert_exif(
    path: Path,
    *,
    dt_original: str | None = None,
    dt_digitized: str | None = None,
    gps: tuple[float, float] | None = None,
) -> None:
    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}}

    if dt_original:
        exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = dt_original
    if dt_digitized:
        exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = dt_digitized

    if gps:
        lat, lon = gps
        exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = "N" if lat >= 0 else "S"
        exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = _degrees_to_dms_rational(lat)
        exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = "E" if lon >= 0 else "W"
        exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = _degrees_to_dms_rational(lon)

    piexif.insert(piexif.dump(exif_dict), str(path))


def test_exif_gate_accepts_datetime_original_and_gps(tmp_path: Path):
    p = tmp_path / "img.jpg"
    _write_minimal_jpeg(p)
    _insert_exif(
        p,
        dt_original="2026:02:10 09:00:00",
        gps=(52.516, 13.377),
    )

    ok, reason, details = _check_inbox_exif_requirements(
        str(p),
        require_datetime=True,
        require_gps=True,
    )

    assert ok is True
    assert reason is None
    assert details["has_exif_datetime"] is True
    assert details["exif_datetime_source"] == "DateTimeOriginal"
    assert details["has_exif_gps"] is True


def test_exif_gate_accepts_datetime_digitized_when_original_missing(tmp_path: Path):
    p = tmp_path / "img.jpg"
    _write_minimal_jpeg(p)
    _insert_exif(
        p,
        dt_digitized="2026:02:10 09:00:00",
        gps=(52.516, 13.377),
    )

    ok, reason, details = _check_inbox_exif_requirements(
        str(p),
        require_datetime=True,
        require_gps=True,
    )

    assert ok is True
    assert reason is None
    assert details["has_exif_datetime"] is True
    assert details["exif_datetime_source"] == "DateTimeDigitized"
    assert details["has_exif_gps"] is True


def test_exif_gate_rejects_missing_gps(tmp_path: Path):
    p = tmp_path / "img.jpg"
    _write_minimal_jpeg(p)
    _insert_exif(p, dt_original="2026:02:10 09:00:00")

    ok, reason, details = _check_inbox_exif_requirements(
        str(p),
        require_datetime=True,
        require_gps=True,
    )

    assert ok is False
    assert reason == "missing_exif_gps"
    assert details["has_exif_datetime"] is True
    assert details["has_exif_gps"] is False


def test_exif_gate_rejects_missing_datetime(tmp_path: Path):
    p = tmp_path / "img.jpg"
    _write_minimal_jpeg(p)
    _insert_exif(p, gps=(52.516, 13.377))

    ok, reason, details = _check_inbox_exif_requirements(
        str(p),
        require_datetime=True,
        require_gps=True,
    )

    assert ok is False
    assert reason == "missing_exif_datetime"
    assert details["has_exif_datetime"] is False
    assert details["has_exif_gps"] is True


def test_exif_gate_rejects_invalid_datetime_format(tmp_path: Path):
    p = tmp_path / "img.jpg"
    _write_minimal_jpeg(p)
    _insert_exif(
        p,
        dt_original="not-a-datetime",
        gps=(52.516, 13.377),
    )

    ok, reason, details = _check_inbox_exif_requirements(
        str(p),
        require_datetime=True,
        require_gps=True,
    )

    assert ok is False
    assert reason == "missing_exif_datetime"
    assert details["has_exif_datetime"] is False
    assert details["has_exif_gps"] is True


def test_db_schema_contains_inbox_ingest_events(tmp_path: Path):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True)

    # Reset schema init flag so we get a fresh schema.
    import utils.db.connection as db_mod

    db_mod._schema_initialized_paths.clear()

    mock_cfg = {"OUTPUT_DIR": str(output_dir)}

    # Patch get_config used by the DB layer.
    from unittest.mock import patch

    with patch("utils.db.connection.get_config", return_value=mock_cfg):
        from utils.db.connection import get_connection

        conn = get_connection()
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]
        conn.close()

    assert "inbox_ingest_events" in tables
