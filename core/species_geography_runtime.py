"""Pinned local CPU occurrence inference for geographic species weighting."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
import threading
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo

import numpy as np
import piexif
import requests

from core.species_geography import GeographicAssessment, rank_geographically
from core.species_taxonomy import GeographicNameIndex, load_geographic_synonyms
from logging_config import get_logger

logger = get_logger(__name__)

GEO_MODEL_SHA256 = "80e410f811bf4455e6cb1a2d48e394737d3828fe943a444646778495b67e7bab"
GEO_LABELS_SHA256 = "8250b457e45d43fc3e77b5cbd06a1d311baf585ab9c51ed8d42e011d98534835"
GEO_MODEL_ID = "birdnet-geomodel-v3.0.4-fp16"
GEO_MODEL_FILENAME = "BirdNET+_Geomodel_V3.0.4_Global_14K_FP16.onnx"
GEO_LABELS_FILENAME = "BirdNET+_Geomodel_V3.0.4_Global_14K_Labels.txt"
GEO_DOWNLOAD_BASE = "https://github.com/birdnet-team/geomodel/releases/download/v3.0.4"


def ensure_geo_model_files(directory: Path) -> tuple[Path, Path]:
    paths = []
    for name, checksum in (
        (GEO_MODEL_FILENAME, GEO_MODEL_SHA256),
        (GEO_LABELS_FILENAME, GEO_LABELS_SHA256),
    ):
        path = directory / name
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != checksum
        ):
            directory.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with requests.get(
                    f"{GEO_DOWNLOAD_BASE}/{quote(name)}", stream=True, timeout=(10, 30)
                ) as response:
                    response.raise_for_status()
                    with tempfile.NamedTemporaryFile(
                        dir=directory, suffix=".download", delete=False
                    ) as handle:
                        temporary = Path(handle.name)
                        for block in response.iter_content(1024 * 1024):
                            handle.write(block)
                if hashlib.sha256(temporary.read_bytes()).hexdigest() != checksum:
                    raise ValueError("BirdNET download checksum mismatch")
                os.replace(temporary, path)
                temporary = None
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        paths.append(path)
    return paths[0], paths[1]


def geographic_exif_context(
    exif_bytes: bytes | str | None,
) -> tuple[datetime | None, dict[str, float] | None]:
    if not exif_bytes:
        return None, None
    try:
        exif = piexif.load(exif_bytes)
        stamp = exif.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal) or exif.get(
            "0th", {}
        ).get(piexif.ImageIFD.DateTime)
        if isinstance(stamp, bytes):
            stamp = stamp.decode("ascii")
        observed_at = datetime.strptime(stamp, "%Y:%m:%d %H:%M:%S")
        gps = exif["GPS"]
        location = {}
        for key, value_tag, ref_tag, negative_ref in (
            ("latitude", piexif.GPSIFD.GPSLatitude, piexif.GPSIFD.GPSLatitudeRef, b"S"),
            (
                "longitude",
                piexif.GPSIFD.GPSLongitude,
                piexif.GPSIFD.GPSLongitudeRef,
                b"W",
            ),
        ):
            degrees, minutes, seconds = [num / den for num, den in gps[value_tag]]
            ref = gps[ref_tag]
            if isinstance(ref, str):
                ref = ref.encode("ascii")
            valid_refs = {b"N", b"S"} if key == "latitude" else {b"E", b"W"}
            if ref not in valid_refs:
                return None, None
            location[key] = (degrees + minutes / 60 + seconds / 3600) * (
                -1 if ref == negative_ref else 1
            )
        return observed_at, location
    except Exception:
        return None, None


def observation_datetime(value: str) -> datetime | None:
    for pattern in ("%Y%m%d_%H%M%S_%f", "%Y%m%d_%H%M%S"):
        try:
            return datetime.strptime(Path(value).stem, pattern)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def capture_datetime(value: str, timezone: str) -> datetime:
    zone = ZoneInfo(timezone)
    try:
        result = datetime.fromisoformat(value)
    except ValueError:
        result = datetime.strptime(value, "%Y%m%d_%H%M%S_%f")
    return result.astimezone(zone) if result.tzinfo else result.replace(tzinfo=zone)


class LocalGeoModel:
    def __init__(self, model_path: Path, labels_path: Path) -> None:
        for path, expected in (
            (model_path, GEO_MODEL_SHA256),
            (labels_path, GEO_LABELS_SHA256),
        ):
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("BirdNET artifact checksum mismatch")
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._labels = [
            line.split("\t")[1].replace(" ", "_")
            for line in labels_path.read_text(encoding="utf-8").splitlines()
        ]
        if len(set(self._labels)) != len(self._labels):
            raise ValueError("Duplicate BirdNET scientific labels")
        self._cache: dict[tuple[float, float, int], dict[str, float]] = {}
        self._lock = threading.Lock()

    def predict(
        self, latitude: float, longitude: float, observed_at: datetime
    ) -> tuple[dict[str, float], dict[str, Any]]:
        if not (
            math.isfinite(latitude)
            and -90 <= latitude <= 90
            and math.isfinite(longitude)
            and -180 <= longitude <= 180
        ):
            raise ValueError("Invalid station coordinates")
        # BirdNET uses four bins per month, not ISO calendar weeks 1..53.
        week = (observed_at.month - 1) * 4 + min((observed_at.day - 1) // 7, 3) + 1
        key = (latitude, longitude, week)
        with self._lock:
            if key not in self._cache:
                values = self._session.run(
                    None,
                    {
                        self._session.get_inputs()[0].name: np.array(
                            [key], dtype=np.float32
                        )
                    },
                )[0][0]
                if len(values) != len(self._labels) or not np.all(np.isfinite(values)):
                    raise ValueError("Invalid BirdNET output")
                if np.any(values < 0) or np.any(values > 1):
                    raise ValueError("BirdNET output outside [0,1]")
                if len(self._cache) >= 48:
                    self._cache.clear()
                self._cache[key] = dict(
                    zip(self._labels, map(float, values), strict=True)
                )
            scores = self._cache[key]
        return scores.copy(), {
            "model_id": GEO_MODEL_ID,
            "model_sha256": GEO_MODEL_SHA256,
            "labels_sha256": GEO_LABELS_SHA256,
            "week": week,
            "latitude": latitude,
            "longitude": longitude,
        }


class GeographicWeighting:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self._config = config
        self._model: LocalGeoModel | None = None
        self._load_lock = threading.Lock()
        self._retry_at = 0.0
        self._taxonomy: GeographicNameIndex | None = None
        self._taxonomy_sha256: str | None = None

    def assess(
        self,
        candidates: list[tuple[str, float]],
        observed_at: datetime | None,
        *,
        location: Mapping[str, Any] | None = None,
    ) -> tuple[GeographicAssessment, dict[str, Any]] | None:
        if (
            not self._config.get("SPECIES_GEOGRAPHY_ENABLED", True)
            or observed_at is None
        ):
            return None
        if location is None:
            if not self._config.get("LOCATION_DATA_CONFIGURED", False):
                return None
            location = self._config.get("LOCATION_DATA")
        try:
            latitude, longitude = (
                float(location["latitude"]),
                float(location["longitude"]),
            )
            if not (
                math.isfinite(latitude)
                and -90 <= latitude <= 90
                and math.isfinite(longitude)
                and -180 <= longitude <= 180
            ):
                return None
            rank_geographically(candidates)
        except (KeyError, TypeError, ValueError):
            return None
        try:
            if time.monotonic() < self._retry_at:
                return None
            with self._load_lock:
                if time.monotonic() < self._retry_at:
                    return None
                if self._model is None:
                    directory = (
                        Path(self._config["MODEL_BASE_PATH"]) / "species_geography"
                    )
                    self._model = LocalGeoModel(*ensure_geo_model_files(directory))
            occurrence, provenance = self._model.predict(
                latitude, longitude, observed_at
            )
            with self._load_lock:
                if self._taxonomy is None:
                    groups, self._taxonomy_sha256 = load_geographic_synonyms()
                    self._taxonomy = GeographicNameIndex(occurrence, groups)
            name_matches = {
                name: self._taxonomy.resolve(name) for name, _score in candidates
            }
            mapped_occurrence = {
                name: occurrence[target]
                for name, target in name_matches.items()
                if target is not None
            }
            assessment = rank_geographically(candidates, mapped_occurrence)
            provenance = {
                key: value
                for key, value in provenance.items()
                if key not in {"latitude", "longitude"}
            }
            provenance["policy"] = (
                "image-score-times-sqrt-occurrence-floor-0.001-complete-taxonomy-v2"
            )
            provenance["taxonomy_sha256"] = self._taxonomy_sha256
            provenance["name_matches"] = name_matches
            return assessment, provenance
        except Exception as exc:
            self._retry_at = time.monotonic() + 60
            logger.warning(
                "Geographic weighting unavailable; retaining image prediction: %s", exc
            )
            return None
