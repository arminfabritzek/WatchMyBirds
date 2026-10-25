"""Build outbound MQTT events from persisted, alert-eligible bird detections."""

from datetime import UTC, datetime
from urllib.parse import quote

from config import get_config
from detectors.interfaces.persistence import DetectionPersistenceResult
from utils.db import closing_connection, is_detection_visible_in_gallery
from utils.mqtt_publisher import MqttPublisher
from utils.path_manager import get_path_manager
from utils.species_names import (
    is_known_bird_species,
    load_common_names,
    resolve_common_name,
)


class MqttDetectionService:
    def __init__(self) -> None:
        self._config = get_config()
        self._publisher = MqttPublisher(self._config)

    def start(self) -> None:
        self._publisher.start()

    def stop(self) -> None:
        self._publisher.stop()

    def settings_changed(self) -> None:
        self._publisher.settings_changed()

    def publish_detection(
        self,
        *,
        result: DetectionPersistenceResult,
        latin_name: str,
        score: float,
        capture_time: datetime,
    ) -> bool:
        if not self._config.get("MQTT_ENABLED") or not result.success:
            return False
        if not result.detection_id or not result.thumbnail_filename:
            return False
        prefix = str(self._config.get("MQTT_TOPIC_PREFIX", "watchmybirds"))
        return self._publisher.publish_factory(
            f"{prefix}/detection",
            lambda: self._build_event(result, latin_name, score, capture_time),
        )

    def _build_event(
        self,
        result: DetectionPersistenceResult,
        latin_name: str,
        score: float,
        capture_time: datetime,
    ) -> dict | None:
        with closing_connection() as conn:
            if not is_detection_visible_in_gallery(
                conn,
                result.detection_id,
                min_score=float(self._config.get("GALLERY_DISPLAY_THRESHOLD", 0.1)),
            ):
                return None
        locale = str(self._config.get("SPECIES_COMMON_NAME_LOCALE", "DE"))
        if not is_known_bird_species(latin_name, locale=locale):
            return None
        common_name = resolve_common_name(latin_name, load_common_names(locale))
        base_url = str(self._config.get("MQTT_IMAGE_BASE_URL", "")).rstrip("/")
        if not base_url:
            return None
        path_mgr = get_path_manager(str(self._config["OUTPUT_DIR"]))
        relative = path_mgr.get_derivative_path(
            result.thumbnail_filename, "thumb"
        ).relative_to(path_mgr.thumbs_dir)
        image_url = f"{base_url}/uploads/derivatives/thumbs/" + "/".join(
            quote(part) for part in relative.parts
        )
        if capture_time.tzinfo is None:
            capture_time = capture_time.astimezone()
        event = {
            "event": "bird_detection",
            "event_type": "bird_detection",
            "detection_id": result.detection_id,
            "species": common_name,
            "scientific_name": latin_name.replace("_", " "),
            "confidence": round(float(score), 4),
            "timestamp": capture_time.astimezone(UTC).isoformat(),
            "image_url": image_url,
        }
        station = str(self._config.get("STATION_NAME", "")).strip()
        if station:
            event["station"] = station
        return event
