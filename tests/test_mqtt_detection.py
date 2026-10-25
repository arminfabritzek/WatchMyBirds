"""MQTT event contract and failure isolation for persisted detections."""

import threading
from contextlib import nullcontext
from datetime import UTC, datetime
from unittest.mock import MagicMock

import paho.mqtt.client as mqtt
from flask import Flask

from config import get_settings_payload, validate_runtime_updates
from detectors.interfaces.persistence import DetectionPersistenceResult
from detectors.services import mqtt_detection_service
from utils.mqtt_publisher import MqttPublisher
from utils.path_manager import get_path_manager


def _config(tmp_path):
    return {
        "MQTT_ENABLED": True,
        "MQTT_HOST": "broker.local",
        "MQTT_PORT": 1883,
        "MQTT_USERNAME": "",
        "MQTT_PASSWORD": "secret",
        "MQTT_TLS": False,
        "MQTT_TOPIC_PREFIX": "watchmybirds",
        "MQTT_IMAGE_BASE_URL": "http://station.local:8050/proxy",
        "OUTPUT_DIR": str(tmp_path),
        "SPECIES_COMMON_NAME_LOCALE": "EN",
        "STATION_NAME": "Garden",
    }


def test_event_uses_persisted_thumbnail_and_utc_timestamp(monkeypatch, tmp_path):
    cfg = _config(tmp_path)
    monkeypatch.setattr(mqtt_detection_service, "get_config", lambda: cfg)
    monkeypatch.setattr(
        mqtt_detection_service, "closing_connection", lambda: nullcontext(None)
    )
    monkeypatch.setattr(
        mqtt_detection_service,
        "is_detection_visible_in_gallery",
        lambda *_args, **_kwargs: True,
    )
    service = mqtt_detection_service.MqttDetectionService()
    service._publisher = MagicMock()
    service._publisher.publish_factory.return_value = True
    result = DetectionPersistenceResult(
        success=True,
        detection_id=42,
        thumbnail_filename="20260919_114203_000000_crop_0.webp",
    )

    assert service.publish_detection(
        result=result,
        latin_name="Parus_major",
        score=0.94,
        capture_time=datetime(2026, 9, 19, 9, 42, 3, tzinfo=UTC),
    )

    topic, factory = service._publisher.publish_factory.call_args.args
    event = factory()
    assert topic == "watchmybirds/detection"
    assert event == {
        "event": "bird_detection",
        "event_type": "bird_detection",
        "detection_id": 42,
        "species": "Great Tit",
        "scientific_name": "Parus major",
        "confidence": 0.94,
        "timestamp": "2026-09-19T09:42:03+00:00",
        "image_url": "http://station.local:8050/proxy/uploads/derivatives/thumbs/2026-09-19/20260919_114203_000000_crop_0.webp",
        "station": "Garden",
    }


def test_event_image_path_is_served_by_existing_media_route(monkeypatch, tmp_path):
    from web.blueprints import media

    monkeypatch.setattr(media, "config", {"OUTPUT_DIR": str(tmp_path)})
    filename = "20260919_114203_000000_crop_0.webp"
    thumb = get_path_manager(str(tmp_path)).get_derivative_path(filename, "thumb")
    thumb.parent.mkdir(parents=True)
    thumb.write_bytes(b"bird crop")
    app = Flask(__name__)
    app.register_blueprint(media.media_bp)

    response = app.test_client().get(
        "/uploads/derivatives/thumbs/2026-09-19/" + filename
    )

    assert response.status_code == 200
    assert response.data == b"bird crop"


def test_no_event_without_saved_known_bird_or_image_url(monkeypatch, tmp_path):
    cfg = _config(tmp_path)
    monkeypatch.setattr(mqtt_detection_service, "get_config", lambda: cfg)
    monkeypatch.setattr(
        mqtt_detection_service, "closing_connection", lambda: nullcontext(None)
    )
    monkeypatch.setattr(
        mqtt_detection_service,
        "is_detection_visible_in_gallery",
        lambda *_args, **_kwargs: True,
    )
    service = mqtt_detection_service.MqttDetectionService()
    service._publisher = MagicMock()
    good = DetectionPersistenceResult(
        success=True,
        detection_id=42,
        thumbnail_filename="20260919_114203_000000_crop_0.webp",
    )
    args = dict(
        result=good,
        latin_name="Parus_major",
        score=0.94,
        capture_time=datetime.now(UTC),
    )
    for result in (
        DetectionPersistenceResult(success=False),
        DetectionPersistenceResult(success=True, detection_id=0),
    ):
        assert not service.publish_detection(**(args | {"result": result}))
    assert service.publish_detection(**(args | {"latin_name": "Phoenicurus_sp."}))
    assert service._publisher.publish_factory.call_args.args[1]() is None
    cfg["MQTT_IMAGE_BASE_URL"] = ""
    assert service.publish_detection(**args)
    assert service._publisher.publish_factory.call_args.args[1]() is None
    cfg["MQTT_IMAGE_BASE_URL"] = "http://station.local:8050"
    monkeypatch.setattr(
        mqtt_detection_service,
        "is_detection_visible_in_gallery",
        lambda *_args, **_kwargs: False,
    )
    assert service.publish_detection(**args)
    assert service._publisher.publish_factory.call_args.args[1]() is None


def test_mqtt_settings_validate_requirements_and_redact_password(monkeypatch, tmp_path):
    import config

    cfg = _config(tmp_path)
    cfg["MQTT_ENABLED"] = False
    monkeypatch.setattr(config, "get_config", lambda: cfg)
    monkeypatch.setattr(config, "load_settings_yaml", lambda _: {})
    valid, errors = validate_runtime_updates(
        {"MQTT_ENABLED": "true", "MQTT_HOST": "", "MQTT_IMAGE_BASE_URL": ""}
    )
    assert valid["MQTT_ENABLED"] is True
    assert "MQTT_HOST" in errors
    assert "MQTT_IMAGE_BASE_URL" in errors
    assert validate_runtime_updates({"MQTT_TOPIC_PREFIX": "a/#"})[1]
    assert validate_runtime_updates({"MQTT_IMAGE_BASE_URL": "http://user:pass@host"})[1]
    payload = get_settings_payload()
    assert payload["MQTT_PASSWORD"]["value"] == ""
    assert payload["MQTT_PASSWORD"]["configured"] is True


def test_mqtt_settings_persist_and_password_stays_masked(monkeypatch, tmp_path):
    import config
    from utils.settings import load_settings_yaml

    cfg = {**config.DEFAULTS, "OUTPUT_DIR": str(tmp_path)}
    monkeypatch.setattr(config, "get_config", lambda: cfg)
    valid, errors = config.validate_runtime_updates(
        {
            "MQTT_ENABLED": "true",
            "MQTT_HOST": "broker.local",
            "MQTT_PORT": "8883",
            "MQTT_TLS": "true",
            "MQTT_PASSWORD": "secret",
            "MQTT_IMAGE_BASE_URL": "http://station.local:8050",
        }
    )
    assert errors == {}
    config.update_runtime_settings(valid)

    saved = load_settings_yaml(str(tmp_path))
    assert saved["MQTT_ENABLED"] is True
    assert saved["MQTT_TLS"] is True
    assert saved["MQTT_PORT"] == 8883
    assert saved["MQTT_PASSWORD"] == "secret"
    assert get_settings_payload()["MQTT_PASSWORD"]["value"] == ""


def test_publish_queue_is_bounded_and_does_not_start_network_thread():
    cfg = {"MQTT_ENABLED": True}
    publisher = MqttPublisher(cfg, queue_size=1)
    assert publisher.publish("events", {"a": 1})
    assert not publisher.publish("events", {"a": 2})
    assert publisher._thread is None
    cfg["MQTT_ENABLED"] = False
    assert not publisher.publish("events", {"a": 3})


def test_background_client_uses_qos1_tls_and_reconnect(monkeypatch, tmp_path):
    cfg = _config(tmp_path)
    cfg.update(MQTT_TLS=True, MQTT_USERNAME="bird", MQTT_PASSWORD="secret")
    sent = threading.Event()
    client = MagicMock()
    client.is_connected.return_value = True
    client.publish.side_effect = lambda *args, **kwargs: sent.set() or MagicMock(rc=0)
    monkeypatch.setattr(mqtt, "Client", lambda *_args, **_kwargs: client)
    publisher = MqttPublisher(cfg)
    try:
        publisher.start()
        assert publisher.publish("watchmybirds/detection", {"event": "bird_detection"})
        assert sent.wait(2)
    finally:
        publisher.stop()

    client.reconnect_delay_set.assert_called_once_with(min_delay=1, max_delay=60)
    client.tls_set.assert_called_once_with()
    client.username_pw_set.assert_called_once_with("bird", "secret")
    client.connect_async.assert_called_once_with("broker.local", 1883, keepalive=60)
    client.loop_start.assert_called_once_with()
    args, kwargs = client.publish.call_args
    assert args[0] == "watchmybirds/detection"
    assert kwargs == {"qos": 1, "retain": False}


def test_unavailable_broker_drops_event_without_publish(monkeypatch, tmp_path):
    cfg = _config(tmp_path)
    checked = threading.Event()
    client = MagicMock()

    def disconnected():
        checked.set()
        return False

    client.is_connected.side_effect = disconnected
    monkeypatch.setattr(mqtt, "Client", lambda *_args, **_kwargs: client)
    publisher = MqttPublisher(cfg)
    try:
        publisher.start()
        assert publisher.publish("watchmybirds/detection", {"event": "bird_detection"})
        assert checked.wait(2)
    finally:
        publisher.stop()

    client.publish.assert_not_called()
