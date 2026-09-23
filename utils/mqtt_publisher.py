"""Bounded, best-effort MQTT transport for outbound application events.

The transport knows nothing about detections. A later sensor consumer can use
the same broker configuration without inheriting Telegram alert policy.
"""

import json
import queue
import threading
from collections.abc import Callable
from typing import Any

from logging_config import get_logger

logger = get_logger(__name__)
EventFactory = Callable[[], dict[str, Any] | None]


class MqttPublisher:
    def __init__(self, config: dict[str, Any], queue_size: int = 128) -> None:
        self._config = config
        self._queue: queue.Queue[tuple[str, EventFactory] | None] = queue.Queue(
            queue_size
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if not self._config.get("MQTT_ENABLED"):
            return
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._stop.clear()
                self._thread = threading.Thread(
                    target=self._run, name="mqtt-publisher", daemon=True
                )
                self._thread.start()

    def settings_changed(self) -> None:
        if self._config.get("MQTT_ENABLED"):
            self.start()

    def publish(self, topic: str, event: dict[str, Any]) -> bool:
        return self.publish_factory(topic, lambda: event)

    def publish_factory(self, topic: str, factory: EventFactory) -> bool:
        if not self._config.get("MQTT_ENABLED"):
            return False
        try:
            self._queue.put_nowait((topic, factory))
            return True
        except queue.Full:
            logger.warning("MQTT event dropped: queue full")
            return False

    def stop(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=2.0)

    def _settings(self) -> tuple[Any, ...] | None:
        if not self._config.get("MQTT_ENABLED"):
            return None
        host = str(self._config.get("MQTT_HOST", "")).strip()
        if not host:
            return None
        return (
            host,
            int(self._config.get("MQTT_PORT", 1883)),
            str(self._config.get("MQTT_USERNAME", "")),
            str(self._config.get("MQTT_PASSWORD", "")),
            bool(self._config.get("MQTT_TLS", False)),
        )

    def _run(self) -> None:
        client = None
        active_settings = None
        while not self._stop.is_set():
            settings = self._settings()
            if settings != active_settings:
                if client is not None:
                    try:
                        client.disconnect()
                        client.loop_stop()
                    except Exception:
                        logger.warning("MQTT client shutdown failed")
                    client = None
                active_settings = settings
                if settings is not None:
                    try:
                        import paho.mqtt.client as mqtt

                        host, port, username, password, tls = settings
                        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
                        client.reconnect_delay_set(min_delay=1, max_delay=60)
                        client.max_queued_messages_set(128)
                        if username:
                            client.username_pw_set(username, password or None)
                        if tls:
                            client.tls_set()  # OS trust store; verify certificates.
                        client.connect_async(host, port, keepalive=60)
                        client.loop_start()
                    except Exception:
                        # Never include the exception: network libraries may
                        # include credentials or broker URLs in their text.
                        logger.warning("MQTT connection setup failed")
                        client = None
                        active_settings = None
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:
                continue
            try:
                connected = client is not None and client.is_connected()
            except Exception:
                connected = False
            if not connected:
                logger.debug("MQTT event dropped while broker is unavailable")
                continue
            try:
                topic, factory = item
                event = factory()
                if event is None:
                    continue
                payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
                info = client.publish(topic, payload, qos=1, retain=False)
                if info.rc != 0:
                    logger.warning("MQTT event dropped: publish failed (%s)", info.rc)
            except Exception:
                logger.warning("MQTT event dropped: preparation or publish failed")
        if client is not None:
            try:
                client.disconnect()
                client.loop_stop()
            except Exception:
                logger.warning("MQTT client shutdown failed")
