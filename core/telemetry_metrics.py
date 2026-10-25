"""Bounded, consent-gated daily diagnostics; no event details or content."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

from config import get_config
from utils.path_manager import PathManager

CONSENT_VERSION = 2
FLAGS = frozenset(
    {
        "setup_completed",
        "camera_frame",
        "camera_error",
        "model_ready",
        "model_error",
        "inference_ok",
        "inference_error",
        "review_used",
        "correction_used",
        "export_used",
        "web_error",
    }
)
RANGES = {"uptime_bucket": 5, "inference_latency_bucket": 5}


def consented(cfg: dict) -> bool:
    return (
        cfg.get("telemetry_enabled") is True
        and type(cfg.get("telemetry_consent_version")) is int
        and cfg["telemetry_consent_version"] == CONSENT_VERSION
    )


class DailyMetrics:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.scope: tuple | None = None
        self.state: dict = {}

    def _load(self, cfg: dict) -> Path:
        path = PathManager(
            str(cfg.get("OUTPUT_DIR", "./data/output"))
        ).get_telemetry_metrics_path()
        scope = (
            str(path),
            cfg.get("telemetry_installation_id", ""),
            cfg.get("telemetry_consent_at", ""),
        )
        if scope != self.scope:
            self.scope, self.state = (
                scope,
                {"id": scope[1], "consent_at": scope[2], "days": {}, "sent": {}},
            )
            try:
                if path.stat().st_size > 65536:
                    raise ValueError("Oversized diagnostic queue")
                data = json.loads(path.read_text())
                if (
                    data.get("id") == scope[1]
                    and data.get("consent_at", "") == scope[2]
                    and isinstance(data.get("days"), dict)
                    and isinstance(data.get("sent"), dict)
                ):
                    valid_days = {}
                    for day, values in data["days"].items():
                        if not isinstance(day, str) or not isinstance(values, dict):
                            continue
                        try:
                            if (
                                datetime.strptime(day, "%Y-%m-%d").date().isoformat()
                                != day
                            ):
                                continue
                        except ValueError:
                            continue
                        valid_days[day] = {
                            key: value
                            for key, value in values.items()
                            if key in FLAGS | RANGES.keys()
                            and type(value) is int
                            and 0 <= value <= RANGES.get(key, 1)
                        }
                    valid_sent = {
                        day: value
                        for day, value in data["sent"].items()
                        if day in valid_days
                        and isinstance(value, dict)
                        and type(value.get("at")) is int
                        and isinstance(value.get("values"), dict)
                    }
                    self.state = {
                        "id": scope[1],
                        "consent_at": scope[2],
                        "days": valid_days,
                        "sent": valid_sent,
                    }
            except (OSError, ValueError, AttributeError):
                pass
        cutoff = (datetime.now(UTC).date() - timedelta(days=7)).isoformat()
        for group in ("days", "sent"):
            self.state[group] = {
                d: v
                for d, v in self.state[group].items()
                if cutoff <= d <= datetime.now(UTC).date().isoformat()
            }
        return path

    def _save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".telemetry-metrics-")
        try:
            with os.fdopen(fd, "w") as out:
                json.dump(self.state, out)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def record(self, key: str, value: int = 1) -> None:
        try:
            cfg = get_config()
            if not consented(cfg) or key not in FLAGS | RANGES.keys():
                return
            if type(value) is not int or not 0 <= value <= RANGES.get(key, 1):
                return
            with self.lock:
                if not consented(cfg):
                    return
                path = self._load(cfg)
                day = datetime.now(UTC).date().isoformat()
                values = self.state["days"].setdefault(day, {})
                if value > values.get(key, -1):
                    previous = values.get(key)
                    values[key] = value
                    try:
                        self._save(path)
                    except OSError:
                        if previous is None:
                            values.pop(key, None)
                        else:
                            values[key] = previous
                        raise
        except (OSError, ValueError, TypeError):
            logging.getLogger(__name__).debug("Cannot record local diagnostics")

    def pending(self, cfg: dict, now: int) -> list[tuple[str, dict]]:
        if not consented(cfg):
            return []
        with self.lock:
            self._load(cfg)
            today = datetime.now(UTC).date().isoformat()
            return [
                (day, dict(values))
                for day, values in sorted(self.state["days"].items())
                if (
                    day < today
                    and self.state["sent"].get(day, {}).get("values") != values
                )
                or (
                    day == today
                    and now - self.state["sent"].get(day, {}).get("at", 0) >= 3600
                )
            ]

    def acknowledge(self, cfg: dict, day: str, values: dict, now: int) -> None:
        with self.lock:
            path = self._load(cfg)
            self.state["sent"][day] = {"at": now, "values": values}
            self._save(path)

    def clear(self, cfg: dict) -> None:
        with self.lock:
            path = self._load(cfg)
            self.state = {
                "id": cfg.get("telemetry_installation_id", ""),
                "consent_at": cfg.get("telemetry_consent_at", ""),
                "days": {},
                "sent": {},
            }
            if path.exists():
                self._save(path)


metrics = DailyMetrics()
