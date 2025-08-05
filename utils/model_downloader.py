"""Hilfsfunktionen für das Herunterladen und Cachen von Modellen."""

import json
import os
import time
from typing import Dict, Tuple

import requests

from logging_config import get_logger

logger = get_logger(__name__)


def _download_file(url: str, dest: str, retries: int = 3, timeout: int = 60) -> bool:
    """Lädt eine Datei von einer URL herunter."""
    if os.path.exists(dest):
        logger.debug(f"Datei existiert bereits und wird übersprungen: {dest}")
        return True
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            logger.info(f"Datei heruntergeladen: {dest}")
            return True
        except requests.RequestException as exc:
            logger.warning(
                f"Download-Versuch {attempt}/{retries} für {url} fehlgeschlagen: {exc}"
            )
            if attempt < retries:
                time.sleep(1)
    logger.error(f"Download endgültig fehlgeschlagen für {url}")
    return False


def fetch_latest_json(base_url: str, cache_dir: str) -> Dict[str, str]:
    """Lädt *latest_models.json* und legt sie im Cache ab."""
    latest_url = f"{base_url}/latest_models.json"
    local_path = os.path.join(cache_dir, "latest_models.json")
    try:
        response = requests.get(latest_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        os.makedirs(cache_dir, exist_ok=True)
        with open(local_path, "w", encoding="utf-8") as file:
            json.dump(data, file)
        logger.info(f"Aktualisierte {local_path}")
        return data
    except requests.RequestException as exc:
        logger.warning(f"Fehler beim Abrufen von {latest_url}: {exc}")
        if os.path.exists(local_path):
            logger.info(f"Verwende lokalen Cache {local_path}")
            with open(local_path, "r", encoding="utf-8") as file:
                return json.load(file)
        raise


def ensure_model_files(
    base_url: str, model_dir: str, weights_key: str, labels_key: str
) -> Tuple[str, str]:
    """Stellt sicher, dass Gewichte und Labels lokal vorhanden sind."""
    data = fetch_latest_json(base_url, model_dir)
    weights_rel = data.get(weights_key)
    labels_rel = data.get(labels_key)
    if not weights_rel or not labels_rel:
        raise ValueError("latest_models.json enthält nicht alle erforderlichen Pfade.")

    weights_path = os.path.join(model_dir, os.path.basename(weights_rel))
    labels_path = os.path.join(model_dir, os.path.basename(labels_rel))

    if not os.path.exists(weights_path):
        _download_file(f"{base_url}/{weights_rel}", weights_path)
    else:
        logger.debug(f"Verwende vorhandene Gewichte {weights_path}")

    if not os.path.exists(labels_path):
        _download_file(f"{base_url}/{labels_rel}", labels_path)
    else:
        logger.debug(f"Verwende vorhandene Labels {labels_path}")

    return weights_path, labels_path
