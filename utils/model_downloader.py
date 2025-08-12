"""Hilfsfunktionen für das Herunterladen und Cachen von Modellen."""

import json
import os
import time
from typing import Dict, Tuple, Optional

import requests

from logging_config import get_logger


logger = get_logger(__name__)

def _first_present(d: Dict[str, str], keys: Tuple[str, ...]) -> Optional[str]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None

def _guess_labels_from_weights(weights_name: str) -> Optional[str]:
    name = os.path.basename(weights_name)
    if "_best." in name:
        # map TIMESTAMP_best.onnx or TIMESTAMP_best.pt -> TIMESTAMP_labels.json
        return name.replace("_best.onnx", "_labels.json").replace("_best.pt", "_labels.json")
    return None


def _normalize_rel_path(base_url: str, rel: str) -> str:
    """Normalize a registry-relative path against a base_url that already ends with
    a task subfolder (e.g., .../resolve/main/classifier). This avoids duplicated
    segments like 'classifier/classifier/...'. Also strips a leading
    'model_registry/' if present.
    """
    if not rel:
        return rel
    rel = rel.lstrip("/")
    if rel.startswith("model_registry/"):
        rel = rel[len("model_registry/"):]
    # derive the last path segment of the base URL (expected: 'classifier' or 'object_detection')
    subdir = base_url.rstrip("/").split("/")[-1]
    if rel.startswith(f"{subdir}/"):
        rel = rel[len(subdir) + 1:]
    return rel


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
    # Resolve weights path using provided key or common alternates
    weights_rel: Optional[str] = _first_present(
        data, (weights_key, "weights_path", "onnx_path", "model", "path")
    )
    # Resolve labels/classes using provided key or common alternates
    labels_rel: Optional[str] = _first_present(
        data, (labels_key, "labels_path", "labels", "classes_path")
    )
    if not weights_rel:
        raise ValueError("latest_models.json enthält keinen gültigen Pfad für Gewichte.")

    # Normalize and try to infer labels if missing
    weights_rel_norm = _normalize_rel_path(base_url, weights_rel)
    if not labels_rel:
        guessed = _guess_labels_from_weights(weights_rel_norm)
        if guessed:
            labels_rel = guessed
            logger.warning(
                f"{labels_key} fehlt. Rate Labels aus Gewichten: {labels_rel}"
            )
    if not labels_rel:
        raise ValueError("latest_models.json enthält nicht alle erforderlichen Pfade.")
    labels_rel_norm = _normalize_rel_path(base_url, labels_rel)

    weights_path = os.path.join(model_dir, os.path.basename(weights_rel_norm))
    labels_path = os.path.join(model_dir, os.path.basename(labels_rel_norm))

    if not os.path.exists(weights_path):
        url = f"{base_url}/{weights_rel_norm}"
        logger.debug(f"Lade Gewichte von {url} nach {weights_path}")
        _download_file(url, weights_path)
    else:
        logger.debug(f"Verwende vorhandene Gewichte {weights_path}")

    if not os.path.exists(labels_path):
        url = f"{base_url}/{labels_rel_norm}"
        logger.debug(f"Lade Labels von {url} nach {labels_path}")
        _download_file(url, labels_path)
    else:
        logger.debug(f"Verwende vorhandene Labels {labels_path}")

    return weights_path, labels_path
