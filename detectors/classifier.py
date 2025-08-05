# ------------------------------------------------------------------------------
# Classifier Module for Image Classification (ONNX Runtime)
# detectors/classifier.py
# ------------------------------------------------------------------------------
import os
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms

from config import load_config
from logging_config import get_logger
from utils.model_downloader import ensure_model_files

logger = get_logger(__name__)

HF_BASE_URL = (
    "https://huggingface.co/arminfabritzek/WatchMyBirds-Models/resolve/main/classifier"
)


class ImageClassifier:
    def __init__(self) -> None:
        """Initialisiert den Bildklassifikator und lädt das Modell."""
        self.config = load_config()
        model_dir = os.path.join(self.config["MODEL_BASE_PATH"], "classifier")
        self.model_path, self.class_path = ensure_model_files(
            HF_BASE_URL, model_dir, "weights_path", "classes_path"
        )
        try:
            self.ort_session = ort.InferenceSession(
                self.model_path, providers=["CPUExecutionProvider"]
            )
            logger.info(f"ONNX-Modell geladen: {self.model_path}")
        except Exception as exc:  # pragma: no cover - ausführliches Logging
            logger.error(f"Konnte ONNX-Modell nicht laden: {exc}")
            raise
        self.classes: List[str] = self._load_classes()
        self.CLASSIFIER_IMAGE_SIZE = self.config.get("CLASSIFIER_IMAGE_SIZE", 224)
        self.transform = self._get_transform()

    def _load_classes(self) -> List[str]:
        """Lädt Klassenbezeichnungen aus der lokalen Datei."""
        if not self.class_path or not os.path.exists(self.class_path):
            logger.warning(
                "Klassen-Datei nicht gefunden. Verwende Index als Klassennamen."
            )
            return self._get_default_class_names()
        try:
            with open(self.class_path, "r", encoding="utf-8") as file:
                classes = [line.strip() for line in file if line.strip()]
            if not classes:
                logger.warning(
                    f"Klassen-Datei {self.class_path} ist leer. Verwende Index als Klassennamen."
                )
                return self._get_default_class_names()
            logger.debug(f"{len(classes)} Klassen geladen.")
            return classes
        except Exception as exc:  # pragma: no cover - ausführliches Logging
            logger.error(f"Fehler beim Laden der Klassen: {exc}")
            return self._get_default_class_names()

    def _get_default_class_names(self) -> List[str]:
        """Erzeugt Standardklassennamen anhand der Modell-Ausgabegröße."""
        try:
            num_outputs = int(self.ort_session.get_outputs()[0].shape[-1])
            return [str(i) for i in range(num_outputs)]
        except Exception:  # pragma: no cover - nur Fallback
            logger.warning("Nutze Fallback mit 1000 Klassen.")
            return [str(i) for i in range(1000)]

    def _get_transform(self) -> transforms.Compose:
        """Definiert die Bildvorverarbeitung."""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return transforms.Compose(
            [
                transforms.Resize(
                    (self.CLASSIFIER_IMAGE_SIZE, self.CLASSIFIER_IMAGE_SIZE)
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def predict(self, image_path: str, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray, str, float]:
        """Führt Inferenz auf einem Bildpfad aus."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")
        image = Image.open(image_path).convert("RGB")
        return self.predict_from_image(image, top_k=top_k)

    def predict_from_image(
        self, image, top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, str, float]:
        """Führt Inferenz auf einem PIL- oder NumPy-Bild aus."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0)
        ort_inputs = {self.ort_session.get_inputs()[0].name: input_tensor.numpy()}
        logits = self.ort_session.run(None, ort_inputs)[0]
        exp_scores = np.exp(logits - np.max(logits))
        probabilities = exp_scores / np.sum(exp_scores)
        top_k_indices = np.argsort(probabilities[0])[::-1][:top_k]
        top_k_confidences = probabilities[0][top_k_indices]
        top1_index = int(top_k_indices[0])
        top1_confidence = float(top_k_confidences[0])
        top1_class_name = self.classes[top1_index]
        return top_k_indices, top_k_confidences, top1_class_name, top1_confidence
