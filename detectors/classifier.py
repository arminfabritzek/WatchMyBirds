# ------------------------------------------------------------------------------
# Classifier Module for Image Classification (ONNX Runtime)
# detectors/classifier.py
# ------------------------------------------------------------------------------
import os
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
# torchvision.transforms imported lazily in _get_transform()

from config import get_config
from logging_config import get_logger
from utils.model_downloader import ensure_model_files, load_latest_identifier

logger = get_logger(__name__)

HF_BASE_URL = (
    "https://huggingface.co/arminfabritzek/WatchMyBirds-Models/resolve/main/classifier"
)


class ImageClassifier:
    def __init__(self) -> None:
        """Initializes the image classifier (lazy-loads model on first use)."""
        self.config = get_config()
        # Lazy loading: defer model check and ONNX session creation until first prediction
        self.model_dir = os.path.join(self.config["MODEL_BASE_PATH"], "classifier")
        self.model_path = None
        self.class_path = None
        self.model_id = "" # Will be populated on lazy load

        self._initialized = False
        self.ort_session = None
        self.classes = None
        self.CLASSIFIER_IMAGE_SIZE = 224  # Default, will be updated on init
        self.transform = None

    def _ensure_initialized(self) -> None:
        """Lazy initialization: loads ONNX model on first use."""
        if self._initialized:
            return
            
        logger.info(f"Initializing classifier (lazy-load triggered)...")
        # Perform blocking model check/download here in the worker thread
        self.model_path, self.class_path = ensure_model_files(
            HF_BASE_URL, self.model_dir, "weights_path", "classes_path"
        )
        ident = load_latest_identifier(self.model_dir)
        self.model_id = ident if ident else os.path.basename(self.model_path)

            
        logger.info(f"Lazy-loading ONNX classifier model: {self.model_path}")
        try:
            self.ort_session = ort.InferenceSession(
                self.model_path, providers=["CPUExecutionProvider"]
            )
            logger.info(f"ONNX model loaded: {self.model_path}")
        except Exception as exc:  # pragma: no cover - ausführliches Logging
            logger.error(f"Could not load ONNX model: {exc}")
            raise
            
        self.classes = self._load_classes()
        
        # Dynamically detect image size from model input
        try:
            input_shape = self.ort_session.get_inputs()[0].shape
            self.CLASSIFIER_IMAGE_SIZE = int(input_shape[2])
            logger.info(
                f"Detected model input size: {self.CLASSIFIER_IMAGE_SIZE}x{input_shape[3]}"
            )
        except Exception as exc:
            logger.warning(
                f"Could not automatically determine input size ({exc}), using fallback 224"
            )
            self.CLASSIFIER_IMAGE_SIZE = 224

        self.transform = self._get_transform()
        self._initialized = True

    def _load_classes(self) -> List[str]:
        """Loads class labels from the local file."""
        if not self.class_path or not os.path.exists(self.class_path):
            logger.warning(
                "Class file not found. Using index as class names."
            )
            return self._get_default_class_names()
        try:
            with open(self.class_path, "r", encoding="utf-8") as file:
                classes = [line.strip() for line in file if line.strip()]
            if not classes:
                logger.warning(
                    f"Class file {self.class_path} is empty. Using index as class names."
                )
                return self._get_default_class_names()
            logger.debug(f"{len(classes)} classes loaded.")
            return classes
        except Exception as exc:  # pragma: no cover - ausführliches Logging
            logger.error(f"Error loading classes: {exc}")
            return self._get_default_class_names()

    def _get_default_class_names(self) -> List[str]:
        """Generates default class names based on the model output size."""
        try:
            num_outputs = int(self.ort_session.get_outputs()[0].shape[-1])
            return [str(i) for i in range(num_outputs)]
        except Exception:  # pragma: no cover - nur Fallback
            logger.warning("Using fallback with 1000 classes.")
            return [str(i) for i in range(1000)]

    def _get_transform(self):
        """Defines the image preprocessing."""
        import torchvision.transforms as transforms  # Lazy import
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
        """Runs inference on an image path."""
        self._ensure_initialized()  # Lazy load on first use
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        return self.predict_from_image(image, top_k=top_k)

    def predict_from_image(
        self, image, top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, str, float]:
        """Runs inference on a PIL or NumPy image."""
        self._ensure_initialized()  # Lazy load on first use
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

