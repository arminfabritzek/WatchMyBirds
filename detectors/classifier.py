import os

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

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
        self.model_id = ""  # Will be populated on lazy load

        self._initialized = False
        self.ort_session = None
        self.classes = None
        self.CLASSIFIER_IMAGE_SIZE = 224  # Default, will be updated on init

        # Normalization constants (ImageNet defaults)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def _ensure_initialized(self) -> None:
        """Lazy initialization: loads ONNX model on first use."""
        if self._initialized:
            return

        logger.info("Initializing classifier (lazy-load triggered)...")
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
        except Exception as exc:
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

        self._initialized = True

    def _load_classes(self) -> list[str]:
        """Loads class labels from the local file."""
        if not self.class_path or not os.path.exists(self.class_path):
            logger.warning("Class file not found. Using index as class names.")
            return self._get_default_class_names()
        try:
            with open(self.class_path, encoding="utf-8") as file:
                classes = [line.strip() for line in file if line.strip()]
            if not classes:
                logger.warning(
                    f"Class file {self.class_path} is empty. Using index as class names."
                )
                return self._get_default_class_names()
            logger.debug(f"{len(classes)} classes loaded.")
            return classes
        except Exception as exc:
            logger.error(f"Error loading classes: {exc}")
            return self._get_default_class_names()

    def _get_default_class_names(self) -> list[str]:
        """Generates default class names based on the model output size."""
        try:
            num_outputs = int(self.ort_session.get_outputs()[0].shape[-1])
            return [str(i) for i in range(num_outputs)]
        except Exception:
            logger.warning("Using fallback with 1000 classes.")
            return [str(i) for i in range(1000)]

    def predict(
        self, image_path: str, top_k: int = 5
    ) -> tuple[np.ndarray, np.ndarray, str, float]:
        """Runs inference on an image path."""
        self._ensure_initialized()  # Lazy load on first use
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Using OpenCV to load image as BGR
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Could not read image at {image_path}")

        # Convert to RGB for classification
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.predict_from_image(img_rgb, top_k=top_k)

    def predict_from_image(
        self, image, top_k: int = 5
    ) -> tuple[np.ndarray, np.ndarray, str, float]:
        """Runs inference on a NumPy RGB image."""
        self._ensure_initialized()  # Lazy load on first use

        # Ensure image is numpy array
        if isinstance(image, Image.Image):
            image = np.asarray(image)

        # 1. Resize
        resized = cv2.resize(
            image,
            (self.CLASSIFIER_IMAGE_SIZE, self.CLASSIFIER_IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )

        # 2. To Tensor: (C, H, W) and normalize to 0-1
        tensor = resized.transpose((2, 0, 1)).astype(np.float32) / 255.0

        # 3. Normalize with ImageNet stats
        tensor = (tensor - self.mean) / self.std

        # 4. Add batch dimension
        input_tensor = np.expand_dims(tensor, axis=0)

        # Run ONNX inference
        ort_inputs = {self.ort_session.get_inputs()[0].name: input_tensor}
        logits = self.ort_session.run(None, ort_inputs)[0]

        # Softmax
        exp_scores = np.exp(logits - np.max(logits))
        probabilities = exp_scores / np.sum(exp_scores)

        # Extract Top-K
        top_k_indices = np.argsort(probabilities[0])[::-1][:top_k]
        top_k_confidences = probabilities[0][top_k_indices]

        top1_index = int(top_k_indices[0])
        top1_confidence = float(top_k_confidences[0])
        top1_class_name = self.classes[top1_index]

        return top_k_indices, top_k_confidences, top1_class_name, top1_confidence
