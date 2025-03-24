# ------------------------------------------------------------------------------
# Classifier Module for Image Classification (ONNX Runtime)
# detectors/classifier.py
# ------------------------------------------------------------------------------
import os

from config import load_config
config = load_config()
from logging_config import get_logger
logger = get_logger(__name__)

import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms
import numpy as np  # Import numpy


class ImageClassifier:
    def __init__(self, model_path, class_path="models/classifier_classes.txt"):
        """
        Initializes the ImageClassifier.

        Args:
            model_path (str): Path to the ONNX model file.
            class_path (str, optional): Path to the file containing class names.
                                        Defaults to "classifier_classes.txt".  Handles
                                        cases where the file might not exist.
        """
        self.model_path = model_path
        self.class_path = class_path

        # Always download the latest ONNX model from the remote repository
        logger.info("Downloading latest ONNX model from remote...")
        ImageClassifier.download_model(self.model_path)

        # Always download the latest classifier classes file from the remote repository
        logger.info("Downloading latest classifier classes file from remote...")
        ImageClassifier.download_classes(self.class_path)

        self.ort_session = ort.InferenceSession(self.model_path)
        self.classes = self._load_classes()
        self.transform = self._get_transform()

    @staticmethod
    def download_model(dest_path):
        """Downloads the latest ONNX model from the remote repository."""
        import requests
        import os

        url = "https://raw.githubusercontent.com/arminfabritzek/WatchMyBirds-Classifier/b0af22302087c8bbd8443a37d50c77d71775c74f/best_model/classifier_best.onnx"
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(dest_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"Downloaded ONNX model to: {dest_path}")
        else:
            logger.debug(f"Failed to download ONNX model. Status code: {response.status_code}")

    @staticmethod
    def download_classes(dest_path):
        """Downloads the latest classifier classes file from the remote repository."""
        import requests
        import os

        url = "https://raw.githubusercontent.com/arminfabritzek/WatchMyBirds-Classifier/b0af22302087c8bbd8443a37d50c77d71775c74f/best_model/classifier_classes.txt"
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(dest_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"Downloaded classifier classes file to: {dest_path}")
        else:
            logger.debug(f"Failed to download classifier classes file. Status code: {response.status_code}")

    def _load_classes(self):
        """Loads class names from the specified file."""
        try:
            with open(self.class_path, "r") as f:
                return [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            logger.info(f"Warning: Class file not found at {self.class_path}. Using index as class name.")
            return [str(i) for i in range(1000)]  # Return indices as strings

    def _get_transform(self):
        """Defines the image preprocessing pipeline."""
        # ImageNet statistics
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        return transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.ToTensor(),  # Convert PIL Image to Tensor
            transforms.Normalize(mean, std)  # Normalize image
        ])

    def predict(self, image_path, top_k=5):
        """
        Performs inference on a single image loaded from disk.

        Args:
            image_path (str): Path to the image file.
            top_k (int): Number of top predictions to return.

        Returns:
            tuple: (top_k_indices, top_k_confidences, top1_class_name, top1_confidence)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load image from disk and delegate to predict_from_image
        image = Image.open(image_path).convert("RGB")
        return self.predict_from_image(image, top_k=top_k)

    def predict_from_image(self, image, top_k=5):
        """
        Performs inference on a single image provided as a PIL Image or numpy array.

        Args:
            image (PIL.Image or np.ndarray): The input image.
            top_k (int): Number of top predictions to return.

        Returns:
            tuple: (top_k_indices, top_k_confidences, top1_class_name, top1_confidence)
        """
        # If image is a numpy array, convert it to a PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image = image.convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension

        # Convert to numpy for ONNX Runtime
        ort_input = {self.ort_session.get_inputs()[0].name: input_tensor.numpy()}

        # Run inference
        ort_outs = self.ort_session.run(None, ort_input)
        logits = ort_outs[0]

        # Compute softmax probabilities (numerically stable)
        exp_scores = np.exp(logits - np.max(logits))
        probabilities = exp_scores / np.sum(exp_scores)

        # Get top-k predictions
        top_k_indices = np.argsort(probabilities[0])[::-1][:top_k]
        top_k_confidences = probabilities[0][top_k_indices]

        # Get top-1 prediction and confidence
        top1_index = top_k_indices[0]
        top1_confidence = top_k_confidences[0]
        top1_class_name = self.classes[top1_index]

        return top_k_indices, top_k_confidences, top1_class_name, top1_confidence

