# ------------------------------------------------------------------------------
# Detector Module for Object Detection (Modularized, ONNX Runtime)
# detectors/detector.py
# ------------------------------------------------------------------------------
from config import load_config
config = load_config()
from logging_config import get_logger
logger = get_logger(__name__)
import os
import cv2
import numpy as np
import onnxruntime
import requests
import hashlib
import json
from PIL import Image, ImageDraw, ImageFont


# ------------------------------------------------------------------------------
# Base Detection Model Interface
# ------------------------------------------------------------------------------
class BaseDetectionModel:
    def detect(self, frame, confidence_threshold):
        """
        Perform detection on the provided frame.
        Must return a tuple: (annotated_frame, detection_info_list)
        """
        raise NotImplementedError


# ------------------------------------------------------------------------------
# ONNX Runtime Model Wrapper
# ------------------------------------------------------------------------------
class ONNXModel(BaseDetectionModel):
    def __init__(self, debug=False):
        """
        Initialize the ONNX Runtime model.
        """
        self.debug = debug
        model_env = config["YOLO8N_MODEL_PATH"]  # Use ONNX model path.
        self.model_path = model_env if model_env else "models/best.onnx"
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        # Check if the model file exists; if not, download it.
        if not os.path.exists(self.model_path):
            logger.info(f"Model file not found at {self.model_path}. Downloading...")
            self.download_best_model()
            self.download_latest_labels()
        else:
            logger.debug(f"Model file found at {self.model_path}. Skipping download.")

        # Initialize ONNX Runtime session.  Handles potential errors.
        try:
            self.session = onnxruntime.InferenceSession(
                self.model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            logger.info(f"ONNX model loaded from {self.model_path} using CPUExecutionProvider")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}", exc_info=True)
            raise

        self.input_size = ONNXModel.get_model_input_size(self.session)

        # Set the labels paths
        labels_json_path = os.path.join(os.path.dirname(self.model_path), "labels.json")

        # Now load the class names from the JSON file.
        if os.path.exists(labels_json_path):
            try:
                with open(labels_json_path, "r") as f:
                    self.class_names = json.load(f)
            except Exception as e:
                logger.error(f"Error loading labels from {labels_json_path}: {e}", exc_info=True)
                self.class_names = {}
        else:
            logger.warning(f"Label JSON file not found at {labels_json_path}. Using default class name.")
            self.class_names = {}

        self.inference_error_count = 0

        # Warm-up (optional, but good practice)
        dummy_image = cv2.imread("assets/static_placeholder.jpg")
        if dummy_image is not None:
            try:
                self.detect(dummy_image, 0.5)  # Warm-up call
                logger.info("Model warm-up successful.")
            except Exception as e:
                logger.error(f"Model warm-up failed: {e}", exc_info=True)
        else:
            logger.warning("Dummy image for model warm-up not found.")

    @staticmethod
    def get_model_input_size(session):
        """Gets the model's expected input size from the ONNX session."""
        input_shape = session.get_inputs()[0].shape
        return (input_shape[2], input_shape[3])

    def download_best_model(self):
        """
        Download the latest best.onnx model from GitHub, handling hash comparison.
        """
        # URL pointing to the ONNX model
        model_url = "https://raw.githubusercontent.com/arminfabritzek/WatchMyBirds-Train-YOLO/main/best_model/weights/best.onnx"
        dest_path = self.model_path
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        response = requests.get(model_url, stream=True)
        if response.status_code != 200:
            logger.error(
                f"Failed to download model. Status code: {response.status_code}")
            return

        downloaded_data = response.content
        downloaded_hash = hashlib.sha256(downloaded_data).hexdigest()
        logger.info(f"Downloaded model hash: {downloaded_hash}")

        if os.path.exists(dest_path):
            with open(dest_path, "rb") as f:
                local_data = f.read()
            local_hash = hashlib.sha256(local_data).hexdigest()
            logger.info(f"Local model hash: {local_hash}")
        else:
            local_hash = None

        if local_hash != downloaded_hash:
            with open(dest_path, "wb") as f:
                f.write(downloaded_data)
            logger.info(f"Best model updated at {dest_path}")
        else:
            logger.info("Local model is already up-to-date.")

    def download_latest_labels(self):
        """
        Download the latest labels.json file from GitHub and store it next to the model.
        """
        labels_url = "https://raw.githubusercontent.com/arminfabritzek/WatchMyBirds-Train-YOLO/main/best_model/weights/labels.json"
        labels_dest_path = os.path.join(os.path.dirname(self.model_path), "labels.json")
        os.makedirs(os.path.dirname(labels_dest_path), exist_ok=True)

        response = requests.get(labels_url, stream=True)
        if response.status_code != 200:
            logger.error(f"Failed to download labels.json. Status code: {response.status_code}")
            return

        downloaded_data = response.content
        downloaded_hash = hashlib.sha256(downloaded_data).hexdigest()
        logger.info(f"Downloaded labels.json hash: {downloaded_hash}")

        if os.path.exists(labels_dest_path):
            with open(labels_dest_path, "rb") as f:
                local_data = f.read()
            local_hash = hashlib.sha256(local_data).hexdigest()
            logger.info(f"Local labels.json hash: {local_hash}")
        else:
            local_hash = None

        if local_hash != downloaded_hash:
            with open(labels_dest_path, "wb") as f:
                f.write(downloaded_data)
            logger.info(f"labels.json updated at {labels_dest_path}")
        else:
            logger.info("Local labels.json is already up-to-date.")

    def preprocess_image(self, img):
        original_image = img.copy()
        h, w = img.shape[:2]
        input_height, input_width = self.input_size
        ratio = min(input_width / w, input_height / h)
        resized_width = int(w * ratio)
        resized_height = int(h * ratio)

        # Resize the image
        resized_image = cv2.resize(original_image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        # Create a padded image with fill value 114
        padded_image = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
        dw = (input_width - resized_width) // 2
        dh = (input_height - resized_height) // 2
        padded_image[dh:dh + resized_height, dw:dw + resized_width, :] = resized_image

        # Convert from BGR to RGB, normalize and change shape from HWC to CHW
        image_data = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
        image_data = image_data.transpose((2, 0, 1)).astype(np.float32) / 255.0
        image_data = np.expand_dims(image_data, axis=0)
        return image_data, original_image, ratio, dw, dh

    def postprocess_output(self, output, ratio, dw, dh, original_width, original_height, conf_threshold=0.25):
        # Remove batch dimension
        predictions = np.squeeze(output[0])

        # Filter detections by confidence threshold.
        mask = predictions[:, 4] > conf_threshold
        predictions = predictions[mask]
        if predictions.size == 0:
            return []

        # Rescale boxes from padded image coordinates back to original image.
        boxes = predictions[:, :4].copy()
        boxes[:, 0] = (boxes[:, 0] - dw) / ratio  # x1
        boxes[:, 1] = (boxes[:, 1] - dh) / ratio  # y1
        boxes[:, 2] = (boxes[:, 2] - dw) / ratio  # x2
        boxes[:, 3] = (boxes[:, 3] - dh) / ratio  # y2

        # Clip the boxes to the original image size.
        boxes[:, 0] = np.clip(boxes[:, 0], 0, original_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, original_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, original_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, original_height)

        detections = []
        for pred in predictions:
            x1, y1, x2, y2, conf, cls = pred
            # Compute coordinates from padded image to original image
            x1_val = int((x1 - dw) / ratio) if (dw or dh) else int(x1)
            y1_val = int((y1 - dh) / ratio) if (dw or dh) else int(y1)
            x2_val = int((x2 - dw) / ratio) if (dw or dh) else int(x2)
            y2_val = int((y2 - dh) / ratio) if (dw or dh) else int(y2)
            # Ensure the top-left is less than bottom-right.
            x1_val, x2_val = min(x1_val, x2_val), max(x1_val, x2_val)
            y1_val, y2_val = min(y1_val, y2_val), max(y1_val, y2_val)

            detections.append({
                "class_name": self.class_names.get(str(int(cls)), str(int(cls))),
                "confidence": float(conf),
                "x1": x1_val,
                "y1": y1_val,
                "x2": x2_val,
                "y2": y2_val,
                "class": int(cls)
            })
        return detections

    def detect(self, frame, confidence_threshold):
        detection_info_list = []
        annotated_frame = frame.copy()
        try:
            # Preprocess the frame and retrieve ratio and padding info
            processed_image, original_image, ratio, dw, dh = self.preprocess_image(frame)
            original_width, original_height = original_image.shape[1], original_image.shape[0]

            # Run inference using ONNX Runtime
            inputs = {self.input_name: processed_image}
            outputs = self.session.run(None, inputs)

            # Postprocess to get final detections
            detections = self.postprocess_output(outputs, ratio, dw, dh, original_width, original_height, confidence_threshold)
            self.inference_error_count = 0  # Reset error count on success

            # Build detection info and annotate the frame
            for detection in detections:
                class_id = detection['class']
                label = self.class_names.get(str(int(class_id)), "unknown")
                detection_info_list.append({
                    "class_name": label,
                    "confidence": detection['confidence'],
                    "x1": int(detection['x1']),
                    "y1": int(detection['y1']),
                    "x2": int(detection['x2']),
                    "y2": int(detection['y2']),
                })
                x1, y1, x2, y2 = detection['x1'], detection['y1'], detection['x2'], detection['y2']
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        except Exception as e:
            self.inference_error_count += 1
            logger.debug(f"Error during ONNX inference: {e} (Error count: {self.inference_error_count})")
            if self.inference_error_count >= 3:
                logger.error("Persistent inference errors encountered. Consider restarting the application.")
                return frame, []  # Return the original frame
            return frame, []  # Return the original frame

        return annotated_frame, detection_info_list


# ------------------------------------------------------------------------------
# Detector Class (Modularized)
# ------------------------------------------------------------------------------
class Detector:
    def __init__(self, model_choice="yolo", debug=False):  # Changed to yolo for onnx
        """
        Loads the detection model.
        """
        self.debug = debug
        self.model_choice = model_choice.lower()
        if self.model_choice == "yolo":  # Corrected model choice
            self.model = ONNXModel(debug=debug)
        else:
            raise ValueError(f"Unsupported model choice: {self.model_choice}")

    def detect_objects(self, frame, confidence_threshold=0.5, save_threshold=0.8):
        """
        Runs object detection on a frame.
        Returns a tuple: (annotated_frame, object_detected, original_frame, detection_info_list)
        """
        original_frame = frame.copy()
        annotated_frame, detection_info_list = self.model.detect(
            frame, confidence_threshold)
        object_detected = any(
            det["confidence"] >= save_threshold for det in detection_info_list)
        return annotated_frame, object_detected, original_frame, detection_info_list