# ------------------------------------------------------------------------------
# Detector Module for Object Detection (Modularized, ONNX Runtime)
# detectors/detector.py
# ------------------------------------------------------------------------------
from config import load_config
from logging_config import get_logger
from utils.model_downloader import ensure_model_files
import os
import cv2
import numpy as np
import onnxruntime
import json
from PIL import Image, ImageDraw, ImageFont

config = load_config()
logger = get_logger(__name__)

HF_BASE_URL = "https://huggingface.co/arminfabritzek/WatchMyBirds-Models/resolve/main/object_detection"


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
class ONNXDetectionModel(BaseDetectionModel):
    def __init__(self, debug=False):
        """
        Initialize the ONNX Runtime model.
        """
        self.debug = debug
        model_dir = os.path.join(config["MODEL_BASE_PATH"], "object_detection")
        self.model_path, self.labels_path = ensure_model_files(
            HF_BASE_URL, model_dir, "weights_path_onnx", "labels_path"
        )

        # Initialize ONNX Runtime session.  Handles potential errors.
        try:
            self.session = onnxruntime.InferenceSession(
                self.model_path, providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            logger.info(
                f"ONNX model loaded from {self.model_path} using CPUExecutionProvider"
            )
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}", exc_info=True)
            raise

        self.input_size = ONNXDetectionModel.get_model_input_size(self.session)

        # Now load the class names from the JSON file.
        if os.path.exists(self.labels_path):
            try:
                with open(self.labels_path, "r") as f:
                    self.class_names = json.load(f)
            except Exception as e:
                logger.error(
                    f"Error loading labels from {self.labels_path}: {e}", exc_info=True
                )
                self.class_names = {}
        else:
            logger.warning(
                f"Label JSON file not found at {self.labels_path}. Using default class name."
            )
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


    def preprocess_image(self, img):
        original_image = img.copy()
        h, w = img.shape[:2]
        input_height, input_width = self.input_size
        ratio = min(input_width / w, input_height / h)
        resized_width = int(w * ratio)
        resized_height = int(h * ratio)

        # Resize the image
        resized_image = cv2.resize(
            original_image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

        # Create a padded image with fill value 114
        padded_image = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
        dw = (input_width - resized_width) // 2
        dh = (input_height - resized_height) // 2
        padded_image[dh : dh + resized_height, dw : dw + resized_width, :] = (
            resized_image
        )

        # Convert from BGR to RGB, normalize and change shape from HWC to CHW
        image_data = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
        image_data = image_data.transpose((2, 0, 1)).astype(np.float32) / 255.0
        image_data = np.expand_dims(image_data, axis=0)
        return image_data, original_image, ratio, dw, dh

    def postprocess_output(
        self,
        output,
        ratio,
        dw,
        dh,
        original_width,
        original_height,
        conf_threshold=0.25,
    ):
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

            detections.append(
                {
                    "class_name": self.class_names.get(str(int(cls)), str(int(cls))),
                    "confidence": float(conf),
                    "x1": x1_val,
                    "y1": y1_val,
                    "x2": x2_val,
                    "y2": y2_val,
                    "class": int(cls),
                }
            )
        return detections

    def detect(self, frame, confidence_threshold):
        detection_info_list = []
        try:
            # Preprocess the frame and retrieve ratio and padding info
            processed_image, original_image, ratio, dw, dh = self.preprocess_image(
                frame
            )
            original_width, original_height = (
                original_image.shape[1],
                original_image.shape[0],
            )

            # Run inference using ONNX Runtime
            inputs = {self.input_name: processed_image}
            outputs = self.session.run(None, inputs)

            # Postprocess to get final detections
            detections = self.postprocess_output(
                outputs,
                ratio,
                dw,
                dh,
                original_width,
                original_height,
                confidence_threshold,
            )
            self.inference_error_count = 0  # Reset error count on success

            # Build detection info and annotate the frame
            for detection in detections:
                class_id = detection["class"]
                label = self.class_names.get(str(int(class_id)), "unknown")
                detection_info_list.append(
                    {
                        "class_name": label,
                        "confidence": detection["confidence"],
                        "x1": int(detection["x1"]),
                        "y1": int(detection["y1"]),
                        "x2": int(detection["x2"]),
                        "y2": int(detection["y2"]),
                    }
                )

        except Exception as e:
            self.inference_error_count += 1
            logger.debug(
                f"Error during ONNX inference: {e} (Error count: {self.inference_error_count})"
            )
            if self.inference_error_count >= 3:
                logger.error(
                    "Persistent inference errors encountered. Consider restarting the application."
                )
                return frame, []  # Return the original frame
            return frame, []  # Return the original frame

        return detection_info_list


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
            self.model = ONNXDetectionModel(debug=debug)
        else:
            raise ValueError(f"Unsupported model choice: {self.model_choice}")

    def detect_objects(self, frame, confidence_threshold=0.5, save_threshold=0.8):
        """
        Runs object detection on a frame.
        Returns a tuple: (object_detected, original_frame, detection_info_list)
        """
        original_frame = frame.copy()
        detection_info_list = self.model.detect(frame, confidence_threshold)
        object_detected = any(
            det["confidence"] >= save_threshold for det in detection_info_list
        )
        return object_detected, original_frame, detection_info_list
