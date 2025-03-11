# ------------------------------------------------------------------------------
# Detector Module for Object Detection (Modularized, ONNX Runtime)
# camera/detector.py
# ------------------------------------------------------------------------------
from config import load_config
config = load_config()
import os
import cv2
import logging
import numpy as np
import onnxruntime
import requests
import hashlib

import yaml
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
            logger.info(f"ONNX model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}", exc_info=True)
            raise  # Re-raise the exception to halt execution

        self.input_size = (640, 640)  # Assuming 640x640 input.  Get from model metadata if possible.
        # Load class names from labels.yaml in the same folder as the model.
        labels_path = os.path.join(os.path.dirname(self.model_path), "labels.yaml")
        if not os.path.exists(labels_path):
            self.download_latest_labels()

        if os.path.exists(labels_path):
            try:
                with open(labels_path, "r") as f:
                    label_data = yaml.safe_load(f)
                names = label_data.get("names")
                if isinstance(names, dict):
                    # Convert dict to a list sorted by key (assuming keys are integers or strings of integers)
                    sorted_items = sorted(names.items(), key=lambda item: int(item[0]))
                    self.class_names = [name for key, name in sorted_items]
                elif isinstance(names, list):
                    self.class_names = names
                else:
                    logger.warning("Unexpected format in labels.yaml; using default class name.")
                    self.class_names = ["bird"]
            except Exception as e:
                logger.error(f"Error loading labels from {labels_path}: {e}", exc_info=True)
                self.class_names = ["bird"]
        else:
            logger.warning(f"Label file not found at {labels_path}. Using default class name.")
            self.class_names = ["bird"]

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
        Download the latest labels.yaml file from GitHub and store it next to the model.
        """
        labels_url = "https://raw.githubusercontent.com/arminfabritzek/WatchMyBirds-Train-YOLO/main/best_model/weights/labels.yaml"
        labels_dest_path = os.path.join(os.path.dirname(self.model_path), "labels.yaml")
        os.makedirs(os.path.dirname(labels_dest_path), exist_ok=True)

        response = requests.get(labels_url, stream=True)
        if response.status_code != 200:
            logger.error(f"Failed to download labels.yaml. Status code: {response.status_code}")
            return

        downloaded_data = response.content
        downloaded_hash = hashlib.sha256(downloaded_data).hexdigest()
        logger.info(f"Downloaded labels.yaml hash: {downloaded_hash}")

        if os.path.exists(labels_dest_path):
            with open(labels_dest_path, "rb") as f:
                local_data = f.read()
            local_hash = hashlib.sha256(local_data).hexdigest()
            logger.info(f"Local labels.yaml hash: {local_hash}")
        else:
            local_hash = None

        if local_hash != downloaded_hash:
            with open(labels_dest_path, "wb") as f:
                f.write(downloaded_data)
            logger.info(f"labels.yaml updated at {labels_dest_path}")
        else:
            logger.info("Local labels.yaml is already up-to-date.")

    def preprocess_image(self, img):
        """Preprocesses an image for YOLOv8 inference (ONNX version)."""
        original_image = img.copy()
        h, w = img.shape[:2]
        input_width, input_height = self.input_size

        # Calculate scaling and padding
        ratio = min(input_width / w, input_height / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        dw = (input_width - new_w) / 2  # width padding
        dh = (input_height - new_h) / 2  # height padding

        # Resize and pad the image
        resized_img = cv2.resize(
            img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        # gray padding
        img = cv2.copyMakeBorder(
            resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Convert to RGB, normalize, and transpose for ONNX Runtime
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        scale_factors = (w / new_w, h / new_h, dw, dh)
        return img, original_image, scale_factors

    def postprocess_output(self, output, scale_factors, original_image_shape, confidence_threshold):
        """Postprocesses the YOLOv8 ONNX output."""
        predictions = np.squeeze(output[0]).T

        scores = predictions[:, 4]
        mask = scores >= confidence_threshold
        predictions = predictions[mask]
        scores = scores[mask]

        if predictions.size == 0:
            return []

        boxes = predictions[:, :4]
        boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        if predictions.shape[1] > 5:
            class_confidences = predictions[:, 5:]
            class_ids = np.argmax(class_confidences, axis=1)
            confidences = scores * np.max(class_confidences, axis=1)
        else:
            class_ids = np.zeros(len(predictions), dtype=np.int32)
            confidences = scores

        detections = []
        sorted_indices = np.argsort(confidences)[::-1]
        boxes = boxes[sorted_indices]
        confidences = confidences[sorted_indices]
        class_ids = class_ids[sorted_indices]

        while boxes.size > 0:
            best_box = boxes[0]
            detections.append(
                {'box': best_box.tolist(), 'confidence': float(confidences[0]), 'class_id': int(class_ids[0])})

            if len(boxes) == 1:
                break

            x1 = np.maximum(best_box[0], boxes[1:, 0])
            y1 = np.maximum(best_box[1], boxes[1:, 1])
            x2 = np.minimum(best_box[2], boxes[1:, 2])
            y2 = np.minimum(best_box[3], boxes[1:, 3])

            intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            box1_area = (best_box[2] - best_box[0]) * (best_box[3] - best_box[1])
            box2_area = (boxes[1:, 2] - boxes[1:, 0]) * \
                (boxes[1:, 3] - boxes[1:, 1])
            union_area = box1_area + box2_area - intersection_area
            iou = intersection_area / (union_area + 1e-9)

            keep_indices = iou <= 0.45 # self.iou_threshold  # Use a consistent IoU threshold
            boxes = boxes[1:][keep_indices]
            confidences = confidences[1:][keep_indices]
            class_ids = class_ids[1:][keep_indices]

        width_scale, height_scale, width_pad, height_pad = scale_factors
        for detection in detections:
            box = detection['box']
            box[0] = int((box[0] - width_pad) * width_scale)
            box[1] = int((box[1] - height_pad) * height_scale)
            box[2] = int((box[2] - width_pad) * width_scale)
            box[3] = int((box[3] - height_pad) * height_scale)
            box[0] = max(0, min(box[0], original_image_shape[1]))
            box[1] = max(0, min(box[1], original_image_shape[0]))
            box[2] = max(0, min(box[2], original_image_shape[1]))
            box[3] = max(0, min(box[3], original_image_shape[0]))
        return detections

    def detect(self, frame, confidence_threshold):
        """
        Performs object detection on a single frame using ONNX Runtime.
        """
        detection_info_list = []
        annotated_frame = frame.copy()

        try:
            # Preprocess
            processed_image, original_image, scale_factors = self.preprocess_image(
                frame)

            # Run inference
            inputs = {self.session.get_inputs()[0]
                .name: processed_image}  # Correct input name
            outputs = self.session.run(None, inputs)

            # Postprocess
            detections = self.postprocess_output(
                outputs, scale_factors, original_image.shape[:2], confidence_threshold)

            self.inference_error_count = 0  # Reset on success

            # Filter and build detection info
            for detection in detections:
                class_id = detection['class_id']
                label = self.class_names[class_id] if class_id < len(
                    self.class_names) else "unknown"
                detection_info_list.append({
                    "class_name": label,
                    "confidence": detection['confidence'],
                    "x1": int(detection['box'][0]),
                    "y1": int(detection['box'][1]),
                    "x2": int(detection['box'][2]),
                    "y2": int(detection['box'][3]),
                })

                # Draw bounding box
                x1, y1, x2, y2 = detection['box']
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Prepare the annotation text
                annotation_text = f"{label} ({detection['confidence']:.2f})"

                # Convert the OpenCV image (BGR) to a PIL image (RGB)
                annotated_pil = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(annotated_pil)

                # Load the TrueType font from the assets folder.
                # If the font file is not found, it falls back to the default PIL font.
                font_size = 24  # Adjust as needed.
                try:
                    font = ImageFont.truetype("assets/WRP_cruft.ttf", font_size)
                except IOError:
                    font = ImageFont.load_default()

                # Calculate the text dimensions using the font's getbbox method.
                bbox = font.getbbox(annotation_text)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # Add some padding around the text.
                padding = 1

                # The rectangle height should accommodate the text height plus the top/bottom padding.
                box_height = text_height + 2 * padding

                # Draw a filled white rectangle directly under the bounding box.
                draw.rectangle([(x1, y2), (x2, y2 + box_height)], fill="white")

                # Adjust text_y to account for bbox[1] ---
                text_x = x1 + padding
                text_y = y2 + padding - bbox[1]  # Subtract bbox[1]

                # Draw the annotation text in black inside the white rectangle.
                draw.text((text_x, text_y), annotation_text, font=font, fill="black")

                # Convert the PIL image back to a NumPy array in BGR format.
                annotated_frame = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)



        except Exception as e:
            self.inference_error_count += 1
            logger.debug(
                f"Error during ONNX inference: {e} (Error count: {self.inference_error_count})")
            if self.inference_error_count >= 3:
                logger.error(
                    "Persistent inference errors encountered.  Consider restarting the application.")
                # Instead of raising, return empty results to allow the caller to handle gracefully.
                return frame, []
            return frame, []  # Return original frame and empty list

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