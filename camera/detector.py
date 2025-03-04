# ------------------------------------------------------------------------------
# Detector Module for Object Detection (Modularized, SSD removed)
# ------------------------------------------------------------------------------
import os
import cv2
import logging
from ultralytics import YOLO
import requests
import hashlib

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ------------------------------------------------------------------------------
# Base Detection Model Interface
# ------------------------------------------------------------------------------
class BaseDetectionModel:
    def detect(self, frame, confidence_threshold, class_filter):
        """
        Perform detection on the provided frame.
        Must return a tuple: (annotated_frame, detection_info_list)
        """
        raise NotImplementedError

# ------------------------------------------------------------------------------
# YOLOv8 Model Wrapper
# ------------------------------------------------------------------------------
class YOLOv8Model(BaseDetectionModel):
    def __init__(self, debug=False):
        """
        Initialize the YOLOv8 model.
        """
        self.debug = debug
        # Use the provided model_path or fall back to the environment variable,
        # defaulting to "models/best.pt" if not set.
        model_path = os.getenv("YOLO8N_MODEL_PATH", "models/best.pt")
        logger.debug(f"Using YOLOv8 model from: {model_path}")

        self.model_path = model_path

        # Download the best model (if needed) before loading.
        self.download_best_model()

        # Load the model.
        self.model = YOLO(self.model_path)
        if hasattr(self.model, 'names'):
            self.names = self.model.names
        else:
            self.names = {}
        logger.info(f"YOLOv8 model loaded from {self.model_path}")

        # Initialize inference error counter
        self.inference_error_count = 0

        # Always perform a dummy inference to warm up the model
        dummy_image = cv2.imread("assets/static_placeholder.jpg")
        if dummy_image is not None:
            try:
                test_results = self.model(dummy_image)
                logger.info(f"Model warm-up successful. Test results: {test_results}")
            except Exception as e:
                logger.error(f"Model warm-up failed: {e}", exc_info=True)
        else:
            logger.warning("Dummy image for model warm-up not found.")

    def download_best_model(self):
        """
        Download the latest best.pt model from GitHub and store it at self.model_path.
        If the file already exists, compare its SHA256 hash to the downloaded file,
        and only overwrite the local file if the hash has changed.
        """
        model_url = "https://raw.githubusercontent.com/arminfabritzek/WatchMyBirds-Train-YOLO/main/best_model/weights/best.pt"
        dest_path = self.model_path
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # Download the file from GitHub.
        response = requests.get(model_url, stream=True)
        if response.status_code != 200:
            logger.error(f"Failed to download model. Status code: {response.status_code}")
            return

        downloaded_data = response.content
        downloaded_hash = hashlib.sha256(downloaded_data).hexdigest()
        logger.info(f"Downloaded model hash: {downloaded_hash}")

        # Check if the file already exists and compute its hash.
        if os.path.exists(dest_path):
            with open(dest_path, "rb") as f:
                local_data = f.read()
            local_hash = hashlib.sha256(local_data).hexdigest()
            logger.info(f"Local model hash: {local_hash}")
        else:
            local_hash = None

        # Overwrite the file only if the hashes differ.
        if local_hash != downloaded_hash:
            with open(dest_path, "wb") as f:
                f.write(downloaded_data)
            logger.info(f"Best model updated at {dest_path}")
        else:
            logger.info("Local model is already up-to-date.")

    def detect(self, frame, confidence_threshold, class_filter):
        detection_info_list = []
        annotated_frame = frame.copy()
        try:
            results = self.model(frame)
            # Reset error counter on successful inference
            self.inference_error_count = 0
            if self.debug:
                logger.debug(f"YOLOv8 raw inference results: {results}")
        except Exception as e:
            self.inference_error_count += 1
            logger.debug(f"Error during YOLOv8 inference: {e} (Error count: {self.inference_error_count})")
            # If persistent errors occur, trigger a reinitialization by raising an exception
            if self.inference_error_count >= 3:
                logger.error("Persistent inference errors encountered. Triggering detector reinitialization.")
                raise Exception("Persistent inference errors encountered, triggering detector reinitialization.") from e
            results = []
        for result in results:
            try:
                boxes = result.boxes.data.cpu().numpy().tolist() if hasattr(result.boxes, "data") else []
                if self.debug:
                    logger.debug(f"Processing YOLOv8 result with boxes: {boxes}")
            except Exception as e:
                logger.debug(f"Error processing YOLOv8 result: {e}")
                boxes = []
            for det in boxes:
                x1, y1, x2, y2, score, class_id = det
                if score >= confidence_threshold:
                    label = self.names[int(class_id)] if self.names and int(class_id) in self.names else "unknown"
                    if class_filter is None or label in class_filter:
                        detection_info_list.append({
                            "class_name": label,
                            "confidence": float(score),
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2)
                        })
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f"{label} ({score:.2f})",
                                    (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return annotated_frame, detection_info_list

# ------------------------------------------------------------------------------
# Detector Class (Modularized)
# ------------------------------------------------------------------------------
class Detector:
    def __init__(self, model_choice="yolo8n", debug=False):
        """
        Loads the detection model.
        Currently, only the 'yolo8n' model is supported.
        """
        self.debug = debug
        self.model_choice = model_choice.lower()
        if self.model_choice == "yolo8n":
            self.model = YOLOv8Model(debug=debug)
        else:
            raise ValueError(f"Unsupported model choice: {self.model_choice}")

    def detect_objects(self, frame, class_filter=None, confidence_threshold=0.5, save_threshold=0.8):
        """
        Runs object detection on a frame.
        Returns a tuple: (annotated_frame, object_detected, original_frame, detection_info_list)
        """
        original_frame = frame.copy()
        annotated_frame, detection_info_list = self.model.detect(frame, confidence_threshold, class_filter)
        object_detected = any(det["confidence"] >= save_threshold for det in detection_info_list)
        return annotated_frame, object_detected, original_frame, detection_info_list

