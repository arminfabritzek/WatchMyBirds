# ------------------------------------------------------------------------------
# Detector Module for Object Detection (Modularized, ONNX Runtime)
# detectors/detector.py
# ------------------------------------------------------------------------------
import json
import os

import cv2
import numpy as np
import onnxruntime

from config import get_config
from logging_config import get_logger
from utils.model_downloader import ensure_model_files, load_latest_identifier

config = get_config()
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
        ident = load_latest_identifier(model_dir)
        self.model_id = ident if ident else os.path.basename(self.model_path)

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
                with open(self.labels_path) as f:
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
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dummy_path = os.path.join(base_dir, "assets", "static_placeholder.jpg")

        # Check if file exists before trying to load (for better error msg)
        if not os.path.exists(dummy_path):
            logger.warning(f"Warm-up image not found at expected path: {dummy_path}")
        else:
            # Try to load
            dummy_image = cv2.imread(dummy_path)
            if dummy_image is not None:
                try:
                    self.detect(dummy_image, 0.5)  # Warm-up call
                    logger.info(f"Model warm-up successful using {dummy_path}")
                except Exception as e:
                    logger.error(f"Model warm-up failed: {e}", exc_info=True)
            else:
                logger.warning(
                    f"cv2.imread failed to load image at {dummy_path} (File exists, size: {os.path.getsize(dummy_path)} bytes)"
                )

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
                return []  # Return empty list on persistent error
            return []  # Return empty list on error

        return detection_info_list

    def exhaustive_detect(self, frame):
        """
        Performs an exhaustive detection using tiling and low thresholds.
        Returns a list of all detections mapped back to the original frame.
        """
        logger.info("Starting exhaustive deep scan (Full + Tiled 0.1)...")
        all_detections = []
        low_conf = 0.1

        # 1. Full frame scan with low confidence
        full_dets = self.detect(frame, low_conf)
        for d in full_dets:
            d["method"] = "full"
        all_detections.extend(full_dets)

        # 2. Tiling (2x2 with overlap)
        h, w = frame.shape[:2]
        # Define 4 tiles with overlap
        # overlap ~20%
        mid_x = w // 2
        mid_y = h // 2

        # Coordinates: x1, y1, x2, y2
        tiles = [
            (0, 0, mid_x + 100, mid_y + 100),  # TL
            (mid_x - 100, 0, w, mid_y + 100),  # TR
            (0, mid_y - 100, mid_x + 100, h),  # BL
            (mid_x - 100, mid_y - 100, w, h),  # BR
        ]

        for tx1, ty1, tx2, ty2 in tiles:
            # Clip to image bounds
            tx1, ty1 = max(0, tx1), max(0, ty1)
            tx2, ty2 = min(w, tx2), min(h, ty2)

            tile_img = frame[ty1:ty2, tx1:tx2]
            if tile_img.size == 0:
                continue

            tile_dets = self.detect(tile_img, low_conf)

            # Map back to original coordinates
            for d in tile_dets:
                d["x1"] += tx1
                d["y1"] += ty1
                d["x2"] += tx1
                d["y2"] += ty1
                d["method"] = "tiled"
                all_detections.append(d)

        # 3. Simple NMS (Non-Maximum Suppression) to remove duplicates
        # We prefer 'tiled' detections if they have higher confidence, but 'full' gives better context.
        # Simple approach: sort by confidence, check IoU.

        keep = []
        all_detections.sort(key=lambda x: x["confidence"], reverse=True)

        for current in all_detections:
            is_new = True
            cx1, cy1, cx2, cy2 = (
                current["x1"],
                current["y1"],
                current["x2"],
                current["y2"],
            )
            current_area = (cx2 - cx1) * (cy2 - cy1)

            for kept in keep:
                kx1, ky1, kx2, ky2 = kept["x1"], kept["y1"], kept["x2"], kept["y2"]

                # Intersection
                ix1 = max(cx1, kx1)
                iy1 = max(cy1, ky1)
                ix2 = min(cx2, kx2)
                iy2 = min(cy2, ky2)

                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    kept_area = (kx2 - kx1) * (ky2 - ky1)
                    union_area = current_area + kept_area - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0

                    if iou > 0.5:  # 50% overlap considered same object
                        is_new = False
                        break

            if is_new:
                keep.append(current)

        logger.info(
            f"Exhaustive scan complete. Found {len(keep)} objects (merged from {len(all_detections)})."
        )
        return keep


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
            self.model_id = getattr(self.model, "model_id", "")
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

    def exhaustive_detect(self, frame):
        """Delegates exhaustive detection to the model."""
        return self.model.exhaustive_detect(frame)
