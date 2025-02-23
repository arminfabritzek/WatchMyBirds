# ------------------------------------------------------------------------------
# Detector Module for Object Detection (Modularized, SSD removed)
# ------------------------------------------------------------------------------
import os
import cv2
from camera.video_capture import VideoCapture  # Import VideoCapture class
import logging
from ultralytics import YOLO
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
    def __init__(self, model_path, debug=False):
        self.debug = debug
        self.model = YOLO(model_path)
        if hasattr(self.model, 'names'):
            self.names = self.model.names
        else:
            self.names = {}
        logger.info(f"YOLOv8 model loaded from {model_path}")

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
    def __init__(self, source=2, model_choice="yolo8n", debug=False):
        """
        Initializes the VideoCapture and loads the detection model.
        Currently, only the 'yolo8n' model is supported.
        """
        self.debug = debug
        logger.debug("Initializing VideoCapture for Detector")
        self.video_capture = VideoCapture(source, debug=debug)
        self.source = source
        self.model_choice = model_choice.lower()
        if self.model_choice == "yolo8n":
            model_path = os.getenv("YOLO8N_MODEL_PATH", "models/yolov8n.pt")
            logger.debug(f"Using YOLOv8 model from: {model_path}")
            self.model = YOLOv8Model(model_path, debug=debug)
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

    def get_frame(self):
        frame = self.video_capture.get_frame()
        if frame is None and self.debug:
            logger.debug("No frame available from VideoCapture.")
        return frame

    def release(self):
        self.video_capture.release()

    @property
    def resolution(self):
        return self.video_capture.resolution