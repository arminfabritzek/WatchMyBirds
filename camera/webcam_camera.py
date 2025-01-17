# ------------------------------------------------------------------------------
# WebcamCamera Module for Object Detection
# ------------------------------------------------------------------------------

import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from camera.base_camera import BaseCamera
import re
import cv2

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def parse_label_map(file_path):
    """
    Parses a TensorFlow label map file to create a mapping of class IDs to names.

    :param file_path: Path to the label map file.
    :return: Dictionary mapping class IDs to display names.
    """
    label_map = {}
    with open(file_path, 'r') as file:
        content = file.read()
        items = re.findall(r'item\s+{(.*?)}', content, re.DOTALL)
        for item in items:
            id_match = re.search(r'id:\s+(\d+)', item)
            name_match = re.search(r'display_name:\s+"(.+?)"', item)
            if id_match and name_match:
                class_id = int(id_match.group(1))
                display_name = name_match.group(1)
                label_map[class_id] = display_name
    return label_map

def download_and_save_model(model_url, save_path):
    """
    Downloads a TensorFlow Hub model and saves it locally if it doesn't already exist.

    :param model_url: URL of the TensorFlow Hub model.
    :param save_path: Local directory path to save the model.
    """
    if not os.path.exists(save_path):
        print(f"Model not found at {save_path}. Downloading...")
        model = hub.load(model_url)
        tf.saved_model.save(model, save_path)
        print(f"Model saved at {save_path}")
    else:
        print(f"Model already exists at {save_path}")


# ------------------------------------------------------------------------------
# WebcamCamera Class
# ------------------------------------------------------------------------------

class WebcamCamera(BaseCamera):
    """
    Webcam-based video capture class with integrated object detection capabilities.
    """
    def __init__(self, source=2, model_choice="ssd_mobilenet_v2", label_map_path="models/coco_label_map.pbtxt"):
        """
        Initializes the webcam and loads the selected object detection model.

        :param source: Camera source index (e.g., 0 for the default webcam).
        :param model_choice: Model choice for object detection ('efficientdet_lite4' or 'ssd_mobilenet_v2').
        :param label_map_path: Path to the label map file.
        """
        super().__init__(source=source)
        self.model_choice = model_choice.lower()

        print(f"Loading model: {self.model_choice}...")

        if self.model_choice == "efficientdet_lite4":
            # Load TFLite model
            model_path = "models/efficientdet_lite4.tflite"
            model_path = "models/efficientdet-tflite-lite4-detection-default-v2.tflite"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"TFLite model not found at {model_path}")

            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()

            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            print("TFLite model loaded successfully!")
            print(f"Input details: {self.input_details}")
            print(f"Output details: {self.output_details}")

        elif self.model_choice == "ssd_mobilenet_v2":
            # Load TF Hub model
            model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
            self.model = hub.load(model_url).signatures['serving_default']
            print("TF Hub model loaded successfully!")
        else:
            raise ValueError(f"Unsupported model choice: {self.model_choice}")

        # Load label map
        self.labels = parse_label_map(label_map_path)
        print("Label map loaded successfully!")

    def detect_objects(self, frame, class_filter=None, confidence_threshold=0.5, save_threshold=0.8):
        """
        Detects objects in a given frame and annotates them with bounding boxes.

        :param frame: Input frame for object detection (numpy array).
        :param class_filter: Optional list of class names to filter detections.
        :param confidence_threshold: Minimum confidence for detections to be considered valid.
        :param save_threshold: Confidence level at which the frame should be marked for saving.
        :return: Annotated frame, save flag, original frame, and detection details.
        """
        # Make a copy BEFORE we annotate
        original_frame = frame.copy()

        if self.model_choice == "efficientdet_lite4":
            # Preprocess for TFLite model
            input_size = self.input_details[0]['shape'][1:3]
            resized_frame = cv2.resize(frame, tuple(input_size))
            input_data = np.expand_dims(resized_frame, axis=0).astype(np.uint8)

            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()

            # 0 => boxes, 1 => classes, 2 => scores, 3 => num_detections
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0].astype(int)
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]

        else:  # SSD MobileNet
            input_tensor = tf.convert_to_tensor([frame], dtype=tf.uint8)
            detections = self.model(input_tensor)

            # Extract detection results
            boxes = detections['detection_boxes'].numpy()[0]
            classes = detections['detection_classes'].numpy()[0].astype(int)
            scores = detections['detection_scores'].numpy()[0]

        height, width, _ = frame.shape
        should_save_interval = False
        detection_info_list = []  # <-- We'll store bounding-box data here

        for i, score in enumerate(scores):
            if score >= confidence_threshold:

                # SHIFT for EfficientDet if necessary
                if self.model_choice == "efficientdet_lite4":
                    model_class_id = classes[i]
                    label_map_id = model_class_id + 1
                    class_name = self.labels.get(label_map_id, f"Unknown ({label_map_id})")
                else:
                    class_name = self.labels.get(classes[i], f"Unknown ({classes[i]})")

                # Filter by class
                if class_filter and class_name not in class_filter:
                    continue

                # Mark frame for saving if any score >= save_threshold
                if score >= save_threshold:
                    should_save_interval = True

                # Convert normalized box coordinates to absolute pixel values
                box = boxes[i]
                y1, x1, y2, x2 = (
                    int(box[0] * height),
                    int(box[1] * width),
                    int(box[2] * height),
                    int(box[3] * width)
                )
                # Ensure box coords are within image boundaries
                y1, x1 = max(0, y1), max(0, x1)
                y2, x2 = min(height, y2), min(width, x2)

                print(f"Detected: {class_name}, Probability: {score:.2f}, Box: [{x1}, {y1}, {x2}, {y2}]")

                # Draw bounding box + label on `frame`
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{class_name} ({score:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

                # Store bounding-box info in a dictionary
                detection_info_list.append({
                    "class_name": class_name,
                    "confidence": float(score),
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2
                })

        if not scores.any():
            print("No detections made.")
        elif not should_save_interval:
            print("Detections made, but none exceed save threshold.")

        # Return annotated/unannotated frames, plus detection data
        return frame, should_save_interval, original_frame, detection_info_list