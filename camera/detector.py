# ------------------------------------------------------------------------------
# Detector Module for Object Detection
# ------------------------------------------------------------------------------

import torch
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection import SSD300_VGG16_Weights
import cv2
from PIL import Image
from camera.video_capture import VideoCapture  # Import VideoCapture class

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def preprocess_pytorch_frame(frame):
    """Preprocesses an image frame for PyTorch."""
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor()  # Convert to PyTorch tensor
    ])
    # Convert BGR (OpenCV) to RGB (Pillow)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to PIL image
    image = Image.fromarray(frame_rgb)
    # Apply the transformation
    return transform(image).unsqueeze(0)  # Add batch dimension
# ------------------------------------------------------------------------------
# Detector Class
# ------------------------------------------------------------------------------

class Detector():
    """
    Webcam-based video capture class with integrated object detection capabilities.
    """
    def __init__(self, source=2, model_choice="pytorch_ssd"):
        """
        Initializes the webcam and loads the selected object detection model.

        :param source: Camera source index (e.g., 0 for the default webcam; on my MacBook my internal webcam is 2).
        :param model_choice: Model choice for object detection ('efficientdet_lite4' or 'ssd_mobilenet_v2' ...).
        """

        print("Using threaded VideoCapture for minimal latency.")
        self.video_capture = VideoCapture(source=source)
        self.model_choice = model_choice.lower()

        if self.model_choice == "pytorch_ssd":
            # Load the SSD300 VGG16 model pre-trained on the COCO dataset using updated weights parameter
            self.model = ssd300_vgg16(weights=SSD300_VGG16_Weights.COCO_V1)
            self.model.eval()

            self.coco_classes = [
                '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
                'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
                'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
                'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
                'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]

            print(f"{self.model_choice.capitalize()} model loaded successfully.")
        else:
            raise ValueError(f"Unsupported model choice: {self.model_choice}")

    def detect_objects(self, frame, class_filter=None, confidence_threshold=0.5, save_threshold=0.8):
        original_frame = frame.copy()

        if self.model_choice == "pytorch_ssd":
            input_tensor = preprocess_pytorch_frame(frame)  # Transform frame for the model
            with torch.no_grad():
                outputs = self.model(input_tensor)  # Run inference

            # Extract boxes, labels, and scores from the outputs
            boxes = outputs[0]['boxes']
            labels = outputs[0]['labels']
            scores = outputs[0]['scores']

            detection_info_list = []
            for i, score in enumerate(scores):
                if score >= confidence_threshold:
                    # Check if the detection is a 'bird' (you can adjust this to filter for other classes)
                    if self.coco_classes[labels[i].item()] == "bird":
                        x1, y1, x2, y2 = boxes[i].tolist()
                        detection_info_list.append({
                            "class_name": "bird",  # Hardcoded as 'bird', change if you want to classify other objects
                            "confidence": float(score),
                            "x1": int(x1), "y1": int(y1),
                            "x2": int(x2), "y2": int(y2)
                        })

                        # Annotate frame with bounding box and label
                        frame = frame.copy()  # Ensure that frame is writable before modifying it
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"bird ({score:.2f})",  # Hardcoded as 'bird', update if using other objects
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2
                        )

            # Determine if frame should be saved based on detection results
            should_save_interval = bool(detection_info_list)

            return frame, should_save_interval, original_frame, detection_info_list

        # If no object detection model is chosen
        return frame, False, original_frame, []

    def get_frame(self):
        """
        Retrieves the latest frame from the VideoCapture object.
        """
        return self.video_capture.get_frame()



    def release(self):
        """
        Releases resources used by VideoCapture.
        """
        self.video_capture.release()