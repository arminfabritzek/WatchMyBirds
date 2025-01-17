# WatchMyBirds

**WatchMyBirds** is a lightweight and customizable object detection application designed for real-time monitoring using webcams and TensorFlow. The application focuses on identifying objects of interest, such as birds, insects or plants, which are then classified with another model for further analysis, and saving frames where specific conditions are met. It is ideal for hobbyists, researchers, or anyone interested in automated visual monitoring.

---

## Features

- **Real-Time Object Detection**: Perform object detection directly from a webcam feed.


- **Customizable Detection Settings**:
  - Filter detections by specific object classes.
  - Configure confidence thresholds for detections and frame saving.


- **Support for Pre-Trained Models**:
  - Choose between `EfficientDet Lite4` and `SSD MobileNet V2` models.
  - Use TensorFlow or TFLite formats.
  - Tested on Raspberry Pi 4 4GB with approximately 1 FPS.
  - Future adaptation planned for OAK-D cameras.


- **Batch Classification**: Crop detected objects for further classification with a separate model.


- **Expandable Camera Support**: Designed with flexibility for additional camera types (e.g., RTSP and PTZ cameras).


- **Modular Design**: Organized file structure for easy updates and extensions.

---

## Installation

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/arminfabritzek/WatchMyBirds.git
   cd WatchMyBirds
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```





4. **Download and Place Pre-Trained Models**:
   - Download the TensorFlow SavedModel `ssd_mobilenet_v2` for from Kaggle: [ssd_mobilenet_v2](https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v2/tensorFlow2/ssd-mobilenet-v2)
   - Download the `EfficientDet Lite4` TFLite model:
     [EfficientDet Lite4 as TFLite](https://www.kaggle.com/models/tensorflow/efficientdet/tfLite/lite4-detection-default)
   - Download the label files:
     [ImageNetLabels.txt](https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt), [coco_label_map.pbtxt](research/object_detection/data/mscoco_label_map.pbtxt)
   - Place these files in the `models/` directory.

---

## Usage

### Running the Application

To start the object detection livestream, run:
```bash
python main.py
```

To start batch classification, run:
```bash
python batch_classification.py
```

### Output

- **Livestream**: Displays a real-time video feed with bounding boxes around detected objects.
- **Saved Frames**: Frames with objects exceeding the confidence threshold are saved in the `output/` directory.
- **CSV Logging**: All detections are logged to a CSV file (`all_bounding_boxes.csv`) for analysis. Batch classification results are saved in `all_bounding_boxes_classified.csv` for further review.

---

## File Structure

```plaintext
WatchMyBirds/
├── camera/
│   ├── __init__.py
│   ├── base_camera.py
│   ├── webcam_camera.py
├── models/
│   ├── ssd_mobilenet_v2/                                       # TensorFlow SavedModel
│   ├── coco_label_map.pbtxt                                    # COCO label map
│   ├── efficientdet-tflite-lite4-detection-default-v2.tflite   # EfficientDet Lite4 detection model
│   ├── efficientdet_labels.pbtxt                               # EfficientDet labels
│   ├── ImageNetLabels.txt                                      # ImageNet labels
├── output/                                                     # Saved frames and logs
├── requirements.txt                                            # Python dependencies
├── README.md                                                   # Project documentation
├── main.py                                                     # Entry point for the application
├── batch_classification.py                                     # Crops and classifies detected objects
└── .gitignore                                                  # Ignored files/folders
```

---

## Customization

### Adjusting Parameters

You can modify parameters in `main.py` to suit your use case:

- **Class Filter**:
  Specify which object classes to detect:
  ```python
  class_filter = ["bird", "cat", "dog"]  # Detect only these classes
  ```

- **Confidence Threshold**:
  Set the minimum confidence for detections:
  ```python
  confidence_threshold = 0.6  # Default: 60%
  ```

- **Save Threshold**:
  Determine when to save frames:
  ```python
  save_threshold = 0.8  # Default: 80%
  ```

### Adding More Cameras

Future updates will include support for:
- RTSP cameras for network streaming.
- PTZ (Pan-Tilt-Zoom) cameras for enhanced control and flexibility.

---

## Further Development

This project is a work in progress. Upcoming features include:

1. **Custom Bird Detection Models**:
   - Train a specialized bird detection model in a separate repository and integrate it into this application.
   - Integrate AIY models for classification, including birds, insects and plants:
     - `google/aiy/vision/classifier/insects_V1/1` ([google/aiy/vision/classifier/insects_V1/1](https://www.kaggle.com/models/google/aiy/tfLite/vision-classifier-insects-v1))


     

2. **Enhanced Camera Support**:
   - Add support for RTSP and PTZ cameras to monitor more complex setups.


3. **Edge Deployment**:
   - Optimize the application for deployment on edge devices like Raspberry Pi or NVIDIA Jetson Nano.


4. **Advanced Classification**:
   - Expand batch classification capabilities with more accurate models for identifying bird species.


---

## Contributing

Contributions are welcome! If you have ideas or improvements, please open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

