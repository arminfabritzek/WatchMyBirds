# WatchMyBirds

**WatchMyBirds** is a lightweight and customizable object detection application designed for real-time monitoring using webcams and TensorFlow. The application focuses on identifying objects of interest, such as birds, insects or plants, which are then classified with another model for further analysis, and saving frames where specific conditions are met. It is ideal for hobbyists, researchers, or anyone interested in automated visual monitoring.

---

[![Build and Push Docker Image](https://github.com/arminfabritzek/WatchMyBirds/actions/workflows/docker.yml/badge.svg)](https://github.com/arminfabritzek/WatchMyBirds/actions/workflows/docker.yml)

---

## Features

- **Real-Time Object Detection**: Perform object detection directly from a webcam or RTSP camera streams.


- **Future Plans**:
  - Integrate PTZ (Pan-Tilt-Zoom) camera control to track objects and keep them centered.
  - Support Raspberry Pi 4 (4GB) for ~1 FPS.
  - Compatibility with OAK-D stereo cameras.


- **Customizable Detection Settings**:
  - Filter detections by specific object classes.
  - Configure confidence thresholds for detections and frame saving.


- **Support for Pre-Trained Models from PyTorch**:
  - Includes `ssd300_vgg16` and `EfficientDet Lite4` and `SSD MobileNet V2`.
  - Supports PyTorch (TensorFlow, and TFLite formats coming soon).



- **Batch Classification**: Crop detected objects for further classification with a separate model.


- **Modular Design**: Organized file structure for easy updates and extensions.

---

## Installation


## Docker Usage

To run **WatchMyBirds** in a Docker container with default settings:

1. **Pull the Docker Image**:
   ```bash
   docker pull starminworks/watchmybirds:latest
   ```

2. **Create a `docker-compose.yml` file** with the following content:

   ```yaml
   version: '3'
   services:
     watchmybirds:
       image: starminworks/watchmybirds:latest
       container_name: watchmybirds
       environment:
         - VIDEO_SOURCE=rtsp://user:password@192.168.0.2:554/1  # Replace with your RTSP stream
         - PUID=1000  # Replace with your user ID
         - PGID=1000   # Replace with your group ID
         - TZ=Europe/Berlin  # Set your timezone
         - DEBUG_MODE=False  
         - MODEL_CHOICE="pytorch_ssd"  
         - CLASS_FILTER="'["bird"]'"  
         - CONFIDENCE_THRESHOLD=0.5  
         - SAVE_THRESHOLD=0.5  
         - SAVE_INTERVAL=1  # Seconds between saving
         - INPUT_FPS=10  # Default FPS is 10
         - PROCESS_TIME=1  # Average detection time per frame
         - STREAM_WIDTH=640  # Width of the output stream. Default width: 640
         - STREAM_HEIGHT=360  # Height of the output stream. Default height: 360
       volumes:
         - /your_path/output:/output  # Path for saving output
       ports:
         - "5001:5001"  # HTTP port for Flask app
         - "8554:8554"  # RTSP port
         - "8081:8081"  # MJPEG port
         - "1936:1936"  # Custom port
         - "8889:8889"  # Custom port
         - "8189:8189/udp"  # UDP port
       restart: unless-stopped
   ```

3. **Start the container** using Docker Compose:

   ```bash
   docker-compose up -d
   ```

   This will run the **WatchMyBirds** application, and you can access the livestream at `http://<your-server-ip>:5001`.

---

## Clone Repository

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


4. **Configure the Video Source**:
- Create or edit the .env file in the project root
- Add the VIDEO_SOURCE variable:

   ```plaintext
   VIDEO_SOURCE=0
   ```
   For an RTSP stream, use:
   ```plaintext
   VIDEO_SOURCE=rtsp://user:password@192.168.0.2:554/1
   ```


**Notes**:
- Planing to switch entirely to PyTorch!
- The .env file allows dynamic switching between a webcam (e.g. VIDEO_SOURCE=2) and an RTSP stream for greater flexibility.



---

## Usage

### Running the Application

To start the object detection livestream, run:
```bash
python main.py
```

To start batch classification, run (on hold):
```bash
python batch_classification.py
```

### Output

- **Livestream**: Displays a real-time video feed with bounding boxes around detected objects.
- **Saved Frames**: Frames with objects exceeding the confidence threshold are saved in the `output/` directory.
- **CSV Logging**: All detections are logged to a CSV file (`all_bounding_boxes.csv`) for analysis. Batch classification results are saved in `all_bounding_boxes_classified.csv` for further review.

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

- **Use Threading**:
  Determine when to save frames:
  ```python
  use_threaded = True  # Set to False to use non-threaded video capture
  ```

### Adding More Cameras

Future updates will include support for:
- RTSP cameras for network streaming.
- PTZ (Pan-Tilt-Zoom) cameras for enhanced control and flexibility.

---

## Further Development

This project is a work in progress. Upcoming features include:

1. **Custom Models**:
   - Train specialized bird detection models and integrate AIY vision classifiers for birds, insects, and plants.


2. **Enhanced Camera Support**:
   - Add support for RTSP and PTZ cameras to monitor more complex setups.


3. **Edge Deployment**:
   - Optimize for edge devices like Raspberry Pi or NVIDIA Jetson Nano.


4. **Advanced Classification**:
   - Expand batch classification capabilities with more accurate models for identifying bird species.


---

## Contributing

Contributions are welcome! If you have ideas or improvements, please open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

