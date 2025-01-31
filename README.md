# WatchMyBirds


![WatchMyBirds in Action](assets/birds_1280.gif)
*Live object detection capturing birds in real-time!*

---

**WatchMyBirds** is a lightweight and customizable object detection application designed for real-time monitoring using webcams and TensorFlow. The application focuses on identifying objects of interest, such as birds, insects or plants, which are then classified with another model for further analysis, and saving frames where specific conditions are met. It is ideal for hobbyists, researchers, or anyone interested in automated visual monitoring.

---

[![Build and Push Docker Image](https://github.com/arminfabritzek/WatchMyBirds/actions/workflows/docker.yml/badge.svg)](https://github.com/arminfabritzek/WatchMyBirds/actions/workflows/docker.yml)

---

## 🚀 Key Features

- **Real-Time Object Detection**:  
  Transform any webcam or RTSP camera stream into a powerful detection system.  
  - ✅ **Seamless integration with MotionEye OS** for network camera support.  
  - ✅ **Tested with Chronics PLZ Camera** for robust performance.  


- **Optimized for Diverse Hardware**:  
  Built to run across various devices with performance in mind.  
  - ⚡ **Runs efficiently on Synology NAS 923+ (Docker)** → Achieves ~0.3 FPS  
  - ⚡ **MacBook Air M1** → Delivers ~1.3 FPS  
  - ⚡ **Planned Support: Raspberry Pi 4 (4GB)** targeting ~1 FPS  


- **Next-Gen Detection & Classification**:  
  - 🎯 Filter detections by specific object classes.  
  - 🎯 Fine-tune confidence thresholds for both detections and frame saving.  


- **State-of-the-Art AI Models**:  
  Powered by **PyTorch** with support for cutting-edge detection networks.  
  - 🔍 `ssd300_vgg16`, `EfficientDet Lite4`, `SSD MobileNet V2` pre-trained models.  
  - 🔍 Future support for **TensorFlow & TFLite models**.  

---

## 🌟 Future Roadmap  

### 🚀 **Camera & Tracking Enhancements**

- 🏆 **PTZ (Pan-Tilt-Zoom) Camera Control** → Auto-tracking & framing of detected objects.  
- 🏆 **RTSP & Network Camera Expansion** → Full support for RTSP streams and **MotionEye OS integration**.  
- 🏆 **Enhanced Stereo Vision** → **Full compatibility with OAK-D stereo cameras** for depth-based tracking.  

### 🛠️ **AI & Model Optimization**  
- 🏆 **Custom AI Models** → Train specialized models for birds, insects, and plants with **AIY vision classifiers**.  
- 🏆 **Advanced Object Classification** → More precise species identification using **batch classification**.  

### ⚡ **Performance & Edge Deployment**  
- 🏆 **Optimized for Edge Devices** → Efficient performance tuning for **Raspberry Pi 4 & 5** and **NVIDIA Jetson Nano**.  
- 🏆 **Low-Latency Processing** → Further optimizations for improved real-time detection with reduced computational overhead.  

### 📊 **Expanded Data & Logging Capabilities**  
- 🏆 **Comprehensive CSV & JSON Logging** → Improved tracking of detections for data-driven insights.  
- 🏆 **Adaptive Filtering** → Dynamic threshold adjustments for smarter detection.  

---

### **Why These Upgrades Matter** 🚀  
These upcoming improvements will **enhance accuracy, expand hardware compatibility, and refine detection capabilities**—making **WatchMyBirds** an even more powerful tool for wildlife monitoring! Stay tuned for exciting updates!

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
         - SAVE_INTERVAL=0  # Seconds between saving # 0 ensures all detected frames are saved
         - MAX_FPS_DETECTION=1  # CPU/GPU usage limiter
         - STREAM_FPS=3  # Max FPS for the output stream
         - STREAM_WIDTH_OUTPUT_RESIZE=800  # Default streaming width for resizing: 800
         - TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN
         - TELEGRAM_CHAT_ID=YOUR_CHAT_ID
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


---


## Contributing

Contributions are welcome! If you have ideas or improvements, please open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

