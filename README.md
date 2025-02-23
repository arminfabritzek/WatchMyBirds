# WatchMyBirds


![WatchMyBirds in Action](assets/birds_1280.gif)
*Live object detection capturing birds in real-time!*

---

## Overview

**WatchMyBirds** is a lightweight, customizable object detection application for real-time monitoring using webcams, RTSP streams, and Docker. It is built using PyTorch and TensorFlow, and it supports live video streaming, automatic frame saving based on detection criteria, and integration with Telegram for notifications. The application is ideal for hobbyists, researchers, and wildlife enthusiasts interested in automated visual monitoring.


---

## Build Status


[![Build and Push Docker Image](https://github.com/arminfabritzek/WatchMyBirds/actions/workflows/docker.yml/badge.svg)](https://github.com/arminfabritzek/WatchMyBirds/actions/workflows/docker.yml)


---

## 🚀 Key Features

- **Real-Time Object Detection**:  
  Transform any webcam or RTSP camera stream into a powerful detection system.
  - Seamless integration with **MotionEye OS** for network camera support.  
  - Tested with **PTZ** Camera.  


- **Optimized for Diverse Hardware**:  
  Built to run across various devices with performance and accuracy in mind.
  - Runs on Docker (e.g., Synology NAS), macOS, and planned support for Raspberry Pi and NVIDIA Jetson  


- **Customizable Detection & Classification**  
  - Filter detections by specific object classes  
  - Fine-tune confidence thresholds for detections and frame saving


- **Integrated Notifications**  
  - Telegram alerts for detections  
  - Configurable cooldowns to prevent spam


- **State-of-the-Art AI Models**  
  - Pre-trained models including `yolov8n`, `EfficientDet Lite4`
  - Planned to integrate Megvii-BaseDetection `YOLOX-Nano` custom trained models
  - Future support for TensorFlow and TFLite models


---

### 📡 **Tested IP Cameras (RTSP Input)**
| Camera Model                                   | Connection          | Status  | Notes                                                           |
|------------------------------------------------|---------------------|---------|-----------------------------------------------------------------|
| **Low-priced PTZ Camera**                      | RTSP                | ✅ Works | Verified stable RTSP stream.                                    |
| **Raspberry Pi 3 + Zero 2 + Raspberry Pi Cam** | MotionEye OS (HTTP Stream) | ✅ Works |                                                                 |
| **Seeking Sponsors**                           | N/A                | ❓ Pending | Looking for sponsors to provide more camera models for testing. |

🔹 *Planned: Expanding RTSP camera compatibility & adding PTZ control.*

📢 *Interested in sponsoring a test? Reach out on GitHub!*

---


### 📌 **Contribute Your Results!**
Have you tested **WatchMyBirds** on another **Synology NAS, IP camera, or edge device**?  
Help expand this list! Share your results by opening an issue or pull request on GitHub with:
- Device model & specs
- OS / Docker setup
- Measured FPS or detection performance
- Additional observations  

Your contributions help improve **WatchMyBirds** for everyone! 🚀



---

## 🌟 Future Roadmap  

### 🚀 **Camera & Tracking Enhancements**

- 🏆 **PTZ (Pan-Tilt-Zoom) Camera Control** → Auto-tracking & framing of detected objects.  

### 🛠️ **AI & Model Optimization**  
- 🏆 **Custom AI Models** → Train specialized models for birds, insects, and plants with **AIY vision classifiers**.  
- 🏆 **Advanced Object Classification** → More precise species identification using **batch classification**.  

### ⚡ **Performance & Edge Deployment**  
- 🏆 **Optimized for Edge Devices** → Efficient performance tuning for **Raspberry Pi 4 & 5** and **NVIDIA Jetson Nano**.  
- 🏆 **Low-Latency Processing** → Further optimizations for improved real-time detection with reduced computational overhead.  

### 📊 **Expanded Data & Logging Capabilities**  
- 🏆 **Adaptive Filtering** → Dynamic threshold adjustments for smarter detection.  
- 🏆 **Bird Activity Analytics** → Generate statistics on **number of visits, species diversity, and time-based patterns**.  
- 🏆 **Graphical Dashboards** → Visualize detection trends and bird activity with **interactive charts & reports**.

---


## Installation and Setup


### Using Docker

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
         - PUID=1000  # User ID (UID) to run the container with proper permissions.
         - PGID=1000   # Group ID (GID) to ensure correct file access within the container.
         - TZ=Europe/Berlin  # Set your local timezone to ensure correct timestamp logging.
         - DEBUG_MODE=False  # Set to "True" for additional logging output for debugging.
         - MODEL_CHOICE="yolo8n"  # The object detection model to use (e.g., yolo8n, pytorch_ssd).
         - YOLO8N_MODEL_PATH="/models/yolov8n.pt"  # The object detection model path. Currently pointing to the COCO pre-trained.
         - CLASS_FILTER="'["bird", "person"]'"  # Define which object classes to detect; default is "bird".
         - CONFIDENCE_THRESHOLD=0.8  # Minimum confidence score for an object to be considered detected.
         - SAVE_THRESHOLD=0.8  # Minimum confidence score for saving an image of a detected object.
         - MAX_FPS_DETECTION=1  # Limit detection FPS to reduce CPU/GPU usage (higher values increase load).
         - STREAM_FPS=3  # Maximum FPS for the video stream output.
         - STREAM_WIDTH_OUTPUT_RESIZE=800  # Resize the output stream width to optimize performance.
         - TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN  # Token for Telegram bot notifications (replace with your actual bot token).
         - TELEGRAM_CHAT_ID=YOUR_CHAT_ID  # Telegram chat ID to send detection alerts to.
         - DAY_AND_NIGHT_CAPTURE=False  # Set to "True" to allow detection at night; "False" stops detection at night.
         - DAY_AND_NIGHT_CAPTURE_LOCATION="Berlin"  # The location to determine daylight hours (used for night mode).
         - CPU_LIMIT=1  # Limit the container to use only 1 CPU core to optimize resource usage.
         - TELEGRAM_COOLDOWN=5  # Delay Telegram Notifications in seconds
       volumes:
         - /your_path/output:/output  # Path for saving output
         - /your_path/models:/models  # Path for your model of choice
       ports:
         - "5001:5001"  # HTTP port for the Flask web app (streaming and API).
         - "8554:8554"  # RTSP port for serving the video stream.
         - "8081:8081"  # MJPEG streaming port.
         - "1936:1936"  # Custom RTMP or alternative streaming port.
         - "8889:8889"  # Custom port for additional services (if needed).
         - "8189:8189/udp"  # UDP port for specific video streaming protocols.
       restart: unless-stopped  # Ensures the container restarts automatically unless manually stopped.
   ```

3. **Start the container** using Docker Compose:

   ```bash
   docker-compose up -d
   ```

This will run the **WatchMyBirds** application, and you can access the livestream at `http://<your-server-ip>:5001`.


The container will use the custom model stored in your host’s ~/your_model_path directory (mounted to /models in the container). Output images and logs are saved in ~/your_output_path.


---
### Manual Setup (Without Docker)

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

For a webcam connected via USB use:
   ```plaintext
   VIDEO_SOURCE=0
   ```

   For an RTSP stream, use:
   ```plaintext
   VIDEO_SOURCE=rtsp://user:password@192.168.0.2:554/1
   ```


Set the path to the model you downloaded.
   ```plaintext
   YOLO8N_MODEL_PATH="/models/yolov8n.pt"
   
   DEBUG_MODE=True
   ```



# Usage

- **Livestream**: `http://<your-server-ip>:5001` displays a real-time video feed with bounding boxes around detected objects.
- **Saved Frames**: Frames with objects exceeding the confidence threshold are saved in the `output/` directory.
- **CSV Logging**: All detections are logged to a CSV file (`all_bounding_boxes.csv`) for analysis.

---





## Contributing

Contributions are welcome! If you have ideas or improvements, please open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

