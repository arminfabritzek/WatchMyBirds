# WatchMyBirds - Real-time object detection and classification

[![Build and Push Docker Image](https://github.com/arminfabritzek/WatchMyBirds/actions/workflows/docker.yml/badge.svg)](https://github.com/arminfabritzek/WatchMyBirds/actions/workflows/docker.yml) ![Python](https://img.shields.io/badge/python-3.10+-blue.svg)


![WatchMyBirds in Action](assets/birds_1280.gif)
*Real-time object detection and classification in action!*

---

![WatchMyBirds in Action](assets/app_landing_page.jpg)
*Clean, user-friendly web interface with image gallery support!*


---


## ⚡ Quickstart
- Create the `output` folder and the model cache directory.
- All models are stored under `MODEL_BASE_PATH` (default `models`) and downloaded automatically from Hugging Face.
```bash
git clone https://github.com/arminfabritzek/WatchMyBirds.git
cd WatchMyBirds
cp docker-compose.example.yml docker-compose.yml
docker-compose up -d
```

---


## 📚 Table of Contents
- [Overview](#overview)
- [🚀 Key Features](#-key-features)
- [📡 Tested Cameras](#-tested-cameras)
- [📌 Share Your Results](#-share-your-results)
- [🌟 Roadmap](#-roadmap)
- [⚙️ Installation and Setup](#-installation-and-setup)
- [📺 Usage](#-usage)
- [🤝 Contributing](#-contributing)
- [🙌 Acknowledgements](#-acknowledgements)
- [📄 License](#-license)

---

## Overview

**WatchMyBirds** is a lightweight, customizable object detection application for real-time monitoring using webcams, RTSP streams, and Docker. It is built using PyTorch and TensorFlow, and it supports live video streaming, automatic frame saving based on detection criteria, and integration with Telegram for notifications. The application is ideal for hobbyists, researchers, and wildlife enthusiasts interested in automated visual monitoring.


---



## 🚀 Key Features

- **Real-Time Object Detection**:  
  Transform any webcam or RTSP camera stream into a powerful detection system.
  - Seamless integration with **MotionEye OS** for network camera support.  
  - Tested with cheap **IP** & **PTZ** Cameras.  


- **Optimized for Diverse Hardware**:  
  Built to run across various devices with performance and accuracy in mind.
  - Runs on Docker (e.g., Synology NAS), macOS, with planned support for Raspberry Pi and NVIDIA Jetson  


- **Integrated Notifications**  
  - Telegram alerts for detections  


- **State-of-the-Art AI Models**
  - Pre-trained models including `yolov11`, `EfficientNet`

- **Modular Capture Pipeline**
  - Explicit start/stop control via `VideoCapture` and timestamped streaming through `FrameGenerator`


---

## 📡 Tested Cameras
| Camera Model                                   | Connection          | Status  | Notes                                                           |
|------------------------------------------------|---------------------|---------|-----------------------------------------------------------------|
| **Low-priced PTZ Camera**                      | RTSP                | ✅ Works | Stable RTSP stream verified.                                    |
| **Raspberry Pi 3 + Zero 2 + Raspberry Pi Cam** | MotionEye OS (HTTP Stream) | ✅ Works |                                                                 |
| **Seeking Sponsors**                           | N/A                | ❓ Pending | Looking for sponsors to provide more camera models for testing. |

🔹 *Planned: Expanding RTSP camera compatibility & adding PTZ control.*

📢 *Interested in sponsoring a test? Reach out on GitHub!*

---


## 📌 Share Your Results
Have you tested **WatchMyBirds** on another **Synology NAS, IP camera, or edge device**?  
Help expand this list! Share your results by opening an issue or pull request on GitHub with:
- Device model & specs
- OS / Docker setup
- Measured FPS or detection performance
- Additional observations  

Your contributions help improve **WatchMyBirds** for everyone! 🚀



---

## 🌟 Roadmap

### 🧠 AI & Model Optimization
- 🏆 Train custom bird/insect/plant models with classifiers  

### ⚡ Performance & Edge Deployment
- 🏆 Optimize for Raspberry Pi 4/5 and Jetson Nano  

### 📊 Analytics & Visualization
- 🏆 Track bird visits, diversity, and time patterns  
- 🏆 Interactive dashboards for visualization  

---


## ⚙️ Installation and Setup

---
### Using Docker


1. Clone the repo and copy the example compose file:
   ```bash
   cp docker-compose.example.yml docker-compose.yml
   ```

2. Edit the .yml file to match your stream settings, then run:

    ```bash
    docker-compose up -d
   ```

➕ See [`docker-compose.example.yml`](docker-compose.example.yml) for all available environment variables.


This will run the **WatchMyBirds** application, and you can access the livestream at `http://<your-server-ip>:8050`.



---
### Manual Setup (Without Docker)

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/arminfabritzek/WatchMyBirds.git
   cd WatchMyBirds
   ```

2. **Set Up a Virtual Environment** (optional but recommended; tested on Python 3.10 and 3.12):
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
   VIDEO_SOURCE=rtsp://user:password@192.168.0.2:8554/stream1
   ```

5. **Start the Application**:
   ```bash
   python main.py
   ```


## 📺 Usage
   The livestream will be available at:
   - Local: `http://localhost:8050`
   - Remote: `http://<your-server-ip>:8050`


---



## 🤝 Contributing

Have ideas or improvements? Open an issue or submit a pull request!


---

## 🙌 Acknowledgements

This project uses **Label Studio** – provided free through the Academic Program by HumanSignal, Inc.  
[![Label Studio Logo](https://user-images.githubusercontent.com/12534576/192582340-4c9e4401-1fe6-4dbb-95bb-fdbba5493f61.png)](https://labelstud.io)

---

## 📄 License
This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.

