# ADAS (Advanced Driver Assistance System)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)

---

## 📜 Description

This project implements a **Level 3 Advanced Driver Assistance System (ADAS)** using only stereo cameras, eliminating the need for physical LiDAR sensor. By employing a **pseudo-LiDAR approach**, it reconstructs the 3D environment from stereo images with multiple advanced perception modules.

### Key Features:

- 🟨 **Lane Detection**: Accurately detects lane markings to assist with lane keeping and path planning.
- 🚦 **Traffic Light Detection**: Identifies traffic lights and determines their current state.
- 📦 **3D Object Detection**: Detects and localizes objects (vehicles, pedestrians, etc.) in 3D space using point cloud data.
- ☁️ **Pseudo-LiDAR Generation**: Converts stereo depth maps into 3D point clouds, simulating LiDAR data.
- 📷 **Stereo Depth Estimation**: Computes disparity maps from stereo image pairs to infer scene depth.
- 🖼️ **2D/3D Visualization**: Renders results on input images, disparity maps, and 3D point clouds for debugging or real-time display.

This system is designed for easy integration into robotics platforms, simulators, or real-world vehicles using only stereo vision for cost-effective and flexible ADAS development.

---

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/batman-911/ADAS.git
cd ADAS

# Create and activate venv
python -m venv adas-env
source adas-env/bin/activate  # Linux/Mac
# OR
adas-env\Scripts\activate    # Windows

# Set up the Python environment and install dependencies
bash setup.sh
```
---

## 📁 Project Structure

```text
ADAS/
├── config/            # YAML configuration files
├── data/              # Input data (e.g., KITTI)
├── models/            # Pre-trained or custom models
├── src/               # Source code (perception, decision, visualization)
│   ├── perception/
│   ├── decision/
│   └── visualization/
├── weights/           # Model weights
├── main.py           # Main pipeline launcher
├── setup.sh          # Installation script
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation
```

---

## Run demo

```bash
# Basic stereo video processing
python main.py --left data/left.avi --right data/right.avi

# Stereo calibration mode (optional)
python main.py --left calibration_images/left/ --right calibration_images/right/ --calibrate
```
---

## 📁 Dataset

Use datasets like [KITTI](http://www.cvlibs.net/datasets/kitti/) and place them under `data/`.
Make sure calibration files are available for stereo modules.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file.
