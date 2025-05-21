# ADAS (Advanced Driver Assistance System)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)

---

## 📜 Description

This project is a Level 3 Advanced Driver Assistance System (ADAS) built entirely using a stereo camera setup, with no physical LiDAR sensor required. Leveraging a pseudo-LiDAR approach, it reconstructs the 3D environment from stereo images and integrates several advanced perception modules.

Key Features:
🟨 Lane Detection: Accurately detects lane markings to assist with lane keeping and path planning.

🚦 Traffic Light Detection: Identifies traffic lights and determines their current state.

📦 3D Object Detection: Detects and localizes objects (vehicles, pedestrians, etc.) in 3D space using point cloud data.

☁️ Pseudo-LiDAR Generation: Converts stereo depth maps into 3D point clouds, simulating LiDAR data.

📷 Stereo Depth Estimation: Computes disparity maps from stereo image pairs to infer scene depth.

🧠 Modular Perception Pipeline: Fully modular architecture—each module (e.g., lane, object, traffic light detection) can be enabled or disabled independently.

🖼️ 2D/3D Visualization: Renders results on input images, disparity maps, and 3D point clouds for debugging or real-time display.

---

## 📁 Structure du projet


---

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/batman-911/ADAS.git
cd ADAS

# Set up the Python environment and install dependencies
bash setup.sh
