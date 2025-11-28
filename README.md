<div align="center">

# 👁️ Multi-Attribute Biometric Profiler

A comprehensive, real-time computer vision dashboard that aggregates heterogeneous deep learning models to generate a holistic biometric profile.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-blueviolet)

</div>

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Models Used](#models-used)
- [License](#license)

## 📖 Overview
This application solves the challenge of running multiple, resource-intensive AI models simultaneously on a live video feed. By utilizing a hybrid runtime environment, it bypasses version conflicts (e.g., Keras 2 vs Keras 3) and provides a "stabilized" reporting engine that eliminates frame-by-frame prediction jitter.

It is designed for use cases in **retail analytics**, **demographic profiling**, and **automated security reporting**.

## ✨ Features
* **Real-Time Analysis:** Processes live webcam feed to detect faces and clothing items.
* **Holistic Profiling:** Simultaneously predicts Age, Gender, Ethnicity, Skin Tone, Hair Style, and Clothing.
* **Hybrid Runtime Engine:** Seamlessly bridges TensorFlow, PyTorch/HuggingFace, and ONNX models.
* **Result Stabilization:** Uses a statistical buffer (Voting/Averaging) to smooth out prediction flickering.
* **Granular Color Logic:** Analyzes clothing items in HSV color space to distinguish specific shades (e.g., "Dark Blue" vs "Black").
* **Interactive GUI:** Built with Tkinter for a responsive, user-friendly experience with a "Scan Mode" workflow.

## 🏗️ Architecture
The system follows a modular "Hub-and-Spoke" architecture:
1.  **Input:** OpenCV captures video frames.
2.  **Preprocessing Hub:** Resizes and normalizes crops for specific model requirements (224x224, 299x299, etc.).
3.  **Inference Engine:**
    * *ONNX Runtime:* Age & Ethnicity (EfficientNet)
    * *TensorFlow (Legacy):* Gender (Xception) & Hair (U-Net)
    * *PyTorch:* Skin Tone (ViT) & Clothing (Object Detection)
4.  **Stabilizer:** Aggregates the last `N` frames to determine the most probable attribute.
5.  **Output:** Tkinter Dashboard updates asynchronously to maintain UI responsiveness.

## ⚙️ Installation

### Prerequisites
* Python 3.9 or higher
* CUDA-capable GPU (Recommended for real-time performance)

### Setup
1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/YourUsername/Biometric-Profiler.git](https://github.com/YourUsername/Biometric-Profiler.git)
    cd Biometric-Profiler
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Model Setup**
    Ensure the following model files are present in the root directory:
    * `age_race.onnx`
    * `gender_recognition_xception_finetuned.keras`
    * `hair_segmentation_model.h5`
    * `skin_tone_model.safetensors` (+ `config.json`)
    * `fashion_model/` (Directory or HuggingFace cache)

## 🚀 Usage

Run the main dashboard script:

```bash
python dashboard.py
