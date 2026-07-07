# 🌉 Underwater Exposed Rebar Detection for Bridges and LIMELight Low-Light Enhancement Based on RDKX5 Robot
## LIMELight Low-Light Enhancement + YOLOv11 Object Detection

> This project targets the scenario of underwater structural health inspection for bridges, and proposes an end-to-end intelligent recognition system integrating **underwater low-light image enhancement algorithm** and **object detection technology**. The system is fully deployed on the RDK X5 edge computing platform to achieve high-precision real-time detection of exposed rebar defects in underwater structures.
>
> To address industry pain points including image quality degradation caused by insufficient illumination, uneven lighting and turbid water in underwater environments, as well as the scarcity of public underwater exposed rebar datasets, this project independently develops the LIMELight low-light enhancement algorithm, constructs a self-built underwater exposed rebar dataset, and implements high-precision detection based on YOLOv11. Finally, through edge-side optimization, a stable inference frame rate of over 13 FPS is achieved on RDK X5.

## 📑 Table of Contents
- [Key Features](#key-features)
- [Performance Metrics](#performance-metrics)
- [Project Structure](#project-structure)
- [Environment Dependencies](#environment-dependencies)
- [Quick Start](#quick-start)
- [Technical Innovations](#technical-innovations)
- [Application Scenarios](#application-scenarios)
- [License](#license)

## ✨ Key Features

### 🔦 LIMELight Underwater Low-Light Enhancement Algorithm
Improved targeted based on Retinex theory and the classic LIME algorithm, it adopts four core mechanisms: **adaptive illumination partitioning, dual gamma correction, highlight protection, and frequency-domain deblurring**, which effectively solves the problems of brightness attenuation and uneven illumination in underwater images. The algorithm reduces the time complexity from O(n²) to O(n) and the space complexity to O(1), achieving approximately 56× speedup and perfectly adapting to real-time operation on edge devices.

### 📦 Self-Built Underwater Exposed Rebar Dataset
A controllable underwater experimental environment is built to collect rebar images under different illumination intensities and water turbidity levels. There are a total of **2,944 high-quality annotated images**, covering rebar samples with various thicknesses and corrosion degrees, filling the gap of public datasets in the field of underwater bridge structure exposed rebar detection.

### 🎯 YOLOv11 High-Precision Exposed Rebar Detection
The detection model is trained based on the Ultralytics YOLOv11 framework. After enhancement by the LIMELight algorithm, the detection precision reaches **86.41%** and mAP@0.5 reaches **81.57%**, which is 9.13% higher than the detection precision on raw images, fully meeting the accuracy requirements of engineering inspection.

### ⚡ Real-Time Edge Deployment on RDK X5
INT8 model quantization is completed based on the Horizon Open Explorer toolchain, and the inference performance is deeply optimized through strategies of **removing redundant dequantization operators and threshold pre-activation pre-screening**. The total single-frame processing time of the system is less than 75 ms, and the overall frame rate is stable above 13 FPS, supporting end-to-end real-time processing of video streams.

## 📊 Performance Metrics

### 1. Comparison of Image Enhancement Effects
A horizontal comparison with mainstream low-light enhancement algorithms on the self-built underwater low-light dataset is as follows:

| Method | PSNR ↑ (dB) | SSIM ↑ | PCQI ↑ |
| :---- | :---------: | :----: | :----: |
| PIE   |    13.25    | 0.298  | 0.486  |
| Fusion|    11.16    | 0.276  | 0.403  |
| LIME  |     7.08    | 0.197  | 0.354  |
| TEBCF |    13.13    | 0.143  | 0.155  |
| ICSP  |    14.28    | 0.213  | 0.227  |
| PCDE  |    14.13    | 0.284  | 0.409  |
| **LIMELight (Ours)** | **22.19** | **0.597** | **0.694** |

### 2. Exposed Rebar Detection Performance
Comparison of YOLOv11 detection metrics on the raw dataset and the enhanced dataset:

| Metric          | Raw Dataset | Enhanced Dataset | Improvement |
| :------------ | :--------: | :--------: | :------: |
| Precision     |   77.28%   |   86.41%   |  +9.13%  |
| Recall        |   65.78%   |   71.92%   |  +6.14%  |
| F1 Score      |   71.07%   |   78.50%   |  +7.43%  |
| mAP@0.5       |   74.94%   |   81.57%   |  +6.63%  |
| mAP@0.5:0.95  |   45.91%   |   54.50%   |  +8.59%  |

### 3. Edge Deployment Performance (RDK X5, 640×640 Resolution)
- ✅ Single-frame image enhancement time: ≤ 60 ms
- ✅ Single-frame inference time of detection model: ≤ 20 ms
- ✅ Total single-frame processing time of the system: ≤ 75 ms
- ✅ Stable running frame rate: ≥ 13 FPS
- ✅ Cosine similarity of model output before and after quantization: ≥ 97.9%

## 📁 Project Structure
qiansai2026/
├── YOLO11/ # YOLOv11 detection code (training, inference, model export)
├── RDKX5/ # RDK X5 board-side deployment code and quantized models
│ ├── main.cc # Board-side main program
│ ├── 1.cc # Auxiliary function implementation
│ ├── LIME.bin # Optimized quantized model
│ └── LIME_no_modified.bin # Original quantized model
├── LIMELight/ # Python implementation of LIMELight low-light enhancement algorithm
│ └── LIMELight.py # Core algorithm source code
└── dataset/ # Self-built underwater exposed rebar dataset (images + YOLO format annotations)


## 🔧 Environment Dependencies

### Python Development Environment
- Python 3.9+
- PyTorch 1.13+
- Ultralytics
- OpenCV-Python
- NumPy
- scikit-image

## 🚀 Quick Start

### 1. LIMELight Image Enhancement (Python Version)
Enter the `LIMELight` directory and call the algorithm to enhance a single underwater image.

### 2. YOLOv11 Model Training and Inference
### 3. RDK X5 Board-Side Deployment
### 4. Dataset Usage
The `dataset` directory contains the self-built underwater exposed rebar dataset, which is divided into training set, validation set and test set in standard YOLO format, and can be directly used for model training and performance evaluation.

## 💡 Technical Innovations
### 1. Algorithm Innovation:
The LIMELight underwater low-light enhancement algorithm is proposed, which solves the problem of uneven underwater illumination through illumination partitioning and dual gamma correction, greatly reduces computational complexity, and realizes real-time enhancement on edge devices.
### 2. Data Innovation:
A multi-condition underwater exposed rebar dataset is independently constructed, covering different lighting, turbidity and rebar morphologies, filling the gap of public data in this field.
### 3. Deployment Innovation:
The full algorithm pipeline is deployed on the RDK X5 platform. Through strategies such as operator removal and post-processing optimization, real-time inference above 13 FPS is achieved, meeting the real-time requirements of underwater inspection.

## 🌊 Application Scenarios
> Automated detection of exposed rebar defects in bridge underwater pile foundations and piers
> 
> Visual inspection of underwater pipelines and wharf structures
> 
> Image preprocessing for underwater vision tasks such as aquaculture and marine debris cleanup
> 
> Visual perception enhancement for various underwater robots

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.
