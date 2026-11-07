# An Edge-AI System for Arabic Sign Language Detection Using YOLOv8m and Raspberry Pi 5

[![Paper](https://img.shields.io/badge/Paper-Future_Internet-blue)](#) [![Dataset](https://img.shields.io/badge/Dataset-Custom_ArSL-green)](https://drive.google.com/drive/folders/1PmPBZ0STuz2Dh63e2UuzwXhnvdIdGEAR?usp=sharing) [![Weights](https://img.shields.io/badge/Weights-YOLOv8m-orange)](https://drive.google.com/drive/folders/1XXze9fPnTA2mRvC8mJ-rIK5xZXEmi15h?usp=sharing)

This repository contains the official implementation, trained model weights, and custom dataset for the paper **"An Edge-AI System for Arabic Sign Language Detection Using YOLOv8m and Raspberry Pi 5"** (Published in *Future Internet*, 2026). 

This project provides a standalone, edge-AI solution engineered to detect isolated Arabic Sign Language (ArSL) gestures and render the translated text locally on a compact display. By pushing the computational workload entirely to the edge, the system eliminates reliance on cloud infrastructure, ensuring privacy and offline capability.

## Key Contributions
* **Edge-Optimized Pipeline:** Full-stack implementation on the Raspberry Pi 5, achieving a measured system throughput of 3.52 FPS (283.65 ms per-frame latency) entirely on CPU.
* **Temporal Stabilization:** Incorporates deterministic temporal filtering—a 3-frame prediction agreement rule and a 3-second inactivity timeout—to stabilize framewise predictions without the computational overhead of sequence models (e.g., LSTMs).
* **Localized RTL Text Rendering:** Integrates a specialized pipeline for contextual Arabic glyph shaping and right-to-left (RTL) sentence buffering directly onto a 128x64 OLED micro-display.
* **Signer-Disjoint Dataset:** A custom 1,881-image ArSL dataset covering nine predefined classes, partitioned using a strict signer-disjoint split to prevent data leakage and evaluate true generalization.

## Architecture & Hardware

The detection pipeline is designed for strict resource constraints while maintaining high accuracy. The system captures video via a MIPI CSI interface, resizes frames to 256x256, and executes inference via an ARM-compatible PyTorch build of YOLOv8m.

* **Hardware Core:** Raspberry Pi 5 (4GB, Broadcom BCM2712)
* **Vision Interface:** Arducam Camera Module V3 (16MP Sony IMX519, motorized autofocus)
* **Output Display:** 128x64 I2C OLED Module
* **Software Stack:** Raspberry Pi OS (64-bit), Python 3.12.3, OpenCV (v4.10.0), PyTorch (v2.7.0 CPU-only), Ultralytics YOLO (v8.3.146), Picamera2 (v0.3.37)

### Operation Preview

<img width="553" height="315" alt="Live Detection Preview" src="https://github.com/user-attachments/assets/9b1bc41d-9ef9-4c64-94df-cb4b10ee1e16" />

## System Performance

The YOLOv8m model (25.86M parameters, 12.65 GFLOPs at 256x256) was selected following extensive profiling against YOLOv5m and YOLOv9m. Below are the evaluation metrics based on the held-out, unseen-signer test set:

* **Mean Average Precision (mAP@0.5):** 0.9583 ± 0.0085
* **Precision:** 0.9018 ± 0.0148
* **Recall:** 0.9408 ± 0.0225
* **F1-Score:** 0.9148 ± 0.0130

Runtime profiling on the Pi 5 demonstrates that aggressive system-cleanup optimization reduced overall pipeline latency by 37.7%, bringing inference time to ~271 ms and enabling interactive, step-by-step gesture translation.

## Data & Resources
* **[Pre-trained Model Weights (.pt)](https://drive.google.com/drive/folders/1XXze9fPnTA2mRvC8mJ-rIK5xZXEmi15h?usp=sharing)** 
* **[Custom ArSL Dataset](https://drive.google.com/drive/folders/1PmPBZ0STuz2Dh63e2UuzwXhnvdIdGEAR?usp=sharing)** (1,881 images, 9 gesture classes, 5 signers)
