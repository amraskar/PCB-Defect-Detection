# PCB Defect Detection and Localization using Deep Learning

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/YOLO11-Ultralytics-blue)](https://github.com/ultralytics/ultralytics)

This repository contains an end-to-end computer vision project focused on automated high-precision PCB (Printed Circuit Board) quality assurance. 

**Academic Context**: Foundation of Deep Learning – Master's Degree in Data Science, University of Milano-Bicocca (UNIMIB)

The project systematically explores the challenge of locating and classifying microscopic physical defects in circuit boards. It progresses from a custom convolutional network built entirely from scratch to state-of-the-art architectures using transfer learning, YOLO11, and a custom multi-scale attention module.

---

## 📌 Project Architecture Evolution

Rather than jumping directly to a pre-trained API, this project was built incrementally to study the physical limits and design choices of deep neural networks:

    [ Raw PCB Images ] 
           │
           ├──► 1. Exploratory Data Analysis (EDA)
           │
           ├──► 2. Custom YOLO-from-Scratch (Grid-based 100x100 Regression)
           │
           ├──► 3. ResNet-34 Feature Backbone Integration
           │
           ├──► 4. YOLO11 State-of-the-Art Baseline
           │
           └──► 5. YOLO11 + MCVAM (Multi-Scale Cross-View Attention Module)

---

## 📂 Repository Structure & Notebook Breakdown

### 📊 1. Exploratory Data Analysis (EDA)
* Analyzes bounding box distributions, aspect ratios, and spatial sizing across the dataset.
* Establishes that microscopic defects (e.g., small spurs or mouse bites) require high-resolution inputs (800 x 800 pixels) to avoid losing key spatial features.
* Evaluates class imbalances to guide later design choices in loss function weighting.

### 🧱 2. Custom YOLO-Style Detector from Scratch
* **Data Preparation:** Dataset split into 70% Train, 10% Validation, 20% Testing. Albumentations (Horizontal/Vertical Flips, Brightness/Contrast adjustments) are applied only to the training set to prevent overfitting and force the model to learn geometric defect shapes rather than memorizing pixels.
* **Custom Architecture (PCBDetectorV3):** Developed using a modular ConvBlock (3x3 Conv + Batch Normalization + ReLU) stacked over three major downsampling stages (halving spatial dimensions using Stride 2: 800 -> 400 -> 200 -> 100).
* **Grid YOLO Concept:** Divides the 800 x 800 image into a 100 x 100 grid. If a defect center falls inside a grid cell, that single cell is 100% responsible for detecting that defect.
* **The Detection Head:** A final 1x1 convolution perfectly squashes the 256 feature channels down into our required 11 output channels (1 Objectness + 6 Classes + 4 Bounding Box coordinates) without changing the 100 x 100 spatial grid.
* **Custom Multi-Task Loss Function:** Simultaneously optimizes three objectives:
  * *Objectness (Binary Cross-Entropy):* Uses a positive weight multiplier (pos_weight=20.0) to counteract the extreme imbalance of empty background cells.
  * *Classification (Cross-Entropy):* Masked to only calculate on cells containing actual defects.
  * *Localization (Smooth L1 Loss):* Scaled by a factor of 5.0 to force the optimizer to prioritize precise bounding box coordinate regression.

### 🧬 3. ResNet Backbone Adaptation
* Replaces the custom convolutional blocks with a pre-trained **ResNet-34** backbone.
* Modifies the feature propagation path to adapt the deep feature maps onto our custom 100 x 100 grid structure for regression.
* Benchmarks the performance improvement gained by migrating from random initialization to transfer learning.

### ⚡ 4. YOLO11 Baseline Implementation
* Evaluates the modern state-of-the-art **YOLO11** (Ultralytics) as an industrial baseline.
* Standardizes evaluations using precision-recall curves, confusion matrices, and mean Average Precision (mAP) metrics.

### 🧠 5. YOLO11 + MCVAM (Multi-Scale Cross-View Attention)
* Integrates a custom **Multi-Scale Cross-View Attention Module (MCVAM)** directly into the YOLO11 pipeline.
* **The Goal:** Focus the attention of the convolutional channels dynamically on tiny anomalies while ignoring highly repetitive background copper paths.
* Achieves a significant boost in Recall and localization accuracy for elusive defect types.

---

## 🔬 Target Defect Classes

The models are trained to recognize and isolate 6 distinct industrial defect types:
1. **Missing Hole:** Drill omissions on terminal pads.
2. **Mouse Bite:** Edge erosion along conduction lines.
3. **Open Circuit:** Broken pathways causing electrical interruptions.
4. **Short:** Spurious connections bridging independent circuits.
5. **Spur:** Protruding copper defects along traces.
6. **Spurious Copper:** Isolated copper droplets remaining after etching.
