# Faster ILOD: Incremental Learning for Object Detection (VOC 2007/2012)

This project reproduces the **Faster ILOD** (Peng et al., 2020) algorithm for incremental object detection, using the PASCAL VOC dataset.

---

## 📖 Overview

Faster ILOD is a knowledge distillation-based approach that enables **incremental object detection** without catastrophic forgetting. It is based on **Faster R-CNN (ResNet-50 C4)** and extends the `maskrcnn-benchmark` framework.

---

## 🛠️ Environment Setup

### 🔧 Requirements

- Python 3.10
- PyTorch 1.10.2 + CUDA 11.3
- torchvision 0.11.3
- GCC ≥ 4.9
- CUDA Toolkit ≥ 9.0
- Other dependencies:
  ```bash
  pip install yacs ninja cython matplotlib tqdm opencv-python
