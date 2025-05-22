# Faster ILOD: Incremental Learning for Object Detection (VOC 2007/2012)

This project reproduces the **Faster ILOD** (Peng et al., 2020) algorithm for incremental object detection, using the PASCAL VOC dataset.

---

## 📖 Overview

Faster ILOD is a knowledge distillation-based approach that enables **incremental object detection** without catastrophic forgetting. It is based on **Faster R-CNN (ResNet-50 C4)** and extends the `maskrcnn-benchmark` framework.

---

## 🛠️ Environment Setup

### ⚙️ Requirements

* Python 3.10
* PyTorch 1.10.2 + CUDA 11.3
* torchvision 0.11.3
* GCC ≥ 4.9
* CUDA Toolkit ≥ 9.0
* Other dependencies:

  ```bash
  pip install yacs ninja cython matplotlib tqdm opencv-python
  ```

### ⚙️ PyTorch & torchvision Installation (CUDA 11.3)

```bash
pip install https://download.pytorch.org/whl/cu113/torch-1.10.2%2Bcu113-cp310-cp310-linux_x86_64.whl
pip install https://download.pytorch.org/whl/cu113/torchvision-0.11.3%2Bcu113-cp310-cp310-linux_x86_64.whl
```

### 📦 Install `maskrcnn-benchmark`

```bash
git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
cd maskrcnn-benchmark
python setup.py build develop
```

> ⚠️ Note: Ensure `nvcc` is available (`/usr/local/cuda/bin/nvcc`). If not, install CUDA Toolkit and add it to your PATH.

---

## 📁 Dataset Setup

* Download and extract the **VOC 2007 + 2012** dataset.
* Set the path in `maskrcnn_benchmark/config/paths_catalog.py`:

```python
class DatasetCatalog(object):
    DATA_DIR = "/home/jin/Faster-ILOD/data/VOCdevkit"
```

---

## 🧪 Experiment Protocol

* **Step 1:** Train base model on **15 classes**
* **Step 2:** Incrementally add **5 new classes**
* Knowledge distillation applied at: **feature**, **RPN**, and **RCN** stages.

### ⚙️ Training Scripts

```bash
# Step 1 - Base training (15 classes)
python tools/train_first_step.py --config-file configs/e2e_faster_rcnn_R_50_C4_1x_Source_model.yaml

# Step 2 - Incremental learning (5 new classes)
python tools/train_incremental.py --config-file configs/e2e_faster_rcnn_R_50_C4_1x_Target_model.yaml
```

---

## 📊 Results

| Stage                    | Old mAP (%) | New mAP (%) | Total mAP (%) |
| ------------------------ | ----------- | ----------- | ------------- |
| Base (15 classes)        | 68.2        | -           | 68.2          |
| Incremental (+5 classes) | 66.3        | 61.5        | 65.7          |

* ✅ Catastrophic forgetting was mitigated.
* ✅ New class performance remained stable.
* ✅ Total performance showed minimal degradation.

### 📈 Visualization

---

## 📚 Reference

Peng, C., Zhao, K., & Lovell, B. C. (2020).
**Faster ILOD: Incremental Learning for Object Detectors based on Faster RCNN**.
*Pattern Recognition Letters.*

```bibtex
@article{peng2020faster,
  title={Faster ILOD: Incremental Learning for Object Detectors based on Faster RCNN},
  author={Peng, Can and Zhao, Kun and Lovell, Brian C},
  journal={Pattern Recognition Letters},
  year={2020}
}
```

---

## 🤝 Acknowledgements

This implementation builds upon [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) by Facebook AI Research.
