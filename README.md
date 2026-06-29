# Tooth Detection and Numbering in Panoramic Radiographs using Enhanced Faster R-CNN

This repository contains the official implementation of the paper **"Tooth detection and numbering in panoramic radiographs using enhanced Faster R-CNN: addressing tooth diversity and image quality challenges"**.

Our method improves Faster R-CNN with four key components—hybrid data augmentation, dynamic anchor generation, Swin-Transformer + FPN backbone, and anatomy-aware weighted loss—to achieve state-of-the-art performance on the DENTEX dental enumeration benchmark.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Model Zoo](#model-zoo)
- [Training](#training)
- [Testing & Evaluation](#testing--evaluation)
- [Inference](#inference)
- [Visualization & Analysis](#visualization--analysis)
- [Experimental Results](#experimental-results)
- [Project Structure](#project-structure)
- [Repository Contents](#repository-contents)
- [License & Citation](#license--citation)

---

## Overview

We address the problem of automatic tooth detection and numbering in dental panoramic radiographs (DPR). Our enhanced Faster R-CNN incorporates:

1. **Hybrid Image Augmentation** — PhotoMetric distortion + geometric erasing to simulate low contrast and metal artifacts.
2. **Dynamic Anchor Generation** — IoU-based clustering with median-enhanced noise-resistant centers to adapt to diverse tooth shapes.
3. **Swin-Transformer + FPN** — Hierarchical window self-attention replaces ResNet for better long-range dependency modeling and multi-scale feature fusion.
4. **Anatomy-Aware Weighted Loss** — Prior knowledge of crown/root positions guides the Smooth L1 loss, emphasizing tooth root localization and excluding rootless structures (implants, dentures).

**FDI tooth numbering**: 32 classes covering permanent dentition (11–18, 21–28, 31–38, 41–48).

---

## Requirements

- **OS**: Ubuntu 18.04+ / Linux
- **Python**: 3.8+
- **CUDA**: 11.8+ (for GPU training)
- **PyTorch**: 2.1.0+cu121
- **MMEngine**: ≥0.7.0
- **MMCV**: ≥2.1.0
- **Hardware**: 4× NVIDIA RTX 3080 (10 GB VRAM each) or equivalent

Install PyTorch first (adjust CUDA version as needed):

```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/wsy258-strar/tooth-detection-faster-rcnn.git
cd mmdetection
```

### 2. Install MMEngine and MMCV

```bash
pip install mmengine
pip install mmcv>=2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
```

### 3. Install MMDetection

```bash
pip install -e .
```

### 4. Install additional dependencies

```bash
pip install -r requirements/optional.txt
```

---

## Dataset Preparation

We use the **DENTEX** dataset from the MICCAI 2023 "Panoramic Dental X-ray Counting and Diagnosis Challenge".

### Download

Download the dataset from the [DENTEX challenge page](https://dentex.grand-challenge.org/) and place it under `data/DENTEX/`.

### Directory Structure

```
data/DENTEX/
└── new_enumeration_data/
    ├── train/              # Training images
    ├── train.json          # COCO-format training annotations (FDI labels)
    ├── val/                # Validation images
    ├── val.json            # COCO-format validation annotations
    ├── test/               # Test images
    └── test.json           # COCO-format test annotations
```

### FDI Label Conversion

The original DENTEX annotations (quadrant + count) have been converted to standard **FDI World Dental Federation notation** using the conversion script. Each tooth is labeled as one of 32 FDI categories:
`11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48`.

### Update Data Root

All config files contain an absolute path `data_root` that you **must update** to match your local setup. Edit each config file in `configs/DETEX/` and change:

```python
data_root = '/data/wangshunyi/mmdetection/data/DENTEX/new_enumeration_data/'
```

to your own absolute path.

### Pre-trained Backbone Weights

Download the Swin-Small ImageNet-1K pre-trained weights:

```bash
mkdir -p pre-trained-models
wget -P pre-trained-models/ https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth
```

Update the `pretrained` path in `configs/DETEX/faster-rcnn_final.py` accordingly.

---

## Model Zoo

| Config File | Description |
|---|---|
| [faster-rcnn_Baseline.py](configs/DETEX/faster-rcnn_Baseline.py) | Baseline Faster R-CNN with ResNet-50 + FPN |
| [faster-rcnn_data_augmentation.py](configs/DETEX/faster-rcnn_data_augmentation.py) | Baseline + hybrid data augmentation |
| [faster-rcnn_set_anchors.py](configs/DETEX/faster-rcnn_set_anchors.py) | Baseline + dynamic anchor generation |
| [faster-rcnnn_swin_fpn_1x_DENTEX.py](configs/DETEX/faster-rcnnn_swin_fpn_1x_DENTEX.py) | Baseline + Swin-Transformer backbone + FPN |
| [faster-rcnn_iou.py](configs/DETEX/faster-rcnn_iou.py) | Baseline + anatomy-aware weighted loss |
| [**faster-rcnn_final.py**](configs/DETEX/faster-rcnn_final.py) | **All improvements combined** (final model) |
| [retinanet_r50_fpn_DENTEX.py](configs/DETEX/retinanet_r50_fpn_DENTEX.py) | RetinaNet baseline (comparison) |
| [ssd_300_DENTEX.py](configs/DETEX/ssd_300_DENTEX.py) | SSD300 baseline (comparison) |
| [dino_swin_l_DENTEX.py](configs/DETEX/dino_swin_l_DENTEX.py) | DINO with Swin-L (comparison) |

---

## Training

All models are trained using distributed training with 4 GPUs.

### Launch Distributed Training

```bash
# Train the final model (all improvements)
bash tools/dist_train.sh configs/DETEX/faster-rcnn_final.py 4

# Train the baseline
bash tools/dist_train.sh configs/DETEX/faster-rcnn_Baseline.py 4

# Train with data augmentation only
bash tools/dist_train.sh configs/DETEX/faster-rcnn_data_augmentation.py 4

# Train with custom anchors only
bash tools/dist_train.sh configs/DETEX/faster-rcnn_set_anchors.py 4

# Train with Swin-Transformer + FPN only
bash tools/dist_train.sh configs/DETEX/faster-rcnnn_swin_fpn_1x_DENTEX.py 4

# Train with prior knowledge loss only
bash tools/dist_train.sh configs/DETEX/faster-rcnn_iou.py 4
```

### Key Training Parameters

| Parameter | Value |
|---|---|
| Image scale | 2000 × 1200 |
| Batch size | 1 per GPU (4 total) |
| Epochs | 100 |
| Optimizer | AdamW (lr=0.001, β=(0.9, 0.999), weight_decay=0.04) |
| LR schedule | Linear warmup (1000 iters) + MultiStepLR (milestones: [8, 11], γ=0.1) |
| Checkpoint saving | Best model by `coco/bbox_mAP`, max 5 checkpoints kept |

### Training Outputs

All training outputs are saved under `work_dirs/<config_name>/`:
- `*.pth` — Model checkpoints
- `202*_*/vis_data/` — TensorBoard logs & JSON scalar logs
- `best_coco_*.pth` — Best checkpoint (selected by validation mAP)

---

## Testing & Evaluation

### Evaluate on Test Set

```bash
# Final model (best result: mAP 0.526)
python tools/test.py \
    configs/DETEX/faster-rcnn_final.py \
    work_dirs/faster-rcnn_final/best_coco_0.517_epoch_12.pth

# Baseline model
python tools/test.py \
    configs/DETEX/faster-rcnn_Baseline.py \
    work_dirs/faster-rcnn_Baseline/best_coco_*.pth
```

### Distribute Testing (if needed)

```bash
bash tools/dist_test.sh configs/DETEX/faster-rcnn_final.py \
    work_dirs/faster-rcnn_final/best_coco_0.517_epoch_12.pth 4
```

---

## Inference

Use the standalone inference script to visualize predictions on a single image:

```bash
python configs/DETEX/inferencer.py
```

Or use `DetInferencer` directly in Python:

```python
from mmdet.apis import DetInferencer

inferencer = DetInferencer(
    model='configs/DETEX/faster-rcnn_final.py',
    weights='work_dirs/faster-rcnn_final/best_coco_0.517_epoch_12.pth'
)
inferencer('/path/to/panoramic_xray.png', out_dir='./outputs/', no_save_pred=False)
```

---

## Visualization & Analysis

### Browse Dataset (with augmentations)

```bash
python tools/analysis_tools/browse_dataset.py \
    configs/DETEX/faster-rcnn_data_augmentation.py \
    --output-dir ./show_data
```

### Plot Training Curves

```bash
# Classification & regression loss
python tools/analysis_tools/analyze_logs.py plot_curve \
    work_dirs/faster-rcnn_final/*/vis_data/*.json \
    --keys loss_cls loss_bbox --out losses.jpg

# mAP metrics across IoU thresholds
python tools/analysis_tools/analyze_logs.py plot_curve \
    work_dirs/faster-rcnn_final/*/vis_data/*.json \
    --keys bbox_mAP bbox_mAP_50 bbox_mAP_75 bbox_mAP_m bbox_mAP_l \
    --out bbox_mAP_all.jpg

# Classification accuracy
python tools/analysis_tools/analyze_logs.py plot_curve \
    work_dirs/faster-rcnn_final/*/vis_data/*.json \
    --keys acc --out acc.jpg
```

### Confusion Matrix

```bash
python tools/analysis_tools/confusion_matrix.py \
    configs/DETEX/faster-rcnn_final.py \
    ./test.pkl \
    ./ --show
```

### Visualize Detection Errors

```bash
python tools/analysis_tools/visualize_errors.py \
    configs/DETEX/faster-rcnn_final.py \
    ./test.pkl \
    ./error-results/bbox
```

---

## Experimental Results

### Comparison with State-of-the-Art Methods

| Method | AR(50-95) | AP(50-95) | AP50 | AP75 | APm | APl |
|---|---|---|---|---|---|---|
| RetinaNet | 0.597 | 0.479 | 0.916 | 0.444 | 0.503 | 0.478 |
| SSD | 0.488 | 0.339 | 0.790 | 0.219 | 0.246 | 0.341 |
| DINO | 0.609 | 0.509 | 0.940 | 0.480 | 0.526 | 0.509 |
| YOLO11n | — | 0.409 | 0.685 | 0.443 | — | — |
| YOLO11s | — | 0.513 | 0.838 | **0.554** | — | — |
| Ours (Faster R-CNN Baseline) | 0.589 | 0.498 | 0.935 | 0.462 | 0.516 | 0.498 |
| Ours (Data Augmentation) | 0.593 | 0.504 | 0.943 | 0.459 | 0.508 | 0.505 |
| Ours (Anchors) | 0.600 | 0.510 | 0.942 | 0.486 | 0.554 | 0.509 |
| Ours (Swin-Transformer+FPN) | 0.601 | 0.505 | 0.942 | 0.471 | **0.561** | 0.506 |
| Ours (Prior Knowledge) | 0.593 | 0.507 | 0.940 | 0.480 | 0.540 | 0.507 |
| **Ours (All Methods)** | **0.613** | **0.526** | **0.961** | 0.503 | 0.557 | **0.520** |

> **Bold** values indicate best performance in each column. Our full model achieves the highest AR(50-95), AP(50-95), AP50, and APl.

### Ablation Study Summary

Each component contributes incrementally to the final performance:

| Component Added | AP(50-95) Gain | Key Effect |
|---|---|---|
| Data Augmentation | +0.006 over baseline | Improves robustness to contrast & artifacts |
| Dynamic Anchors | +0.006 over aug | Better adapts to diverse tooth morphologies (APm +4.6%) |
| Swin-Transformer + FPN | +0.001 over anchors | Best single-component APm (0.561), captures long-range dependencies |
| Prior Knowledge Loss | +0.003 over anchors | Improves AP75 (0.480), excludes rootless structures |
| **All Combined** | **+0.028 over baseline** | Synergistic effect exceeds linear sum of individual gains |

### Key Findings

- **AP50 = 96.1%**: New state-of-the-art at standard IoU threshold, surpassing DINO (94.0%) and YOLO11s (83.8%).
- **Tooth diversity handling**: Dynamic anchors and weighted loss effectively distinguish natural teeth from implants, crowns, and fixed partial dentures.
- **Image quality robustness**: Swin-Transformer + hybrid augmentation handle low-contrast regions (posterior mandible) and metal artifacts.

---

## Project Structure

```
mmdetection/
├── configs/DETEX/                 # All DENTEX model configs (core contribution)
│   ├── faster-rcnn_final.py       # ★ Final model (all improvements)
│   ├── faster-rcnn_Baseline.py    #   Baseline Faster R-CNN
│   ├── faster-rcnn_data_augmentation.py
│   ├── faster-rcnn_set_anchors.py
│   ├── faster-rcnnn_swin_fpn_1x_DENTEX.py
│   ├── faster-rcnn_iou.py         #   Prior knowledge weighted loss
│   ├── retinanet_r50_fpn_DENTEX.py
│   ├── ssd_300_DENTEX.py
│   ├── dino_swin_l_DENTEX.py
│   ├── inferencer.py              #   Inference script
│   └── divide_data.ipynb          #   Data splitting notebook
├── mmdet/                         # Core MMDetection library
│   └── models/
│       ├── losses/
│       │   └── smooth_l1_loss.py  # CustomSmoothL1Loss (weighted loss impl.)
│       ├── backbones/
│       │   └── swin.py            # Swin Transformer backbone
│       └── detectors/
│           └── faster_rcnn.py     # Faster R-CNN detector
├── tools/                         # Training, testing, analysis scripts
│   ├── dist_train.sh              # Distributed training launcher
│   ├── test.py                    # Model testing
│   └── analysis_tools/            # Visualization & analysis utilities
├── data/DENTEX/                   # Dataset (not tracked by git)
│   └── new_enumeration_data/
├── work_dirs/                     # Training outputs (not tracked by git)
├── pre-trained-models/            # Pre-trained weights (not tracked by git)
├── requirements/                  # Split dependency files
├── setup.py                       # Package installation
└── README.md                      # This file
```

---

## Repository Contents

This repository contains only the core code needed to reproduce our experiments. Large files (datasets, model weights, training outputs) are excluded via `.gitignore` and must be obtained separately.

| Directory / File | Description |
|---|---|
| `configs/DETEX/` | All model configurations — **core contribution** |
| `configs/DETEX/divide_data.ipynb` | Data splitting and FDI label conversion notebook |
| `mmdet/` | Modified MMDetection library (losses, backbones, detectors) |
| `tools/` | Training, testing, and analysis scripts |
| `requirements/`, `requirements.txt` | Python dependencies |
| `setup.py`, `setup.cfg`, `MANIFEST.in` | Package installation |
| `README.md` | Project documentation |
| `CITATION.cff` | Citation metadata |
| `LICENSE` | Apache 2.0 License |

**Not included** (must be downloaded separately): DENTEX dataset, pre-trained backbone weights, trained model checkpoints, training outputs.

---

## License & Citation

This project is built upon [MMDetection](https://github.com/open-mmlab/mmdetection) by OpenMMLab, released under the Apache License 2.0.

If you use this work in your research, please cite:

```bibtex
@article{wang2025tooth,
  title={Tooth detection and numbering in panoramic radiographs using enhanced Faster R-CNN: addressing tooth diversity and image quality challenges},
  author={Wang, Shunyi and Yu, Xin and Feng, Zhien and Wang, Shuping and Liu, Peize and Li, Pengchao and Zhang, Shu and Liu, Jianghua},
  journal={},
  year={2025}
}
```

For questions or issues, please open an issue on GitHub or contact the corresponding author.
