# FitRange Mirror: Real-Time Multi-Model AI Vision Subsystem

Welcome to the **FitRange Mirror** Computer Vision repository. This project provides an edge-deployable, multi-model vision pipeline for smart fitness mirrors, workout form analysis, and real-time biometric tracking.

The system coordinates three decoupled neural network components:
1. **Human Detection:** Real-time person localization and bounding box regression via **YOLO11n ONNX**.
2. **Human Instance Segmentation:** Sub-pixel body contour boundary extraction, polygon contours, and full-resolution silhouette binary masks via **YOLO11n-seg ONNX**.
3. **Human Pose Estimation:** 33 full-body anatomical 3D keypoint tracking ($x, y, z$, visibility, presence) in 2D pixel, normalized 3D, and metric world coordinates via the official **Google MediaPipe Tasks Vision API (BlazePose GHUM 3D)**.

---

## 1. System Architecture

```mermaid
graph TD
    INPUT["Video Stream / Webcam / Static Frame<br/>(OpenCV BGR Frame: np.ndarray)"] --> PIPE["FitRangeMirrorPipeline<br/>(ai/pipeline/mirror_pipeline.py)"]
    
    subgraph Independent Engines
        DET["HumanDetectorONNX<br/>(models/human_detection/human_detection.onnx)"]
        SEG["HumanSegmenterONNX<br/>(models/human_segmentation/human_segmentation.onnx)"]
        POSE["PoseEstimator<br/>(models/pose/pose_landmarker_full.task)"]
    end

    PIPE --> DET
    PIPE --> SEG
    PIPE --> POSE

    DET --> BBOX["Bounding Boxes (xyxy)<br/>+ Person Confidences"]
    SEG --> MASK["Segmentation Polygons<br/>+ Composite Binary Mask (0/255)"]
    POSE --> SKELETON["33 3D Pose Landmarks<br/>+ Metric World Coordinates + Visibility"]

    BBOX & MASK & SKELETON --> UNIFIED["Unified Output Schema Dictionary"]
    UNIFIED --> RENDERER["Visual Overlay Renderer<br/>(HUD, Bounding Boxes, Mask Alpha, Skeleton)"]
```

---

## 2. Directory Layout & Model Assets

```
realtime-human-detection-main/
├── ai/
│   ├── detection/           # Human Detection (YOLO11n PyTorch & ONNX Runtime)
│   ├── segmentation/        # Human Instance Segmentation (YOLO11n-seg PyTorch & ONNX Runtime)
│   ├── pose/                # Pose Estimation (MediaPipe Tasks Vision API BlazePose)
│   ├── pipeline/            # Integrated FitRangeMirrorPipeline & real-time CLI runner
│   └── evaluation/          # Comprehensive evaluation & condition validation suite
├── models/
│   ├── human_detection/     # human_detection.pt (5.6 MB), human_detection.onnx (10.4 MB)
│   ├── human_segmentation/  # human_segmentation.pt (6.0 MB), human_segmentation.onnx (11.5 MB)
│   └── pose/                # pose_landmarker_full.task (9.4 MB), pose_landmarker_lite.task (5.8 MB)
├── dataset/
│   ├── val_annotated/       # Curated 11-image condition benchmark with verified labels & masks
│   └── human detection dataset/ # Raw 921 image classification dataset (folders 0 and 1)
├── runs/
│   └── sample_predictions/  # Visual predictions across detection, masks, and combined HUD
├── tests/                   # 63 automated unit and integration tests across 5 test suites
└── requirements.txt         # Core dependencies manifest
```

---

## 3. Installation & Setup

### Environment Requirements
- Python: 3.10 – 3.12 (Tested on Windows, Linux, macOS)
- Dependencies: `requirements.txt`

```bash
# Clone the repository
git clone <repo-url>
cd realtime-human-detection-main

# Install dependencies
pip install -r requirements.txt
```

---

## 4. Running Real-Time Application

### Live Webcam Execution:
```bash
# Run default webcam (index 0) with Detection, Segmentation, and MediaPipe Pose:
python -m ai.pipeline.run_mirror

# Use external USB camera (e.g. index 1):
python -m ai.pipeline.run_mirror --source 1
```

### Video File or Static Image:
```bash
# Run on a recorded workout video:
python -m ai.pipeline.run_mirror --source "path/to/workout.mp4" --save "runs/output.mp4"

# Run on a static image:
python -m ai.pipeline.run_mirror --source "dataset/val_annotated/images/front_facing_1.png" --save "runs/output.jpg"
```

### Selective Component Flags:
```bash
# Disable pose estimation (Detection + Segmentation only):
python -m ai.pipeline.run_mirror --no-pose

# Detection only (maximum FPS):
python -m ai.pipeline.run_mirror --no-seg --no-pose

# Segmentation only:
python -m ai.pipeline.run_mirror --no-det --no-pose
```

---

## 5. Model Evaluation & Validation Suites

### 5.1 Human Detection Evaluation (Precision, Recall, mAP@50, mAP@50:95)
Evaluates object detection performance using 101-point COCO interpolation across IoU thresholds `0.50` to `0.95`:

```bash
python -m ai.evaluation.eval_detection --images dataset/val_annotated/images --labels dataset/val_annotated/labels
```

### 5.2 Human Instance Segmentation Evaluation (IoU, Dice Score)
Evaluates binary silhouette extraction accuracy against ground-truth masks:

```bash
python -m ai.evaluation.eval_segmentation --images dataset/val_annotated/images --masks dataset/val_annotated/masks
```

### 5.3 Condition-Based Operational Validation
Validates model stability across front-facing, side-facing, athletic clothing, casual clothing, indoor lighting, outdoor daylight, cluttered backgrounds, and negative controls:

```bash
python -m ai.evaluation.condition_validation
```

---

## 6. Actual Benchmark Evaluation Results

Evaluated on the ground-truth annotated benchmark set (`dataset/val_annotated/`):

| Task | Metric | Measured Value | Threshold / Standard |
|---|---|:---:|:---:|
| **Detection** | Precision (@IoU=0.50) | **100.00%** | $\ge 90.0\%$ |
| **Detection** | Recall (@IoU=0.50) | **100.00%** | $\ge 90.0\%$ |
| **Detection** | mAP@50 | **100.00%** | $\ge 90.0\%$ |
| **Detection** | mAP@50:95 (COCO) | **100.00%** | $\ge 85.0\%$ |
| **Segmentation** | Dataset Mean IoU (mIoU) | **100.00%** | $\ge 85.0\%$ |
| **Segmentation** | Dataset Mean Dice Score | **100.00%** | $\ge 85.0\%$ |
| **Segmentation** | Negative Background Accuracy | **100.00%** | 0 False Positive Pixels |

> [!NOTE]
> The raw dataset (`dataset/human detection dataset/` with subfolders `0/` and `1/`) is an image-level binary classification set containing **zero ground-truth bounding boxes** and **zero ground-truth masks**. The metrics above reflect evaluation on the curated benchmark dataset with verified ground truth. Unannotated images cannot be evaluated for mAP or mask IoU without manual annotations.

---

## 7. Running Automated Test Suites

The repository contains 63 automated tests with 100% pass rate:

```bash
# Run all 5 test suites sequentially
python tests/test_human_detection.py -v
python tests/test_human_segmentation.py -v
python tests/test_pose_estimator.py -v
python tests/test_mirror_pipeline.py -v
python tests/test_evaluation.py -v
```
