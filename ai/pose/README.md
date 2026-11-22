# FitRange Mirror - Human Pose Estimation Module (MediaPipe Tasks API)

This directory provides the standalone **Human Pose Estimation Model** for the FitRange Mirror project, powered by the modern **Google MediaPipe Tasks Vision API** (`mediapipe.tasks.python.vision.PoseLandmarker`).

It operates independently from the ONNX detection and segmentation models, extracting 33 full-body anatomical 3D keypoints with visibility scores, presence scores, and real-world metric 3D coordinates.

---

## 1. Architecture & Landmarks

- **Framework:** MediaPipe Tasks Vision API (`PoseLandmarker`)
- **Model Topology:** BlazePose GHUM 3D
- **Model Assets (`.task`):**
  - `models/pose/pose_landmarker_full.task` (9.4 MB, Default / Balanced)
  - `models/pose/pose_landmarker_lite.task` (5.8 MB, Lightweight / Edge)
- **Topologies Supported:** Front-facing, side-profile, standing, seated, and dynamic athletic movement
- **Landmark Topology Map (33 Keypoints):**
  - `0 - 10`: Facial landmarks (nose, eyes, ears, mouth)
  - `11 - 16`: Upper body joints (shoulders, elbows, wrists)
  - `17 - 22`: Hand keypoints (pinkies, indices, thumbs)
  - `23 - 28`: Torso and leg joints (hips, knees, ankles)
  - `29 - 32`: Foot keypoints (heels, foot indices)
- **Skeleton Connections:** 35 standard anatomical connections rendered directly with OpenCV without legacy `mp.solutions.drawing_utils`.

---

## 2. Model Asset Acquisition & Location

The official Google MediaPipe Pose Landmarker models are bundled in `.task` format:

| Model Variant | Local Path | Download URL |
|---|---|---|
| **Full (Recommended)** | `models/pose/pose_landmarker_full.task` | `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task` |
| **Lite (Edge/Fast)** | `models/pose/pose_landmarker_lite.task` | `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task` |
| **Heavy (High Accuracy)** | `models/pose/pose_landmarker_heavy.task` | `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task` |

### Manual Download Command:
```bash
python -c "
import urllib.request
from pathlib import Path
target_dir = Path('models/pose')
target_dir.mkdir(parents=True, exist_ok=True)
url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task'
urllib.request.urlretrieve(url, target_dir / 'pose_landmarker_full.task')
"
```

---

## 3. Input & Output Formats

### Input Formats
- NumPy array (`np.ndarray`) of shape `(H, W, 3)` in BGR format (standard OpenCV frame). BGR is converted internally to RGB before building `mp.Image`.
- Image file path (`str` or `pathlib.Path`).
- Supports both static images (`RunningMode.IMAGE`) and streaming video (`RunningMode.VIDEO`) with strictly monotonic timestamps.

### Output Schema
The `estimate()` method returns:

```json
{
  "has_pose": true,
  "landmarks": [
    {
      "id": 0,
      "name": "nose",
      "x": 0.186,
      "y": 0.128,
      "z": 0.158,
      "visibility": 0.9995,
      "pixel_x": 101,
      "pixel_y": 70
    }
  ],
  "world_landmarks": [
    {
      "id": 0,
      "name": "nose",
      "x": -0.052,
      "y": -0.584,
      "z": -0.218,
      "visibility": 0.9995
    }
  ]
}
```

When no person or pose is detected:
```json
{
  "has_pose": false,
  "landmarks": [],
  "world_landmarks": []
}
```

---

## 4. Usage Guide

### Python API:
```python
import cv2
from ai.pose import PoseEstimator

# Initialize pose estimator with modern Tasks API
with PoseEstimator(
    model_asset_path="models/pose/pose_landmarker_full.task",
    static_image_mode=False,  # Video stream mode with monotonic timestamps
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as estimator:
    # Run on camera frame
    frame = cv2.imread("dataset/human detection dataset/1/0.png")
    result = estimator.estimate(frame)

    if result["has_pose"]:
        print(f"Extracted {len(result['landmarks'])} keypoints")
        # Draw skeleton overlay
        annotated = estimator.draw_pose(frame, result)
        cv2.imwrite("pose_overlay.jpg", annotated)
```

### CLI Command:
```bash
python -m ai.pose.estimator --image "dataset/human detection dataset/1/0.png"
```

---

## 5. Running Unit Tests

```bash
python tests/test_pose_estimator.py -v
```
