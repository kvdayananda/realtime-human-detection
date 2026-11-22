# FitRange Mirror - Human Detection Module

This directory contains the standalone **Human Detection Model** implementation for the FitRange Mirror project.

It provides both PyTorch (Ultralytics YOLO) and pure ONNX Runtime inference engines without requiring live webcam access, ByteTrack, or MediaPipe.

---

## 1. Model Architecture

- **Architecture:** Ultralytics YOLO11 Nano (`yolo11n`)
- **Task:** Object Detection (`detect`)
- **Target Class:** Filtered strictly to `class_id: 0` (`person`)
- **Model Parameters:** ~2.6M parameters (100 layers, 6.5 GFLOPs)
- **Base Resolution:** 640×640 px (configurable)

---

## 2. Checkpoint Locations

| Checkpoint Type | Path | Purpose |
| :--- | :--- | :--- |
| **PyTorch Checkpoint** | `models/human_detection/human_detection.pt` | Primary PyTorch detection weights |
| **ONNX Optimized Model**| `models/human_detection/human_detection.onnx` | Standalone edge runtime inference |
| **Original Source Weights** | `models/yolo11n.pt` | Unmodified base source checkpoint |

---

## 3. Installation & Dependencies

Ensure the required Python packages are installed:

```bash
pip install ultralytics torch torchvision onnx onnxruntime opencv-python numpy
```

---

## 4. Input & Output Formats

### Supported Input Formats
- **Image File Path:** `str` or `pathlib.Path` pointing to a valid image file (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`).
- **In-Memory Array:** `numpy.ndarray` of shape `(H, W, 3)` (BGR or RGB) or `(H, W)` (grayscale).

### Expected Output Format
The module outputs a dictionary adhering strictly to the FitRange detection schema:

```json
{
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.9542,
      "class_id": 0,
      "class_name": "person"
    }
  ]
}
```

- `bbox`: Bounding box corner coordinates `[x1, y1, x2, y2]` mapped to the original image pixel coordinates.
- `confidence`: Detection confidence score (between `0.0` and `1.0`).
- `class_id`: Integer class ID (`0` for person).
- `class_name`: String label (`"person"`).
- When no person is detected, the function returns `{"detections": []}`.

---

## 5. Usage Guide

### 5.1 PyTorch Inference (`HumanDetector`)

#### Python API:
```python
from ai.detection import HumanDetector

# Initialize detector
detector = HumanDetector(
    model_path="models/human_detection/human_detection.pt",
    conf_threshold=0.35,
    imgsz=640,
    device="cpu"  # or "cuda"
)

# Run detection on image path
results = detector.detect("path/to/image.jpg")
print(results)

# Run detection on in-memory NumPy array
import cv2
frame = cv2.imread("path/to/image.jpg")
results = detector.detect(frame)
```

#### CLI Command:
```bash
python -m ai.detection.detect --image dataset/human\ detection\ dataset/1/0.png --conf 0.35
```

---

### 5.2 ONNX Export (`export_onnx.py`)

Exports the PyTorch `.pt` model to an optimized `.onnx` file:

#### Python API:
```python
from ai.detection import export_onnx

onnx_path = export_onnx(
    model_path="models/human_detection/human_detection.pt",
    output_path="models/human_detection/human_detection.onnx",
    imgsz=640,
    opset=12
)
print("Exported to:", onnx_path)
```

#### CLI Command:
```bash
python -m ai.detection.export_onnx --weights models/human_detection/human_detection.pt --output models/human_detection/human_detection.onnx --imgsz 640
```

---

### 5.3 ONNX Runtime Inference (`HumanDetectorONNX`)

Runs high-performance inference using `onnxruntime` without loading PyTorch or Ultralytics:

#### Python API:
```python
from ai.detection import HumanDetectorONNX

detector = HumanDetectorONNX(
    model_path="models/human_detection/human_detection.onnx",
    conf_threshold=0.35,
    iou_threshold=0.45,
    imgsz=640
)

# Run detection
results = detector.detect("path/to/image.jpg")
print(results)
```

#### CLI Command:
```bash
python -m ai.detection.inference_onnx --image dataset/human\ detection\ dataset/1/0.png --conf 0.35
```

---

## 6. Preprocessing & Coordinate Mapping

To ensure accurate bounding boxes regardless of input image dimensions:

1. **Letterbox Resize:**
   - The image is scaled proportionally to fit within `640×640` preserving the aspect ratio:
     $$r = \min\left(\frac{640}{W_{\text{orig}}}, \frac{640}{H_{\text{orig}}}\right)$$
   - The remaining area is symmetrically padded with neutral gray `(114, 114, 114)`:
     $$dx = \frac{640 - \text{round}(W_{\text{orig}} \cdot r)}{2}, \quad dy = \frac{640 - \text{round}(H_{\text{orig}} \cdot r)}{2}$$

2. **Inverse Projection (Postprocessing):**
   - Predicted bounding box coordinates $(x, y)$ in the 640×640 space are projected back to the original image frame:
     $$x_{\text{orig}} = \frac{x - dx}{r}, \quad y_{\text{orig}} = \frac{y - dy}{r}$$
   - Coordinates are clamped to the original image bounds $[0, W_{\text{orig}}]$ and $[0, H_{\text{orig}}]$.

---

## 7. Running Unit Tests

Execute the comprehensive test suite:

```bash
python tests/test_human_detection.py -v
```
