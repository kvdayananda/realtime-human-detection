# FitRange Mirror - Human Instance Segmentation Module

This directory contains the standalone **Human Instance Segmentation Model** implementation for the FitRange Mirror project.

It provides both PyTorch (Ultralytics YOLO11n-seg) and pure ONNX Runtime inference engines with independent weights, inference code, and ONNX export, without requiring live webcam access, ByteTrack, or MediaPipe.

---

## 1. Model Architecture

- **Architecture:** Ultralytics YOLO11 Nano Segmentation (`yolo11n-seg`)
- **Task:** Instance Segmentation (`segment`)
- **Target Class:** Filtered strictly to `class_id: 0` (`person`)
- **Model Parameters:** ~2.87M parameters (113 layers, 9.8 GFLOPs)
- **Base Resolution:** 640×640 px (configurable)
- **Mask Output:** 32 prototype masks ($160 \times 160$) combined with 32 instance mask coefficients

---

## 2. Checkpoint Locations

| Checkpoint Type | Path | Purpose | Size |
| :--- | :--- | :--- | :--- |
| **PyTorch Checkpoint** | `models/human_segmentation/human_segmentation.pt` | Primary PyTorch segmentation weights | ~6.2 MB |
| **ONNX Optimized Model**| `models/human_segmentation/human_segmentation.onnx` | Standalone edge runtime inference | ~11.2 MB |

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

### Output Schema
The module outputs a structured dictionary containing instance segmentation details:

```json
{
  "segmentations": [
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.9542,
      "class_id": 0,
      "class_name": "person",
      "polygon": [[x1, y1], [x2, y2], "..."],
      "mask_area": 25997
    }
  ],
  "detections": [...]
}
```

- `bbox`: Bounding box corner coordinates `[x1, y1, x2, y2]` in original image pixel space.
- `confidence`: Detection confidence score (between `0.0` and `1.0`).
- `class_id`: Integer class ID (`0` for person).
- `class_name`: String label (`"person"`).
- `polygon`: List of `[x, y]` coordinates tracing the boundary contour of the segmented person in original image coordinates.
- `mask_area`: Total pixel area occupied by the person mask.
- `detections`: Aliased to `segmentations` for drop-in interoperability with detection consumers.
- When no person is detected, returns `{"segmentations": [], "detections": []}`.

---

## 5. Usage Guide

### 5.1 PyTorch Segmentation (`HumanSegmenter`)

#### Python API:
```python
from ai.segmentation import HumanSegmenter

# Initialize segmenter
segmenter = HumanSegmenter(
    model_path="models/human_segmentation/human_segmentation.pt",
    conf_threshold=0.35,
    imgsz=640,
    device="cpu"  # or "cuda"
)

# Run segmentation
results = segmenter.segment("path/to/image.jpg")
print(results)

# Generate full-resolution binary mask (0 or 255)
mask = segmenter.get_composite_mask("path/to/image.jpg")
```

#### CLI Command:
```bash
python -m ai.segmentation.segment --image "dataset/human detection dataset/1/0.png" --conf 0.35
```

---

### 5.2 ONNX Export (`export_onnx.py`)

Exports the PyTorch segmentation weights into an optimized `.onnx` model:

#### Python API:
```python
from ai.segmentation import export_onnx

onnx_path = export_onnx(
    model_path="models/human_segmentation/human_segmentation.pt",
    output_path="models/human_segmentation/human_segmentation.onnx",
    imgsz=640,
    opset=12
)
print("Exported to:", onnx_path)
```

#### CLI Command:
```bash
python -m ai.segmentation.export_onnx --weights models/human_segmentation/human_segmentation.pt --output models/human_segmentation/human_segmentation.onnx --imgsz 640
```

---

### 5.3 ONNX Runtime Inference (`HumanSegmenterONNX`)

Performs high-performance instance segmentation using `onnxruntime` without loading PyTorch or Ultralytics:

#### Python API:
```python
from ai.segmentation import HumanSegmenterONNX

segmenter = HumanSegmenterONNX(
    model_path="models/human_segmentation/human_segmentation.onnx",
    conf_threshold=0.35,
    iou_threshold=0.45,
    imgsz=640
)

# Run segmentation
results = segmenter.segment("path/to/image.jpg")
print(results)

# Generate binary composite mask (uint8, 0/255)
mask = segmenter.get_composite_mask("path/to/image.jpg")
```

#### CLI Command:
```bash
python -m ai.segmentation.inference_onnx --image "dataset/human detection dataset/1/0.png" --conf 0.35
```

---

## 6. Preprocessing & Mask Synthesis Pipeline

1. **Letterbox Preprocessing:**
   - Image is scaled proportionally with neutral gray padding `(114, 114, 114)` to $640 \times 640$.
   - Scale factor: $r = \min(640 / W_{\text{orig}}, 640 / H_{\text{orig}})$.
2. **Dual-Tensor Output:**
   - **`output0`** $(1, 116, 8400)$: 4 box coordinates + 80 class scores + 32 mask coefficients.
   - **`output1`** $(1, 32, 160, 160)$: 32 prototype mask feature maps.
3. **Instance Mask Synthesis:**
   - Linear combination of prototype feature maps weighted by the 32 mask coefficients:
     $$M_{\text{raw}} = \sum_{i=1}^{32} c_i \cdot P_i$$
   - Sigmoid activation: $M_{\text{prob}} = \sigma(M_{\text{raw}})$.
   - Upscaled bilinearly to letterbox canvas and cropped to bounding box.
   - Unpadded and resized to original $(W_{\text{orig}}, H_{\text{orig}})$ resolution.
   - Binary threshold at $0.5$ followed by contour polygon extraction (`cv2.findContours`).

---

## 7. Running Unit Tests

Execute the comprehensive segmentation test suite:

```bash
python tests/test_human_segmentation.py -v
```
