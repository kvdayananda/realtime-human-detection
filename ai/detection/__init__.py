"""FitRange Mirror - Human Detection Module.

This package provides standalone human detection capabilities for FitRange Mirror:
- `HumanDetector`: PyTorch-based inference using Ultralytics YOLO11n.
- `HumanDetectorONNX`: Pure ONNX Runtime inference without PyTorch dependency.
- `export_onnx`: Utility to export PyTorch weights to optimized ONNX format.
"""

from .detect import HumanDetector
from .export_onnx import export_onnx
from .inference_onnx import HumanDetectorONNX

__all__ = [
    "HumanDetector",
    "HumanDetectorONNX",
    "export_onnx",
]
