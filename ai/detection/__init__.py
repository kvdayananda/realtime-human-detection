"""FitRange Mirror - Human Detection Module.

This package provides standalone human detection capabilities for FitRange Mirror:
- `HumanDetector`: PyTorch-based inference using Ultralytics YOLO11n.
- `HumanDetectorONNX`: Pure ONNX Runtime inference without PyTorch dependency.
- `export_onnx`: Utility to export PyTorch weights to optimized ONNX format.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .detect import HumanDetector
    from .export_onnx import export_onnx
    from .inference_onnx import HumanDetectorONNX

__all__ = [
    "HumanDetector",
    "HumanDetectorONNX",
    "export_onnx",
]


def __getattr__(name: str) -> Any:
    """Lazy-load submodules to allow fast ONNX imports and clean CLI execution."""
    if name == "HumanDetector":
        from .detect import HumanDetector
        return HumanDetector
    if name == "HumanDetectorONNX":
        from .inference_onnx import HumanDetectorONNX
        return HumanDetectorONNX
    if name == "export_onnx":
        from .export_onnx import export_onnx
        return export_onnx
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
