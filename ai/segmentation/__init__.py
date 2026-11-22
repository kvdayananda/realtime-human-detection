"""FitRange Mirror - Human Segmentation Module.

This package provides standalone human instance segmentation capabilities:
- `HumanSegmenter`: PyTorch-based segmentation using Ultralytics YOLO11n-seg.
- `HumanSegmenterONNX`: Pure ONNX Runtime segmentation without PyTorch runtime dependency.
- `export_onnx`: Utility to export PyTorch segmentation weights to optimized ONNX format.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .export_onnx import export_onnx
    from .inference_onnx import HumanSegmenterONNX
    from .segment import HumanSegmenter

__all__ = [
    "HumanSegmenter",
    "HumanSegmenterONNX",
    "export_onnx",
]


def __getattr__(name: str) -> Any:
    """Lazy-load submodules to allow fast ONNX imports and clean CLI execution."""
    if name == "HumanSegmenter":
        from .segment import HumanSegmenter
        return HumanSegmenter
    if name == "HumanSegmenterONNX":
        from .inference_onnx import HumanSegmenterONNX
        return HumanSegmenterONNX
    if name == "export_onnx":
        from .export_onnx import export_onnx
        return export_onnx
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
