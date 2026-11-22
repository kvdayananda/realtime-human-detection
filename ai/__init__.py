"""FitRange Mirror AI package."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .detection import HumanDetector, HumanDetectorONNX
    from .segmentation import HumanSegmenter, HumanSegmenterONNX

__all__ = [
    "HumanDetector",
    "HumanDetectorONNX",
    "HumanSegmenter",
    "HumanSegmenterONNX",
]


def __getattr__(name: str) -> Any:
    if name in ("HumanDetector", "HumanDetectorONNX"):
        from . import detection
        return getattr(detection, name)
    if name in ("HumanSegmenter", "HumanSegmenterONNX"):
        from . import segmentation
        return getattr(segmentation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
