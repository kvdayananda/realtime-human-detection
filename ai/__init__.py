"""FitRange Mirror AI package."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .detection import HumanDetector, HumanDetectorONNX
    from .evaluation import evaluate_detection, evaluate_segmentation
    from .pipeline import FitRangeMirrorPipeline
    from .pose import PoseEstimator
    from .segmentation import HumanSegmenter, HumanSegmenterONNX

__all__ = [
    "HumanDetector",
    "HumanDetectorONNX",
    "HumanSegmenter",
    "HumanSegmenterONNX",
    "PoseEstimator",
    "FitRangeMirrorPipeline",
    "evaluate_detection",
    "evaluate_segmentation",
]


def __getattr__(name: str) -> Any:
    if name in ("HumanDetector", "HumanDetectorONNX"):
        from . import detection
        return getattr(detection, name)
    if name in ("HumanSegmenter", "HumanSegmenterONNX"):
        from . import segmentation
        return getattr(segmentation, name)
    if name == "PoseEstimator":
        from . import pose
        return getattr(pose, name)
    if name == "FitRangeMirrorPipeline":
        from . import pipeline
        return getattr(pipeline, name)
    if name in ("evaluate_detection", "evaluate_segmentation"):
        from . import evaluation
        return getattr(evaluation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
