"""FitRange Mirror - Human Pose Estimation Module.

Provides body landmark detection and skeleton estimation using
Google MediaPipe Pose Landmarker.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .estimator import POSE_CONNECTIONS, POSE_LANDMARK_NAMES, PoseEstimator

__all__ = [
    "PoseEstimator",
    "POSE_LANDMARK_NAMES",
    "POSE_CONNECTIONS",
]


def __getattr__(name: str) -> Any:
    """Lazy-load pose estimator attributes."""
    if name in ("PoseEstimator", "POSE_LANDMARK_NAMES", "POSE_CONNECTIONS"):
        from . import estimator
        return getattr(estimator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
