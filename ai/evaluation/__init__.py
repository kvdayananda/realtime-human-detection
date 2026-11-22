"""FitRange Mirror - Computer Vision Model Evaluation Package.

Provides comprehensive evaluation suites for:
- Human Detection: Precision, Recall, mAP@50, mAP@50:95
- Human Segmentation: Intersection over Union (IoU), Dice Score
- Condition-based validation across diverse postures, clothing, lighting, and backgrounds
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "calculate_box_iou",
    "compute_ap",
    "evaluate_detection",
    "calculate_mask_iou_dice",
    "evaluate_segmentation",
]


def __getattr__(name: str) -> Any:
    if name in {"calculate_box_iou", "compute_ap", "evaluate_detection"}:
        from ai.evaluation.eval_detection import (
            calculate_box_iou,
            compute_ap,
            evaluate_detection,
        )
        return {
            "calculate_box_iou": calculate_box_iou,
            "compute_ap": compute_ap,
            "evaluate_detection": evaluate_detection,
        }[name]

    if name in {"calculate_mask_iou_dice", "evaluate_segmentation"}:
        from ai.evaluation.eval_segmentation import (
            calculate_mask_iou_dice,
            evaluate_segmentation,
        )
        return {
            "calculate_mask_iou_dice": calculate_mask_iou_dice,
            "evaluate_segmentation": evaluate_segmentation,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
