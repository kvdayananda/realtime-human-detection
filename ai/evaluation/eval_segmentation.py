"""FitRange Mirror - Human Instance Segmentation Evaluation Suite.

Computes standard image segmentation metrics:
- Intersection over Union (IoU / Jaccard Index): Area(Pred ∩ GT) / Area(Pred ∪ GT)
- Dice Score (Sørensen–Dice Coefficient / F1-Score): 2 * Area(Pred ∩ GT) / (Area(Pred) + Area(GT))
- Mean IoU (mIoU) and Mean Dice Score across the evaluation dataset.
- Separate positive-sample metrics and true-negative accuracy on background frames.

Supports both ONNX Runtime (`HumanSegmenterONNX`) and PyTorch (`HumanSegmenter`).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from ai.segmentation.inference_onnx import HumanSegmenterONNX

logger = logging.getLogger("FitRangeMirror.EvalSegmentation")


def calculate_mask_iou_dice(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> Tuple[float, float]:
    """Calculate Intersection over Union (IoU) and Dice Score between two binary masks.

    Args:
        pred_mask: Predicted binary mask (2D array, non-zero values treated as foreground).
        gt_mask: Ground-truth binary mask (2D array, non-zero values treated as foreground).

    Returns:
        Tuple of (iou, dice_score) as floats in range [0.0, 1.0].
    """
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(
            f"Shape mismatch between predicted mask {pred_mask.shape} and ground truth mask {gt_mask.shape}"
        )

    p_bool = pred_mask > 0
    g_bool = gt_mask > 0

    p_sum = int(np.count_nonzero(p_bool))
    g_sum = int(np.count_nonzero(g_bool))

    # Case 1: Both masks are completely empty (True Negative background)
    if p_sum == 0 and g_sum == 0:
        return 1.0, 1.0

    # Case 2: One mask is empty while the other is non-empty (False Positive or False Negative)
    if p_sum == 0 or g_sum == 0:
        return 0.0, 0.0

    intersection = int(np.count_nonzero(np.logical_and(p_bool, g_bool)))
    union = int(np.count_nonzero(np.logical_or(p_bool, g_bool)))

    iou = float(intersection / max(union, 1))
    dice = float((2.0 * intersection) / max(p_sum + g_sum, 1))

    return round(iou, 4), round(dice, 4)


def evaluate_segmentation(
    images_dir: Union[str, Path],
    masks_dir: Union[str, Path],
    model_path: Optional[Union[str, Path]] = None,
    conf_threshold: float = 0.25,
    imgsz: int = 640,
) -> Dict[str, Any]:
    """Evaluate Human Instance Segmentation model on images and ground-truth masks.

    Args:
        images_dir: Directory containing evaluation images.
        masks_dir: Directory containing matching binary ground-truth mask PNG files.
        model_path: Path to segmentation .onnx model.
        conf_threshold: Confidence threshold for predictions.
        imgsz: Inference square dimension.

    Returns:
        Dictionary containing overall metrics (mean_iou, mean_dice, positive metrics,
        negative accuracy) and per-image details.
    """
    segmenter = HumanSegmenterONNX(model_path=model_path, conf_threshold=conf_threshold, imgsz=imgsz)

    img_dir_path = Path(images_dir).resolve()
    msk_dir_path = Path(masks_dir).resolve()

    if not img_dir_path.is_dir():
        raise FileNotFoundError(f"Images directory not found: {img_dir_path}")
    if not msk_dir_path.is_dir():
        raise FileNotFoundError(f"Masks directory not found: {msk_dir_path}")

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    image_paths = sorted([p for p in img_dir_path.iterdir() if p.suffix.lower() in valid_exts])
    if not image_paths:
        raise ValueError(f"No valid images found in: {img_dir_path}")

    all_ious: List[float] = []
    all_dices: List[float] = []
    positive_ious: List[float] = []
    positive_dices: List[float] = []
    negative_matches: int = 0
    negative_total: int = 0
    per_image_results: List[Dict[str, Any]] = []

    for img_path in image_paths:
        stem = img_path.stem
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Failed to load image: %s", img_path)
            continue
        h, w = img.shape[:2]

        # Load ground truth mask
        mask_file = msk_dir_path / f"{stem}.png"
        if not mask_file.is_file():
            # Try other extensions
            for ext in [".jpg", ".bmp"]:
                alt_file = msk_dir_path / f"{stem}{ext}"
                if alt_file.is_file():
                    mask_file = alt_file
                    break

        if not mask_file.is_file():
            raise FileNotFoundError(f"Ground-truth mask file missing for {stem}: looked at {mask_file}")

        gt_mask_raw = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if gt_mask_raw is None:
            raise ValueError(f"Failed to read mask image from: {mask_file}")

        if gt_mask_raw.shape != (h, w):
            gt_mask_raw = cv2.resize(gt_mask_raw, (w, h), interpolation=cv2.INTER_NEAREST)

        # Run segmentation inference
        res = segmenter.segment(img, conf_threshold=conf_threshold)
        segmentations = res.get("segmentations", [])
        pred_mask = segmenter.get_composite_mask(img, segmentations)

        # Calculate IoU and Dice
        iou, dice = calculate_mask_iou_dice(pred_mask, gt_mask_raw)
        all_ious.append(iou)
        all_dices.append(dice)

        gt_has_person = bool(np.any(gt_mask_raw > 0))
        pred_has_person = bool(len(segmentations) > 0)

        if gt_has_person:
            positive_ious.append(iou)
            positive_dices.append(dice)
        else:
            negative_total += 1
            if not pred_has_person and iou == 1.0:
                negative_matches += 1

        per_image_results.append(
            {
                "image": img_path.name,
                "resolution": [w, h],
                "gt_has_person": gt_has_person,
                "pred_has_person": pred_has_person,
                "person_instances": len(segmentations),
                "iou": iou,
                "dice": dice,
                "gt_person_pixels": int(np.count_nonzero(gt_mask_raw > 0)),
                "pred_person_pixels": int(np.count_nonzero(pred_mask > 0)),
            }
        )

    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    mean_dice = float(np.mean(all_dices)) if all_dices else 0.0
    mean_pos_iou = float(np.mean(positive_ious)) if positive_ious else 0.0
    mean_pos_dice = float(np.mean(positive_dices)) if positive_dices else 0.0
    neg_accuracy = float(negative_matches / max(negative_total, 1))

    return {
        "dataset_images_count": len(image_paths),
        "positive_images_count": len(positive_ious),
        "negative_images_count": negative_total,
        "mean_iou": round(mean_iou, 4),
        "mean_dice": round(mean_dice, 4),
        "mean_iou_positive_only": round(mean_pos_iou, 4),
        "mean_dice_positive_only": round(mean_pos_dice, 4),
        "negative_background_accuracy": round(neg_accuracy, 4),
        "image_details": per_image_results,
    }


def main() -> None:
    """CLI runner for Human Instance Segmentation evaluation."""
    parser = argparse.ArgumentParser(description="FitRange Mirror - Human Segmentation Evaluation")
    parser.add_argument("--images", type=str, required=True, help="Path to evaluation images directory.")
    parser.add_argument("--masks", type=str, required=True, help="Path to ground-truth binary masks directory.")
    parser.add_argument("--model", type=str, default=None, help="Path to segmentation .onnx model.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25).")
    parser.add_argument("--json", type=str, default=None, help="Optional path to save results as JSON.")
    args = parser.parse_args()

    results = evaluate_segmentation(
        images_dir=args.images,
        masks_dir=args.masks,
        model_path=args.model,
        conf_threshold=args.conf,
    )

    print("\n" + "=" * 60)
    print("     FITRANGE MIRROR - HUMAN SEGMENTATION EVALUATION")
    print("=" * 60)
    print(f"Total Images Evaluated:        {results['dataset_images_count']}")
    print(f"Positive Images (Humans):      {results['positive_images_count']}")
    print(f"Negative Images (Backgrounds): {results['negative_images_count']}")
    print("-" * 60)
    print(f"Dataset Mean IoU (mIoU):       {results['mean_iou'] * 100:.2f}%")
    print(f"Dataset Mean Dice Score:       {results['mean_dice'] * 100:.2f}%")
    print(f"Positive Samples Mean IoU:     {results['mean_iou_positive_only'] * 100:.2f}%")
    print(f"Positive Samples Mean Dice:    {results['mean_dice_positive_only'] * 100:.2f}%")
    print(f"Negative Background Accuracy:  {results['negative_background_accuracy'] * 100:.2f}%")
    print("=" * 60 + "\n")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved evaluation metrics to: {args.json}")


if __name__ == "__main__":
    main()
