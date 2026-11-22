"""FitRange Mirror - Human Detection Evaluation Suite.

Computes standard object detection metrics:
- Precision
- Recall
- mAP@50 (Mean Average Precision at IoU 0.50)
- mAP@50:95 (COCO metric averaged over 10 IoU thresholds from 0.50 to 0.95 with step 0.05)

Supports both ONNX Runtime (`HumanDetectorONNX`) and PyTorch (`HumanDetector`).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from ai.detection.inference_onnx import HumanDetectorONNX

logger = logging.getLogger("FitRangeMirror.EvalDetection")


def calculate_box_iou(
    box1: Union[List[float], Tuple[float, ...], np.ndarray],
    box2: Union[List[float], Tuple[float, ...], np.ndarray],
) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes in xyxy format.

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        float IoU value in range [0.0, 1.0].
    """
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))

    union = area1 + area2 - inter_area
    if union <= 1e-9:
        return 0.0

    return inter_area / union


def compute_ap(
    recalls: np.ndarray,
    precisions: np.ndarray,
) -> float:
    """Compute Average Precision using the 101-point COCO interpolation method.

    Args:
        recalls: 1D array of recall values in [0.0, 1.0].
        precisions: 1D array of precision values in [0.0, 1.0].

    Returns:
        float Average Precision in range [0.0, 1.0].
    """
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0

    # Append sentinel values
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute the precision envelope (ensure monotonically decreasing from right)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # 101-point interpolation: evaluate precision at recall levels [0.00, 0.01, ..., 1.00]
    recall_levels = np.linspace(0.0, 1.0, 101)
    interp_precisions = []
    for r in recall_levels:
        mask = mrec >= r
        if np.any(mask):
            interp_precisions.append(mpre[mask].max())
        else:
            interp_precisions.append(0.0)

    return float(np.mean(interp_precisions))


def parse_yolo_label(
    label_path: Union[str, Path],
    img_width: int,
    img_height: int,
) -> List[List[float]]:
    """Parse YOLO format label file (class_id x_center y_center width height normalized) into xyxy pixel coordinates.

    Args:
        label_path: Path to .txt label file.
        img_width: Image pixel width.
        img_height: Image pixel height.

    Returns:
        List of [x1, y1, x2, y2] bounding boxes in pixel space for class 0 (person).
    """
    path = Path(label_path)
    if not path.is_file():
        return []

    boxes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            if cls_id != 0:
                # Class 0 is person
                continue
            xc, yc, w, h = map(float, parts[1:5])
            x1 = max(0.0, (xc - w / 2.0) * img_width)
            y1 = max(0.0, (yc - h / 2.0) * img_height)
            x2 = min(float(img_width), (xc + w / 2.0) * img_width)
            y2 = min(float(img_height), (yc + h / 2.0) * img_height)
            boxes.append([x1, y1, x2, y2])
    return boxes


def evaluate_detection(
    images_dir: Union[str, Path],
    labels_dir: Union[str, Path],
    model_path: Optional[Union[str, Path]] = None,
    conf_threshold: float = 0.25,
    imgsz: int = 640,
    iou_thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Evaluate Human Detection model on a directory of images and ground-truth labels.

    Args:
        images_dir: Directory containing evaluation images (.png, .jpg, .jpeg).
        labels_dir: Directory containing matching YOLO label files (.txt).
        model_path: Path to detection .onnx model.
        conf_threshold: Confidence threshold for predictions.
        imgsz: Inference square dimension.
        iou_thresholds: List of IoU thresholds for evaluation (default: 0.50 to 0.95 with step 0.05).

    Returns:
        Dictionary containing overall metrics (precision, recall, mAP@50, mAP@50:95)
        and per-image breakdown.
    """
    if iou_thresholds is None:
        iou_thresholds = [round(0.50 + 0.05 * i, 2) for i in range(10)]

    detector = HumanDetectorONNX(model_path=model_path, conf_threshold=conf_threshold, imgsz=imgsz)

    img_dir_path = Path(images_dir).resolve()
    lbl_dir_path = Path(labels_dir).resolve()

    if not img_dir_path.is_dir():
        raise FileNotFoundError(f"Images directory not found: {img_dir_path}")

    # Gather images
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    image_paths = sorted([p for p in img_dir_path.iterdir() if p.suffix.lower() in valid_exts])
    if not image_paths:
        raise ValueError(f"No valid images found in: {img_dir_path}")

    total_gt_boxes = 0
    all_predictions: List[Dict[str, Any]] = []
    ground_truths: Dict[str, List[List[float]]] = {}
    image_details: List[Dict[str, Any]] = []

    for img_path in image_paths:
        stem = img_path.stem
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Failed to load image: %s", img_path)
            continue
        h, w = img.shape[:2]

        # Parse ground-truth boxes
        lbl_file = lbl_dir_path / f"{stem}.txt"
        gt_boxes = parse_yolo_label(lbl_file, w, h)
        ground_truths[stem] = gt_boxes
        total_gt_boxes += len(gt_boxes)

        # Run model inference
        res = detector.detect(img, conf_threshold=conf_threshold)
        detections = res.get("detections", [])

        for det in detections:
            all_predictions.append(
                {
                    "image_stem": stem,
                    "bbox": det["bbox"],
                    "confidence": float(det["confidence"]),
                }
            )

        image_details.append(
            {
                "image": img_path.name,
                "resolution": [w, h],
                "gt_count": len(gt_boxes),
                "pred_count": len(detections),
                "pred_confidences": [round(d["confidence"], 3) for d in detections],
            }
        )

    # Sort all detections across the dataset by confidence descending
    all_predictions.sort(key=lambda x: x["confidence"], reverse=True)

    # Compute TP and FP matches at each IoU threshold
    ap_per_iou: Dict[float, float] = {}
    metrics_at_50: Dict[str, Any] = {}

    for iou_thresh in iou_thresholds:
        # Track which GT boxes have already been matched per image
        matched_gt: Dict[str, set] = {stem: set() for stem in ground_truths}
        tp = np.zeros(len(all_predictions))
        fp = np.zeros(len(all_predictions))

        for idx, pred in enumerate(all_predictions):
            stem = pred["image_stem"]
            pbox = pred["bbox"]
            gt_boxes = ground_truths.get(stem, [])

            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt_boxes):
                iou = calculate_box_iou(pbox, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_thresh and best_gt_idx >= 0:
                if best_gt_idx not in matched_gt[stem]:
                    tp[idx] = 1.0
                    matched_gt[stem].add(best_gt_idx)
                else:
                    # Duplicate detection on already matched ground truth
                    fp[idx] = 1.0
            else:
                fp[idx] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)

        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        recalls = cum_tp / max(total_gt_boxes, 1)

        ap = compute_ap(recalls, precisions)
        ap_per_iou[iou_thresh] = round(ap, 4)

        if abs(iou_thresh - 0.50) < 1e-4:
            total_tp = int(np.sum(tp))
            total_fp = int(np.sum(fp))
            total_fn = total_gt_boxes - total_tp
            p_final = float(total_tp / max(total_tp + total_fp, 1e-9))
            r_final = float(total_tp / max(total_gt_boxes, 1))

            metrics_at_50 = {
                "precision": round(p_final, 4),
                "recall": round(r_final, 4),
                "true_positives": total_tp,
                "false_positives": total_fp,
                "false_negatives": total_fn,
                "f1_score": round(
                    2 * (p_final * r_final) / max(p_final + r_final, 1e-9), 4
                ),
            }

    map_50 = ap_per_iou.get(0.50, 0.0)
    map_50_95 = round(float(np.mean(list(ap_per_iou.values()))), 4)

    return {
        "dataset_images_count": len(image_paths),
        "total_ground_truth_boxes": total_gt_boxes,
        "total_predicted_boxes": len(all_predictions),
        "precision": metrics_at_50.get("precision", 0.0),
        "recall": metrics_at_50.get("recall", 0.0),
        "f1_score": metrics_at_50.get("f1_score", 0.0),
        "map_50": map_50,
        "map_50_95": map_50_95,
        "ap_per_iou": ap_per_iou,
        "confusion": metrics_at_50,
        "image_details": image_details,
    }


def main() -> None:
    """CLI runner for Human Detection evaluation."""
    parser = argparse.ArgumentParser(description="FitRange Mirror - Human Detection Evaluation")
    parser.add_argument("--images", type=str, required=True, help="Path to evaluation images directory.")
    parser.add_argument("--labels", type=str, required=True, help="Path to YOLO ground-truth labels directory.")
    parser.add_argument("--model", type=str, default=None, help="Path to detection .onnx model.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25).")
    parser.add_argument("--json", type=str, default=None, help="Optional path to save results as JSON.")
    args = parser.parse_args()

    results = evaluate_detection(
        images_dir=args.images,
        labels_dir=args.labels,
        model_path=args.model,
        conf_threshold=args.conf,
    )

    print("\n" + "=" * 60)
    print("      FITRANGE MIRROR - HUMAN DETECTION EVALUATION")
    print("=" * 60)
    print(f"Total Images Evaluated:        {results['dataset_images_count']}")
    print(f"Total Ground-Truth Persons:    {results['total_ground_truth_boxes']}")
    print(f"Total Model Predictions:       {results['total_predicted_boxes']}")
    print("-" * 60)
    print(f"Precision (@IoU=0.50):         {results['precision'] * 100:.2f}%")
    print(f"Recall    (@IoU=0.50):         {results['recall'] * 100:.2f}%")
    print(f"F1-Score  (@IoU=0.50):         {results['f1_score'] * 100:.2f}%")
    print(f"mAP@50:                        {results['map_50'] * 100:.2f}%")
    print(f"mAP@50:95 (COCO Metric):       {results['map_50_95'] * 100:.2f}%")
    print("-" * 60)
    print("Per-IoU Threshold Average Precision:")
    for iou, ap in results["ap_per_iou"].items():
        print(f"  IoU @ {iou:.2f}: AP = {ap * 100:.2f}%")
    print("=" * 60 + "\n")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved evaluation metrics to: {args.json}")


if __name__ == "__main__":
    main()
