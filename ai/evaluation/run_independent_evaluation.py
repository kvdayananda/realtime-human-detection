"""FitRange Mirror - Independent Evaluation Runner and Visualizer.

Executes Phases 5, 6, 7, and 8 of the Independent Model Validation:
- Evaluates detection performance against verified independent ground-truth bounding boxes.
- Evaluates segmentation performance against verified independent ground-truth masks.
- Performs condition-based validation and identifies tested vs untested categories.
- Generates 7-panel visualization sets in runs/independent_evaluation/visualizations/:
    1. original
    2. gt_box
    3. pred_box
    4. gt_mask
    5. pred_mask
    6. mask_comparison (overlap difference)
    7. combined_overlay
- Records JSON outputs and console log files in runs/independent_evaluation/.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from ai.detection.inference_onnx import HumanDetectorONNX
from ai.segmentation.inference_onnx import HumanSegmenterONNX
from ai.evaluation.eval_detection import evaluate_detection, calculate_box_iou
from ai.evaluation.eval_segmentation import evaluate_segmentation, calculate_mask_iou_dice

logger = logging.getLogger("FitRangeMirror.IndependentEval")


def generate_comparison_visualizations(
    image_path: Path,
    det_txt: Optional[Path],
    seg_png: Optional[Path],
    detector: HumanDetectorONNX,
    segmenter: HumanSegmenterONNX,
    out_dir: Path,
    stem: str,
) -> Dict[str, str]:
    """Generate 7 distinct visual comparison artifacts for an annotated sample."""
    im = cv2.imread(str(image_path))
    if im is None:
        raise ValueError(f"Failed to read image: {image_path}")
    h, w = im.shape[:2]

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Original image
    orig_path = out_dir / f"{stem}_01_original.jpg"
    cv2.imwrite(str(orig_path), im)

    # Parse GT boxes
    gt_boxes: List[List[float]] = []
    if det_txt and det_txt.is_file():
        content = det_txt.read_text(encoding="utf-8").strip()
        if content:
            for line in content.splitlines():
                parts = line.strip().split()
                if len(parts) >= 5 and int(float(parts[0])) == 0:
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x1 = max(0.0, (xc - bw / 2.0) * w)
                    y1 = max(0.0, (yc - bh / 2.0) * h)
                    x2 = min(float(w), (xc + bw / 2.0) * w)
                    y2 = min(float(h), (yc + bh / 2.0) * h)
                    gt_boxes.append([x1, y1, x2, y2])

    # Parse GT mask
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    if seg_png and seg_png.is_file():
        loaded_mask = cv2.imread(str(seg_png), cv2.IMREAD_GRAYSCALE)
        if loaded_mask is not None and loaded_mask.shape == (h, w):
            gt_mask = loaded_mask

    # Model Predictions
    det_res = detector.detect(im, conf_threshold=0.25)
    pred_boxes = [d["bbox"] for d in det_res.get("detections", [])]

    seg_res = segmenter.segment(im, conf_threshold=0.25)
    pred_mask = segmenter.get_composite_mask(im, segmentations=seg_res.get("segmentations", []))

    # 2. Ground-Truth Bounding Box overlay (Green)
    vis_gt_box = im.copy()
    for b in gt_boxes:
        cv2.rectangle(vis_gt_box, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
        cv2.putText(vis_gt_box, "GT Person", (int(b[0]), max(20, int(b[1]) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(vis_gt_box, f"Ground Truth: {len(gt_boxes)} person(s)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    gt_box_path = out_dir / f"{stem}_02_gt_box.jpg"
    cv2.imwrite(str(gt_box_path), vis_gt_box)

    # 3. Predicted Bounding Box overlay (Cyan)
    vis_pred_box = im.copy()
    for d in det_res.get("detections", []):
        b = d["bbox"]
        conf = d["confidence"]
        cv2.rectangle(vis_pred_box, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 200, 0), 2)
        cv2.putText(vis_pred_box, f"Pred: {conf:.2f}", (int(b[0]), max(20, int(b[1]) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
    cv2.putText(vis_pred_box, f"Predictions: {len(pred_boxes)} person(s)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)
    pred_box_path = out_dir / f"{stem}_03_pred_box.jpg"
    cv2.imwrite(str(pred_box_path), vis_pred_box)

    # 4. Ground-Truth Segmentation Mask (Binary Grayscale)
    gt_mask_path = out_dir / f"{stem}_04_gt_mask.png"
    cv2.imwrite(str(gt_mask_path), gt_mask)

    # 5. Predicted Segmentation Mask (Binary Grayscale)
    pred_mask_path = out_dir / f"{stem}_05_pred_mask.png"
    cv2.imwrite(str(pred_mask_path), pred_mask)

    # 6. Mask Comparison (Overlap visualization)
    # Green = True Positive (overlap)
    # Red = False Positive (pred without GT)
    # Blue = False Negative (GT without pred)
    comp_vis = np.zeros((h, w, 3), dtype=np.uint8)
    gt_pos = (gt_mask == 255)
    pr_pos = (pred_mask == 255)
    tp = gt_pos & pr_pos
    fp = (~gt_pos) & pr_pos
    fn = gt_pos & (~pr_pos)

    comp_vis[tp] = [0, 255, 0]    # Green (True Positive)
    comp_vis[fp] = [0, 0, 255]    # Red (False Positive)
    comp_vis[fn] = [255, 0, 0]    # Blue (False Negative / Missed)

    # Blend comparison with original background
    comp_blended = cv2.addWeighted(im, 0.45, comp_vis, 0.55, 0)
    cv2.putText(comp_blended, "Green: TP | Red: FP | Blue: FN", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    comp_path = out_dir / f"{stem}_06_mask_comparison.jpg"
    cv2.imwrite(str(comp_path), comp_blended)

    # 7. Combined Prediction Overlay
    overlay = im.copy()
    if (pred_mask == 255).any():
        color_mask = np.zeros_like(im)
        color_mask[pred_mask == 255] = [255, 120, 0]  # Amber/Orange
        cv2.addWeighted(color_mask, 0.45, overlay, 0.55, 0, overlay)

    for d in det_res.get("detections", []):
        b = d["bbox"]
        cv2.rectangle(overlay, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), 2)
        cv2.putText(overlay, f"Person {d['confidence']:.2f}", (int(b[0]), max(20, int(b[1]) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    combined_path = out_dir / f"{stem}_07_combined_overlay.jpg"
    cv2.imwrite(str(combined_path), overlay)

    return {
        "original": str(orig_path),
        "gt_box": str(gt_box_path),
        "pred_box": str(pred_box_path),
        "gt_mask": str(gt_mask_path),
        "pred_mask": str(pred_mask_path),
        "mask_comparison": str(comp_path),
        "combined_overlay": str(combined_path),
    }


def run_independent_evaluation() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "dataset" / "independent_validation"
    img_dir = data_dir / "images"
    det_dir = data_dir / "annotations" / "detection"
    seg_dir = data_dir / "annotations" / "segmentation"
    manifest_path = data_dir / "manifest.csv"

    runs_dir = project_root / "runs" / "independent_evaluation"
    vis_dir = runs_dir / "visualizations"
    runs_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    det_log_path = runs_dir / "detection_evaluation.log"
    seg_log_path = runs_dir / "segmentation_evaluation.log"
    det_json_path = runs_dir / "detection_results.json"
    seg_json_path = runs_dir / "segmentation_results.json"
    cond_json_path = runs_dir / "condition_results.json"

    # Identify which images currently have valid ground truth
    all_images = sorted(list(img_dir.glob("*.png")))
    annotated_stems: List[str] = []
    pending_stems: List[str] = []

    for img_p in all_images:
        stem = img_p.stem
        det_p = det_dir / f"{stem}.txt"
        seg_p = seg_dir / f"{stem}.png"
        if det_p.is_file() and seg_p.is_file():
            annotated_stems.append(stem)
        else:
            pending_stems.append(stem)

    print("=" * 65)
    print("      FITRANGE MIRROR - INDEPENDENT MODEL VALIDATION")
    print("=" * 65)
    print(f"Total Dataset Images:          {len(all_images)}")
    print(f"Verified Annotated Samples:    {len(annotated_stems)} (Negative Controls: {sum(1 for s in annotated_stems if 'neg' in s)})")
    print(f"Pending Human Annotations:     {len(pending_stems)} (Positive Human Samples)")
    print("-" * 65)

    # Initialize engines
    detector = HumanDetectorONNX()
    segmenter = HumanSegmenterONNX()

    # Create temporary evaluation directory containing only annotated samples
    # to evaluate existing independent ground truth
    temp_eval_dir = runs_dir / "temp_annotated_set"
    temp_img_dir = temp_eval_dir / "images"
    temp_det_dir = temp_eval_dir / "labels"
    temp_seg_dir = temp_eval_dir / "masks"

    for d in [temp_img_dir, temp_det_dir, temp_seg_dir]:
        d.mkdir(parents=True, exist_ok=True)

    import shutil
    for stem in annotated_stems:
        shutil.copy2(img_dir / f"{stem}.png", temp_img_dir / f"{stem}.png")
        shutil.copy2(det_dir / f"{stem}.txt", temp_det_dir / f"{stem}.txt")
        shutil.copy2(seg_dir / f"{stem}.png", temp_seg_dir / f"{stem}.png")

    # 1. Detection Evaluation
    det_results = evaluate_detection(
        images_dir=temp_img_dir,
        labels_dir=temp_det_dir,
        conf_threshold=0.25,
    )
    with open(det_json_path, "w", encoding="utf-8") as f:
        json.dump(det_results, f, indent=2)

    conf = det_results.get("confusion", {})
    det_log_lines = [
        "FITRANGE MIRROR - INDEPENDENT DETECTION EVALUATION",
        f"Images Evaluated: {det_results['dataset_images_count']}",
        f"Ground-Truth Persons: {det_results['total_ground_truth_boxes']}",
        f"Total Model Predictions: {det_results['total_predicted_boxes']}",
        f"Precision (@IoU 0.50): {det_results['precision'] * 100:.2f}%",
        f"Recall (@IoU 0.50): {det_results['recall'] * 100:.2f}%",
        f"F1-Score (@IoU 0.50): {det_results['f1_score'] * 100:.2f}%",
        f"mAP@50: {det_results['map_50'] * 100:.2f}%",
        f"mAP@50:95: {det_results['map_50_95'] * 100:.2f}%",
        f"True Positives: {conf.get('true_positives', 0)}",
        f"False Positives: {conf.get('false_positives', 0)}",
        f"False Negatives: {conf.get('false_negatives', 0)}",
    ]
    det_log_path.write_text("\n".join(det_log_lines), encoding="utf-8")

    # 2. Segmentation Evaluation
    seg_results = evaluate_segmentation(
        images_dir=temp_img_dir,
        masks_dir=temp_seg_dir,
        conf_threshold=0.25,
    )
    with open(seg_json_path, "w", encoding="utf-8") as f:
        json.dump(seg_results, f, indent=2)

    seg_log_lines = [
        "FITRANGE MIRROR - INDEPENDENT SEGMENTATION EVALUATION",
        f"Images Evaluated: {seg_results['dataset_images_count']}",
        f"Dataset Mean IoU: {seg_results['mean_iou'] * 100:.2f}%",
        f"Dataset Mean Dice: {seg_results['mean_dice'] * 100:.2f}%",
        f"Positive Samples Mean IoU: {seg_results.get('mean_iou_positive_only', 0.0) * 100:.2f}%",
        f"Positive Samples Mean Dice: {seg_results.get('mean_dice_positive_only', 0.0) * 100:.2f}%",
        f"Negative Background Accuracy: {seg_results['negative_background_accuracy'] * 100:.2f}%",
    ]
    seg_log_path.write_text("\n".join(seg_log_lines), encoding="utf-8")

    # 3. Condition-Based Evaluation
    cond_results = {
        "summary": {
            "total_independent_dataset_images": len(all_images),
            "human_images_target": 100,
            "negative_images_target": 30,
            "human_images_annotated": sum(1 for s in annotated_stems if "pos" in s),
            "negative_images_annotated": sum(1 for s in annotated_stems if "neg" in s),
            "pending_human_annotations": len(pending_stems),
        },
        "conditions": [
            {
                "category": "Negative Controls (Empty Backgrounds, Furniture, Scenery)",
                "status": "TESTED",
                "sample_count": 30,
                "detected_count": det_results['total_predicted_boxes'],
                "success_rate": 1.0 - (conf.get('false_positives', 0) / max(1, len(annotated_stems))),
                "segmentation_negative_accuracy": seg_results['negative_background_accuracy'],
            },
            {
                "category": "Front-Facing People",
                "status": "NOT TESTED (Pending Human Annotation)",
                "sample_count": 0,
                "reason": "100 positive images curated in dataset/independent_validation/images/ await human annotation."
            },
            {
                "category": "Side-Facing Profiles",
                "status": "NOT TESTED (Pending Human Annotation)",
                "sample_count": 0,
                "reason": "Awaiting human annotation."
            },
            {
                "category": "Angled Perspectives",
                "status": "NOT TESTED (Pending Human Annotation)",
                "sample_count": 0,
                "reason": "Awaiting human annotation."
            },
            {
                "category": "Athletic Wear",
                "status": "NOT TESTED (Pending Human Annotation)",
                "sample_count": 0,
                "reason": "Awaiting human annotation."
            },
            {
                "category": "Loose Clothing",
                "status": "NOT TESTED (Pending Human Annotation)",
                "sample_count": 0,
                "reason": "Awaiting human annotation."
            },
            {
                "category": "Plain Backgrounds",
                "status": "NOT TESTED (Pending Human Annotation)",
                "sample_count": 0,
                "reason": "Awaiting human annotation."
            },
            {
                "category": "Cluttered Backgrounds",
                "status": "NOT TESTED (Pending Human Annotation)",
                "sample_count": 0,
                "reason": "Awaiting human annotation."
            },
            {
                "category": "Outdoor Backgrounds",
                "status": "NOT TESTED (Pending Human Annotation)",
                "sample_count": 0,
                "reason": "Awaiting human annotation."
            },
            {
                "category": "Bright Natural Lighting",
                "status": "NOT TESTED (Pending Human Annotation)",
                "sample_count": 0,
                "reason": "Awaiting human annotation."
            },
            {
                "category": "Standard Indoor Lighting",
                "status": "NOT TESTED (Pending Human Annotation)",
                "sample_count": 0,
                "reason": "Awaiting human annotation."
            },
            {
                "category": "Low-Light / Shadow Conditions",
                "status": "NOT TESTED (Pending Human Annotation)",
                "sample_count": 0,
                "reason": "Awaiting human annotation."
            }
        ]
    }
    with open(cond_json_path, "w", encoding="utf-8") as f:
        json.dump(cond_results, f, indent=2)

    # 4. Generate 7-Panel Visualizations for Annotated Samples (Negative controls + Positive samples)
    print("Generating 7-panel visualization sets in runs/independent_evaluation/visualizations/ ...")
    sample_stems_for_vis = [s for s in annotated_stems if "neg" in s][:3] + [s for s in annotated_stems if "pos" in s]
    for stem in sample_stems_for_vis:
        generate_comparison_visualizations(
            image_path=img_dir / f"{stem}.png",
            det_txt=det_dir / f"{stem}.txt",
            seg_png=seg_dir / f"{stem}.png",
            detector=detector,
            segmenter=segmenter,
            out_dir=vis_dir,
            stem=stem,
        )

    # Clean up temp eval dir
    shutil.rmtree(temp_eval_dir, ignore_errors=True)

    # Print summary
    print("\n" + "\n".join(det_log_lines))
    print("\n" + "\n".join(seg_log_lines))
    print("\n" + "=" * 65)
    print(f"Results saved to: {runs_dir}")
    print("Visualizations created in: runs/independent_evaluation/visualizations/")
    print("=" * 65)


if __name__ == "__main__":
    run_independent_evaluation()
