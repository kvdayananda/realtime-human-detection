"""Unit tests for Evaluation Suite (ai/evaluation).

Covers:
- calculate_box_iou accuracy and edge cases
- compute_ap COCO 101-point interpolation calculation
- calculate_mask_iou_dice accuracy, zero overlap, and empty mask cases
- End-to-end evaluate_detection execution on annotated benchmark dataset
- End-to-end evaluate_segmentation execution on annotated benchmark dataset
- End-to-end condition validation execution
"""

import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from ai.evaluation.eval_detection import (
    calculate_box_iou,
    compute_ap,
    evaluate_detection,
)
from ai.evaluation.eval_segmentation import (
    calculate_mask_iou_dice,
    evaluate_segmentation,
)
from ai.evaluation.condition_validation import run_condition_validation


class TestEvaluationSuite(unittest.TestCase):
    """Test suite for evaluation algorithms and benchmark runners."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.val_images = cls.project_root / "dataset" / "val_annotated" / "images"
        cls.val_labels = cls.project_root / "dataset" / "val_annotated" / "labels"
        cls.val_masks = cls.project_root / "dataset" / "val_annotated" / "masks"

    # ----------------------------------------------------
    # 1. Bounding Box IoU Calculation
    # ----------------------------------------------------

    def test_01_box_iou_perfect_match(self):
        """Test IoU of identical bounding boxes is exactly 1.0."""
        box = [10.0, 20.0, 100.0, 200.0]
        iou = calculate_box_iou(box, box)
        self.assertAlmostEqual(iou, 1.0, places=5)

    def test_02_box_iou_disjoint(self):
        """Test IoU of completely non-overlapping boxes is 0.0."""
        box1 = [0.0, 0.0, 10.0, 10.0]
        box2 = [20.0, 20.0, 30.0, 30.0]
        iou = calculate_box_iou(box1, box2)
        self.assertEqual(iou, 0.0)

    def test_03_box_iou_partial_overlap(self):
        """Test IoU on known partial overlap geometry."""
        # Box1: 0..10, 0..10 (area=100)
        # Box2: 5..15, 0..10 (area=100)
        # Intersection: 5..10, 0..10 (area=50)
        # Union: 100 + 100 - 50 = 150
        # IoU: 50 / 150 = 1/3 ~ 0.3333
        box1 = [0.0, 0.0, 10.0, 10.0]
        box2 = [5.0, 0.0, 15.0, 10.0]
        iou = calculate_box_iou(box1, box2)
        self.assertAlmostEqual(iou, 1.0 / 3.0, places=4)

    # ----------------------------------------------------
    # 2. Average Precision (AP) Calculation
    # ----------------------------------------------------

    def test_04_compute_ap_perfect(self):
        """Test AP calculation for perfect precision across all recall levels."""
        recalls = np.linspace(0.1, 1.0, 10)
        precisions = np.ones(10)
        ap = compute_ap(recalls, precisions)
        self.assertAlmostEqual(ap, 1.0, places=4)

    def test_05_compute_ap_empty(self):
        """Test AP calculation with empty arrays returns 0.0."""
        ap = compute_ap(np.array([]), np.array([]))
        self.assertEqual(ap, 0.0)

    # ----------------------------------------------------
    # 3. Mask IoU and Dice Score Calculation
    # ----------------------------------------------------

    def test_06_mask_iou_dice_identical(self):
        """Test IoU and Dice for identical masks are 1.0."""
        m = np.zeros((50, 50), dtype=np.uint8)
        m[10:30, 10:30] = 255
        iou, dice = calculate_mask_iou_dice(m, m)
        self.assertEqual(iou, 1.0)
        self.assertEqual(dice, 1.0)

    def test_07_mask_iou_dice_both_empty(self):
        """Test IoU and Dice when both ground truth and prediction are empty is 1.0."""
        m_empty = np.zeros((50, 50), dtype=np.uint8)
        iou, dice = calculate_mask_iou_dice(m_empty, m_empty)
        self.assertEqual(iou, 1.0)
        self.assertEqual(dice, 1.0)

    def test_08_mask_iou_dice_one_empty(self):
        """Test IoU and Dice when one mask is empty while the other is non-empty is 0.0."""
        m_empty = np.zeros((50, 50), dtype=np.uint8)
        m_filled = np.zeros((50, 50), dtype=np.uint8)
        m_filled[10:20, 10:20] = 255

        iou1, dice1 = calculate_mask_iou_dice(m_empty, m_filled)
        self.assertEqual(iou1, 0.0)
        self.assertEqual(dice1, 0.0)

        iou2, dice2 = calculate_mask_iou_dice(m_filled, m_empty)
        self.assertEqual(iou2, 0.0)
        self.assertEqual(dice2, 0.0)

    def test_09_mask_iou_dice_partial_overlap(self):
        """Test mask IoU and Dice calculation on known partial overlap geometry."""
        m1 = np.zeros((10, 10), dtype=np.uint8)
        m2 = np.zeros((10, 10), dtype=np.uint8)
        m1[:6, :] = 255  # 60 pixels
        m2[2:8, :] = 255  # 60 pixels
        # Intersection: rows 2..6 = 40 pixels
        # Union: rows 0..8 = 80 pixels
        # IoU = 40 / 80 = 0.50
        # Dice = 2 * 40 / (60 + 60) = 80 / 120 = 2/3 ~ 0.6667
        iou, dice = calculate_mask_iou_dice(m1, m2)
        self.assertEqual(iou, 0.5)
        self.assertAlmostEqual(dice, 2.0 / 3.0, places=4)

    # ----------------------------------------------------
    # 4. End-to-End Detection Evaluation Execution
    # ----------------------------------------------------

    def test_10_evaluate_detection_benchmark(self):
        """Test running detection evaluation on the annotated benchmark dataset."""
        results = evaluate_detection(
            images_dir=self.val_images,
            labels_dir=self.val_labels,
            conf_threshold=0.25,
        )

        self.assertIn("precision", results)
        self.assertIn("recall", results)
        self.assertIn("map_50", results)
        self.assertIn("map_50_95", results)
        self.assertEqual(results["dataset_images_count"], 11)
        self.assertGreaterEqual(results["map_50"], 0.90)
        self.assertGreaterEqual(results["map_50_95"], 0.85)

    # ----------------------------------------------------
    # 5. End-to-End Segmentation Evaluation Execution
    # ----------------------------------------------------

    def test_11_evaluate_segmentation_benchmark(self):
        """Test running segmentation evaluation on the annotated benchmark dataset."""
        results = evaluate_segmentation(
            images_dir=self.val_images,
            masks_dir=self.val_masks,
            conf_threshold=0.25,
        )

        self.assertIn("mean_iou", results)
        self.assertIn("mean_dice", results)
        self.assertIn("negative_background_accuracy", results)
        self.assertEqual(results["dataset_images_count"], 11)
        self.assertGreaterEqual(results["mean_iou"], 0.90)
        self.assertGreaterEqual(results["mean_dice"], 0.90)
        self.assertEqual(results["negative_background_accuracy"], 1.0)

    # ----------------------------------------------------
    # 6. Condition-Based Operational Validation Execution
    # ----------------------------------------------------

    def test_12_condition_validation_execution(self):
        """Test running condition-based validation across all categories."""
        res = run_condition_validation(enable_pose=False)
        summary = res["summary"]
        self.assertEqual(summary["total_conditions_audited"], 15)
        self.assertEqual(summary["conditions_tested"], 12)
        self.assertEqual(summary["conditions_untested"], 3)
        self.assertEqual(summary["conditions_passed"], 12)
        self.assertEqual(summary["conditions_failed"], 0)


if __name__ == "__main__":
    unittest.main()
