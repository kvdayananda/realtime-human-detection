"""Unit tests for FitRange Mirror - Human Segmentation Module.

Covers:
- Model loading (PyTorch and ONNX)
- Image segmentation (file path and NumPy array)
- Empty segmentation handling on negative images
- Segmentation output schema and polygon validation
- Binary composite mask generation
- ONNX model loading and inference
- Cross-engine consistency (PyTorch vs ONNX Runtime)
"""

import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from ai.segmentation.inference_onnx import HumanSegmenterONNX
from ai.segmentation.segment import HumanSegmenter


class TestHumanSegmentation(unittest.TestCase):
    """Test suite for PyTorch and ONNX human instance segmentation modules."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.pt_model_path = (
            cls.project_root
            / "models"
            / "human_segmentation"
            / "human_segmentation.pt"
        )
        cls.onnx_model_path = (
            cls.project_root
            / "models"
            / "human_segmentation"
            / "human_segmentation.onnx"
        )

        cls.positive_image_path = (
            cls.project_root
            / "dataset"
            / "human detection dataset"
            / "1"
            / "0.png"
        )
        cls.blank_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # ----------------------------------------------------
    # 1. PyTorch Segmentation Model Loading Tests
    # ----------------------------------------------------

    def test_01_pytorch_model_loading_success(self):
        """Test loading PyTorch segmentation model with default and explicit paths."""
        segmenter = HumanSegmenter(model_path=self.pt_model_path)
        self.assertIsNotNone(segmenter._model)
        self.assertEqual(segmenter.PERSON_CLASS_ID, 0)
        self.assertEqual(segmenter.PERSON_CLASS_NAME, "person")

    def test_02_pytorch_model_loading_missing_file(self):
        """Test that non-existent model path raises FileNotFoundError."""
        bogus_path = self.project_root / "models" / "missing_seg.pt"
        with self.assertRaises(FileNotFoundError):
            HumanSegmenter(model_path=bogus_path)

    def test_03_pytorch_invalid_parameters(self):
        """Test validation of confidence threshold and resolution."""
        with self.assertRaises(ValueError):
            HumanSegmenter(conf_threshold=-0.1)
        with self.assertRaises(ValueError):
            HumanSegmenter(imgsz=0)

    # ----------------------------------------------------
    # 2. PyTorch Image Segmentation Tests
    # ----------------------------------------------------

    def test_04_pytorch_image_segmentation_file_path(self):
        """Test segmentation using an image file path string and Path object."""
        self.assertTrue(
            self.positive_image_path.exists(),
            f"Test image missing: {self.positive_image_path}",
        )
        segmenter = HumanSegmenter(
            model_path=self.pt_model_path, conf_threshold=0.35
        )

        result_str = segmenter.segment(str(self.positive_image_path))
        self.assertIn("segmentations", result_str)
        self.assertGreater(len(result_str["segmentations"]), 0)

        result_path = segmenter.segment(self.positive_image_path)
        self.assertEqual(
            len(result_str["segmentations"]), len(result_path["segmentations"])
        )

    def test_05_pytorch_image_segmentation_numpy_array(self):
        """Test segmentation using an in-memory NumPy BGR image array."""
        segmenter = HumanSegmenter(
            model_path=self.pt_model_path, conf_threshold=0.35
        )
        img_bgr = cv2.imread(str(self.positive_image_path))
        self.assertIsNotNone(img_bgr)

        result = segmenter.segment(img_bgr)
        self.assertIn("segmentations", result)
        self.assertGreater(len(result["segmentations"]), 0)

    def test_06_pytorch_empty_segmentation_blank_image(self):
        """Test that an image with no humans returns an empty segmentations list."""
        segmenter = HumanSegmenter(
            model_path=self.pt_model_path, conf_threshold=0.35
        )
        result = segmenter.segment(self.blank_image)
        self.assertIn("segmentations", result)
        self.assertEqual(result["segmentations"], [])

    def test_07_pytorch_segmentation_format(self):
        """Verify strict adherence to segmentation schema including polygons."""
        segmenter = HumanSegmenter(
            model_path=self.pt_model_path, conf_threshold=0.35
        )
        img = cv2.imread(str(self.positive_image_path))
        h, w = img.shape[:2]

        result = segmenter.segment(img)
        segmentations = result["segmentations"]
        self.assertGreater(len(segmentations), 0)

        for seg in segmentations:
            self.assertIn("bbox", seg)
            self.assertIn("confidence", seg)
            self.assertIn("class_id", seg)
            self.assertIn("class_name", seg)
            self.assertIn("polygon", seg)
            self.assertIn("mask_area", seg)

            # Check bbox
            x1, y1, x2, y2 = seg["bbox"]
            self.assertGreaterEqual(x1, 0.0)
            self.assertGreaterEqual(y1, 0.0)
            self.assertLessEqual(x2, float(w))
            self.assertLessEqual(y2, float(h))
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)

            # Check polygon
            polygon = seg["polygon"]
            self.assertIsInstance(polygon, list)
            self.assertGreater(len(polygon), 2)  # Valid polygon has at least 3 points
            for pt in polygon:
                self.assertEqual(len(pt), 2)
                px, py = pt
                self.assertGreaterEqual(px, 0.0)
                self.assertLessEqual(px, float(w))
                self.assertGreaterEqual(py, 0.0)
                self.assertLessEqual(py, float(h))

            # Check mask area
            self.assertGreater(seg["mask_area"], 0)

    def test_08_pytorch_composite_mask(self):
        """Test generating full-resolution binary mask from PyTorch segmenter."""
        segmenter = HumanSegmenter(
            model_path=self.pt_model_path, conf_threshold=0.35
        )
        img = cv2.imread(str(self.positive_image_path))
        mask = segmenter.get_composite_mask(img)

        self.assertEqual(mask.shape, img.shape[:2])
        self.assertEqual(mask.dtype, np.uint8)
        self.assertGreater((mask == 255).sum(), 0)

    # ----------------------------------------------------
    # 3. ONNX Segmentation Model Loading Tests
    # ----------------------------------------------------

    def test_09_onnx_model_loading_success(self):
        """Test initializing HumanSegmenterONNX with exported model."""
        self.assertTrue(
            self.onnx_model_path.exists(),
            f"Exported ONNX file missing: {self.onnx_model_path}",
        )
        segmenter = HumanSegmenterONNX(model_path=self.onnx_model_path)
        self.assertIsNotNone(segmenter._session)
        self.assertEqual(segmenter.PERSON_CLASS_ID, 0)
        self.assertEqual(segmenter.PERSON_CLASS_NAME, "person")

    def test_10_onnx_model_loading_missing_file(self):
        """Test that missing ONNX file raises FileNotFoundError."""
        bogus_path = self.project_root / "models" / "missing_seg.onnx"
        with self.assertRaises(FileNotFoundError):
            HumanSegmenterONNX(model_path=bogus_path)

    # ----------------------------------------------------
    # 4. ONNX Segmentation Inference Tests
    # ----------------------------------------------------

    def test_11_onnx_image_segmentation_file_and_array(self):
        """Test ONNX Runtime segmentation using path and NumPy array."""
        segmenter = HumanSegmenterONNX(
            model_path=self.onnx_model_path, conf_threshold=0.35
        )

        res_file = segmenter.segment(self.positive_image_path)
        self.assertIn("segmentations", res_file)
        self.assertGreater(len(res_file["segmentations"]), 0)

        img = cv2.imread(str(self.positive_image_path))
        res_arr = segmenter.segment(img)
        self.assertIn("segmentations", res_arr)
        self.assertEqual(
            len(res_file["segmentations"]), len(res_arr["segmentations"])
        )

    def test_12_onnx_empty_segmentation_blank_image(self):
        """Test that ONNX model returns empty list on blank image."""
        segmenter = HumanSegmenterONNX(
            model_path=self.onnx_model_path, conf_threshold=0.35
        )
        result = segmenter.segment(self.blank_image)
        self.assertIn("segmentations", result)
        self.assertEqual(result["segmentations"], [])

    def test_13_onnx_segmentation_format(self):
        """Verify ONNX segmentation schema adherence, polygons, and mask areas."""
        segmenter = HumanSegmenterONNX(
            model_path=self.onnx_model_path, conf_threshold=0.35
        )
        img = cv2.imread(str(self.positive_image_path))
        h, w = img.shape[:2]

        result = segmenter.segment(img)
        for seg in result["segmentations"]:
            x1, y1, x2, y2 = seg["bbox"]
            self.assertGreaterEqual(x1, 0.0)
            self.assertLessEqual(x2, float(w))
            self.assertGreaterEqual(y1, 0.0)
            self.assertLessEqual(y2, float(h))
            self.assertEqual(seg["class_id"], 0)
            self.assertEqual(seg["class_name"], "person")
            self.assertGreater(len(seg["polygon"]), 2)
            self.assertGreater(seg["mask_area"], 0)

    def test_14_onnx_composite_mask(self):
        """Test generating full-resolution binary mask from ONNX segmenter."""
        segmenter = HumanSegmenterONNX(
            model_path=self.onnx_model_path, conf_threshold=0.35
        )
        img = cv2.imread(str(self.positive_image_path))
        mask = segmenter.get_composite_mask(img)

        self.assertEqual(mask.shape, img.shape[:2])
        self.assertEqual(mask.dtype, np.uint8)
        self.assertGreater((mask == 255).sum(), 0)

    def test_15_pytorch_onnx_cross_consistency(self):
        """Verify high numerical agreement between PyTorch and ONNX Runtime segmentations."""
        pt_segmenter = HumanSegmenter(
            model_path=self.pt_model_path, conf_threshold=0.35
        )
        onnx_segmenter = HumanSegmenterONNX(
            model_path=self.onnx_model_path, conf_threshold=0.35
        )

        pt_res = pt_segmenter.segment(self.positive_image_path)["segmentations"]
        onnx_res = onnx_segmenter.segment(self.positive_image_path)[
            "segmentations"
        ]

        self.assertGreater(len(pt_res), 0)
        self.assertEqual(len(pt_res), len(onnx_res))

        # Compare bounding box IoU
        b1 = pt_res[0]["bbox"]
        b2 = onnx_res[0]["bbox"]

        inter_x1 = max(b1[0], b2[0])
        inter_y1 = max(b1[1], b2[1])
        inter_x2 = min(b1[2], b2[2])
        inter_y2 = min(b1[3], b2[3])

        inter_area = max(0.0, inter_x2 - inter_x1) * max(
            0.0, inter_y2 - inter_y1
        )
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union_area = area1 + area2 - inter_area

        iou = inter_area / union_area if union_area > 0 else 0.0
        self.assertGreater(
            iou,
            0.90,
            f"Segmentation IoU between PyTorch and ONNX is too low: {iou:.3f}",
        )

        # Check mask areas are within 25% tolerance
        area_diff = (
            abs(pt_res[0]["mask_area"] - onnx_res[0]["mask_area"])
            / pt_res[0]["mask_area"]
        )
        self.assertLess(
            area_diff,
            0.25,
            f"Mask area difference too high: {area_diff:.3f}",
        )


if __name__ == "__main__":
    unittest.main()
