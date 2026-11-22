"""Unit tests for FitRange Mirror - Human Detection Module.

Covers:
- Model loading (PyTorch and ONNX)
- Image inference (file path and NumPy array)
- Empty detection result handling
- Bounding-box format and schema validation
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

from ai.detection.detect import HumanDetector
from ai.detection.inference_onnx import HumanDetectorONNX


class TestHumanDetection(unittest.TestCase):
    """Test suite for PyTorch and ONNX human detection modules."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.pt_model_path = (
            cls.project_root
            / "models"
            / "human_detection"
            / "human_detection.pt"
        )
        cls.onnx_model_path = (
            cls.project_root
            / "models"
            / "human_detection"
            / "human_detection.onnx"
        )

        # Select a suitable existing project image with a person
        cls.positive_image_path = (
            cls.project_root
            / "dataset"
            / "human detection dataset"
            / "1"
            / "0.png"
        )
        if not cls.positive_image_path.exists():
            # Fallback to test directory if needed
            alt = (
                cls.project_root
                / "dataset"
                / "person_detection_test"
                / "images"
                / "class1_0.png"
            )
            if alt.exists():
                cls.positive_image_path = alt

        # Create a synthetic blank black image for empty detection test
        cls.blank_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # ----------------------------------------------------
    # 1. PyTorch Model Loading & Validation Tests
    # ----------------------------------------------------

    def test_01_pytorch_model_loading_success(self):
        """Test loading PyTorch model with default and explicit paths."""
        detector = HumanDetector(model_path=self.pt_model_path)
        self.assertIsNotNone(detector._model)
        self.assertEqual(detector.PERSON_CLASS_ID, 0)
        self.assertEqual(detector.PERSON_CLASS_NAME, "person")

    def test_02_pytorch_model_loading_missing_file(self):
        """Test that non-existent model path raises FileNotFoundError."""
        bogus_path = self.project_root / "models" / "non_existent_model.pt"
        with self.assertRaises(FileNotFoundError):
            HumanDetector(model_path=bogus_path)

    def test_03_pytorch_invalid_parameters(self):
        """Test validation of confidence threshold and image resolution."""
        with self.assertRaises(ValueError):
            HumanDetector(conf_threshold=-0.1)
        with self.assertRaises(ValueError):
            HumanDetector(conf_threshold=1.5)
        with self.assertRaises(ValueError):
            HumanDetector(imgsz=0)

    # ----------------------------------------------------
    # 2. PyTorch Image Inference Tests
    # ----------------------------------------------------

    def test_04_pytorch_image_inference_file_path(self):
        """Test detection using an image file path string and Path object."""
        self.assertTrue(
            self.positive_image_path.exists(),
            f"Test image missing: {self.positive_image_path}",
        )
        detector = HumanDetector(model_path=self.pt_model_path, conf_threshold=0.35)

        # Test with str path
        result_str = detector.detect(str(self.positive_image_path))
        self.assertIn("detections", result_str)
        self.assertIsInstance(result_str["detections"], list)
        self.assertGreater(len(result_str["detections"]), 0)

        # Test with Path object
        result_path = detector.detect(self.positive_image_path)
        self.assertEqual(len(result_str["detections"]), len(result_path["detections"]))

    def test_05_pytorch_image_inference_numpy_array(self):
        """Test detection using an in-memory NumPy BGR image array."""
        detector = HumanDetector(model_path=self.pt_model_path, conf_threshold=0.35)
        img_bgr = cv2.imread(str(self.positive_image_path))
        self.assertIsNotNone(img_bgr)

        result = detector.detect(img_bgr)
        self.assertIn("detections", result)
        self.assertGreater(len(result["detections"]), 0)

    def test_06_pytorch_image_input_validation(self):
        """Test handling of invalid image paths and unsupported input types."""
        detector = HumanDetector(model_path=self.pt_model_path)

        with self.assertRaises(FileNotFoundError):
            detector.detect("non_existent_image_file.png")

        with self.assertRaises(ValueError):
            detector.detect(np.array([]))

        with self.assertRaises(TypeError):
            detector.detect(12345)

    # ----------------------------------------------------
    # 3. Empty Detection Handling
    # ----------------------------------------------------

    def test_07_pytorch_empty_detection_blank_image(self):
        """Test that an image with no humans returns an empty detections list."""
        detector = HumanDetector(model_path=self.pt_model_path, conf_threshold=0.35)
        result = detector.detect(self.blank_image)

        self.assertIn("detections", result)
        self.assertEqual(result["detections"], [])

    # ----------------------------------------------------
    # 4. Bounding Box & Schema Validation Tests
    # ----------------------------------------------------

    def test_08_pytorch_bounding_box_format(self):
        """Verify strict adherence to required result schema."""
        detector = HumanDetector(model_path=self.pt_model_path, conf_threshold=0.35)
        img = cv2.imread(str(self.positive_image_path))
        h, w = img.shape[:2]

        result = detector.detect(img)
        detections = result["detections"]
        self.assertGreater(len(detections), 0)

        for det in detections:
            # Check required keys
            self.assertIn("bbox", det)
            self.assertIn("confidence", det)
            self.assertIn("class_id", det)
            self.assertIn("class_name", det)

            # Check types
            self.assertIsInstance(det["bbox"], list)
            self.assertEqual(len(det["bbox"]), 4)
            self.assertIsInstance(det["confidence"], float)
            self.assertIsInstance(det["class_id"], int)
            self.assertIsInstance(det["class_name"], str)

            # Check value constraints
            x1, y1, x2, y2 = det["bbox"]
            self.assertGreaterEqual(x1, 0.0)
            self.assertGreaterEqual(y1, 0.0)
            self.assertLessEqual(x2, float(w))
            self.assertLessEqual(y2, float(h))
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)

            self.assertGreaterEqual(det["confidence"], 0.35)
            self.assertLessEqual(det["confidence"], 1.0)
            self.assertEqual(det["class_id"], 0)
            self.assertEqual(det["class_name"], "person")

    # ----------------------------------------------------
    # 5. ONNX Model Loading & Validation Tests
    # ----------------------------------------------------

    def test_09_onnx_model_loading_success(self):
        """Test initializing HumanDetectorONNX with exported model."""
        self.assertTrue(
            self.onnx_model_path.exists(),
            f"Exported ONNX file missing: {self.onnx_model_path}",
        )
        detector = HumanDetectorONNX(model_path=self.onnx_model_path)
        self.assertIsNotNone(detector._session)
        self.assertEqual(detector.PERSON_CLASS_ID, 0)
        self.assertEqual(detector.PERSON_CLASS_NAME, "person")

    def test_10_onnx_model_loading_missing_file(self):
        """Test that missing ONNX file raises FileNotFoundError."""
        bogus_path = self.project_root / "models" / "missing.onnx"
        with self.assertRaises(FileNotFoundError):
            HumanDetectorONNX(model_path=bogus_path)

    # ----------------------------------------------------
    # 6. ONNX Inference & Consistency Tests
    # ----------------------------------------------------

    def test_11_onnx_image_inference_file_and_array(self):
        """Test ONNX Runtime inference using path and NumPy array."""
        detector = HumanDetectorONNX(
            model_path=self.onnx_model_path, conf_threshold=0.35
        )

        # Via file path
        res_file = detector.detect(self.positive_image_path)
        self.assertIn("detections", res_file)
        self.assertGreater(len(res_file["detections"]), 0)

        # Via NumPy array
        img = cv2.imread(str(self.positive_image_path))
        res_arr = detector.detect(img)
        self.assertIn("detections", res_arr)
        self.assertEqual(len(res_file["detections"]), len(res_arr["detections"]))

    def test_12_onnx_empty_detection_blank_image(self):
        """Test that ONNX model returns empty list on blank image."""
        detector = HumanDetectorONNX(
            model_path=self.onnx_model_path, conf_threshold=0.35
        )
        result = detector.detect(self.blank_image)
        self.assertIn("detections", result)
        self.assertEqual(result["detections"], [])

    def test_13_onnx_bounding_box_format(self):
        """Verify ONNX detection schema adherence and coordinate bounds."""
        detector = HumanDetectorONNX(
            model_path=self.onnx_model_path, conf_threshold=0.35
        )
        img = cv2.imread(str(self.positive_image_path))
        h, w = img.shape[:2]

        result = detector.detect(img)
        for det in result["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            self.assertGreaterEqual(x1, 0.0)
            self.assertGreaterEqual(y1, 0.0)
            self.assertLessEqual(x2, float(w))
            self.assertLessEqual(y2, float(h))
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)
            self.assertEqual(det["class_id"], 0)
            self.assertEqual(det["class_name"], "person")

    def test_14_pytorch_onnx_cross_consistency(self):
        """Verify high numerical agreement between PyTorch and ONNX Runtime detections."""
        pt_detector = HumanDetector(
            model_path=self.pt_model_path, conf_threshold=0.35
        )
        onnx_detector = HumanDetectorONNX(
            model_path=self.onnx_model_path, conf_threshold=0.35
        )

        pt_res = pt_detector.detect(self.positive_image_path)["detections"]
        onnx_res = onnx_detector.detect(self.positive_image_path)["detections"]

        self.assertGreater(len(pt_res), 0)
        self.assertEqual(len(pt_res), len(onnx_res))

        # Compare primary detection bounding box IoU
        b1 = pt_res[0]["bbox"]
        b2 = onnx_res[0]["bbox"]

        # Calculate IoU
        inter_x1 = max(b1[0], b2[0])
        inter_y1 = max(b1[1], b2[1])
        inter_x2 = min(b1[2], b2[2])
        inter_y2 = min(b1[3], b2[3])

        inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union_area = area1 + area2 - inter_area

        iou = inter_area / union_area if union_area > 0 else 0.0
        self.assertGreater(
            iou,
            0.90,
            f"IoU between PyTorch and ONNX is too low: {iou:.3f}",
        )

        # Check confidence difference
        conf_diff = abs(pt_res[0]["confidence"] - onnx_res[0]["confidence"])
        self.assertLess(
         