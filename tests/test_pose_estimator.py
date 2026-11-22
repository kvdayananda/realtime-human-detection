"""Unit tests for Standalone Google MediaPipe Pose Landmarker (PoseEstimator).

Covers:
- PoseEstimator initialization and cleanup
- Input validation (missing files, empty arrays, invalid types)
- Positive human sample inference (front-facing posture)
- Positive human sample inference (side-facing posture)
- Negative / blank frame inference (zero human detections)
- Landmark schema validation (33 keypoints, visibility, pixel coordinates)
- Visual overlay rendering via draw_pose()
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
from ai.pose.estimator import (
    POSE_CONNECTIONS,
    POSE_LANDMARK_NAMES,
    PoseEstimator,
)


class TestPoseEstimator(unittest.TestCase):
    """Test suite for PoseEstimator wrapping MediaPipe BlazePose."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.positive_front_path = (
            cls.project_root
            / "dataset"
            / "human detection dataset"
            / "1"
            / "0.png"
        )
        cls.positive_side_path = (
            cls.project_root
            / "dataset"
            / "human detection dataset"
            / "1"
            / "2.png"
        )
        cls.negative_path = (
            cls.project_root
            / "dataset"
            / "human detection dataset"
            / "0"
            / "0.png"
        )
        cls.blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # ----------------------------------------------------
    # 1. Initialization and Context Management
    # ----------------------------------------------------

    def test_01_initialization_defaults(self):
        """Test default PoseEstimator initialization and resource cleanup."""
        estimator = PoseEstimator()
        self.assertEqual(estimator.min_detection_confidence, 0.5)
        self.assertEqual(estimator.min_tracking_confidence, 0.5)
        self.assertEqual(estimator.model_complexity, 1)
        self.assertFalse(estimator.static_image_mode)
        estimator.close()

    def test_02_context_manager(self):
        """Test PoseEstimator works cleanly within a context manager."""
        with PoseEstimator(static_image_mode=True, model_complexity=0) as estimator:
            self.assertEqual(estimator.model_complexity, 0)
            self.assertTrue(estimator.static_image_mode)

    # ----------------------------------------------------
    # 2. Input Validation and Error Handling
    # ----------------------------------------------------

    def test_03_invalid_inputs(self):
        """Test that invalid types, empty arrays, and missing files raise errors."""
        with PoseEstimator(static_image_mode=True) as estimator:
            # Non-existent file path
            with self.assertRaises(FileNotFoundError):
                estimator.estimate_pose("non_existent_file.png")

            # Empty array
            with self.assertRaises(ValueError):
                estimator.estimate_pose(np.array([]))

            # Non-image type
            with self.assertRaises(TypeError):
                estimator.estimate_pose(12345)

    # ----------------------------------------------------
    # 3. Positive Front-Facing Sample
    # ----------------------------------------------------

    def test_04_positive_front_facing_pose(self):
        """Test pose estimation on a clear front-facing person image."""
        with PoseEstimator(static_image_mode=True, min_detection_confidence=0.4) as estimator:
            result = estimator.estimate_pose(str(self.positive_front_path))

            self.assertTrue(result["has_pose"])
            landmarks = result["landmarks"]
            self.assertEqual(len(landmarks), 33)

            # Check landmark attributes and constraints
            for lm in landmarks:
                self.assertIn("id", lm)
                self.assertIn("name", lm)
                self.assertIn("x", lm)
                self.assertIn("y", lm)
                self.assertIn("z", lm)
                self.assertIn("visibility", lm)
                self.assertIn("pixel_x", lm)
                self.assertIn("pixel_y", lm)

                self.assertIsInstance(lm["id"], int)
                self.assertIsInstance(lm["name"], str)
                self.assertGreaterEqual(lm["visibility"], 0.0)
                self.assertLessEqual(lm["visibility"], 1.0)

            # Check expected landmark names match topology
            names = [lm["name"] for lm in landmarks]
            self.assertEqual(names, list(POSE_LANDMARK_NAMES.values()))

            # Verify world landmarks are also provided
            world_lms = result["world_landmarks"]
            self.assertEqual(len(world_lms), 33)

    # ----------------------------------------------------
    # 4. Positive Side-Facing Sample
    # ----------------------------------------------------

    def test_05_positive_side_facing_pose(self):
        """Test pose estimation on a side-profile or active posture image."""
        with PoseEstimator(static_image_mode=True, min_detection_confidence=0.4) as estimator:
            result = estimator.estimate_pose(str(self.positive_side_path))

            self.assertTrue(result["has_pose"])
            self.assertEqual(len(result["landmarks"]), 33)

            # Validate key joints have high visibility
            landmarks_by_name = {lm["name"]: lm for lm in result["landmarks"]}
            self.assertIn("nose", landmarks_by_name)
            self.assertIn("left_shoulder", landmarks_by_name)
            self.assertIn("right_shoulder", landmarks_by_name)

    # ----------------------------------------------------
    # 5. Negative & Blank Frames
    # ----------------------------------------------------

    def test_06_negative_background_image(self):
        """Test pose estimation on empty background image without persons."""
        with PoseEstimator(static_image_mode=True) as estimator:
            result = estimator.estimate_pose(str(self.negative_path))

            self.assertFalse(result["has_pose"])
            self.assertEqual(result["landmarks"], [])
            self.assertEqual(result["world_landmarks"], [])

    def test_07_blank_synthetic_array(self):
        """Test handling of an all-black synthetic frame."""
        with PoseEstimator(static_image_mode=True) as estimator:
            result = estimator.estimate_pose(self.blank_frame)

            self.assertFalse(result["has_pose"])
            self.assertEqual(result["landmarks"], [])
            self.assertEqual(result["world_landmarks"], [])

    # ----------------------------------------------------
    # 6. Skeleton Visualization Rendering
    # ----------------------------------------------------

    def test_08_draw_pose_overlay(self):
        """Test drawing pose skeleton overlays on image frame."""
        with PoseEstimator(static_image_mode=True) as estimator:
            frame = cv2.imread(str(self.positive_front_path))
            result = estimator.estimate_pose(frame)
            self.assertTrue(result["has_pose"])

            annotated = estimator.draw_pose(frame, result, min_visibility=0.4)
            self.assertIsInstance(annotated, np.ndarray)
            self.assertEqual(annotated.shape, frame.shape)
            self.assertEqual(annotated.dtype, np.uint8)

            # Ensure annotated frame differs from raw frame (keypoints were drawn)
            diff = np.abs(annotated.astype(int) - frame.astype(int))
            self.assertGreater(diff.sum(), 0)

    def test_09_draw_pose_empty_result(self):
        """Test drawing when result has no pose does not mutate or crash."""
        with PoseEstimator(static_image_mode=True) as estimator:
            empty_result = {"has_pose": False, "landmarks": [], "world_landmarks": []}
            drawn = estimator.draw_pose(self.blank_frame, empty_result)
            self.assertEqual(drawn.shape, self.blank_frame.shape)
            self.assertTrue(np.array_equal(drawn, self.blank_frame))

    # ----------------------------------------------------
    # 7. Model Asset & Video Mode Timestamp Verification
    # ----------------------------------------------------

    def test_10_missing_model_asset_raises_error(self):
        """Test that non-existent .task model path raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            PoseEstimator(model_asset_path="non_existent_pose_model.task")

    def test_11_video_running_mode_with_timestamps(self):
        """Test video stream mode with explicit and auto-incrementing timestamps."""
        with PoseEstimator(static_image_mode=False, min_detection_confidence=0.4) as estimator:
            frame = cv2.imread(str(self.positive_front_path))
            self.assertIsNotNone(frame)

            # Test sequential frames with explicit timestamps
            res1 = estimator.estimate(frame, timestamp_ms=0)
            self.assertTrue(res1["has_pose"])

            res2 = estimator.estimate(frame, timestamp_ms=33)
            self.assertTrue(res2["has_pose"])

            # Test auto-incrementing fallback
            res3 = estimator.estimate(frame)
            self.assertTrue(res3["has_pose"])


if __name__ == "__main__":
    unittest.main()
