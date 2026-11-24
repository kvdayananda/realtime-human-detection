"""Integration tests for FitRange Mirror - Integrated AI Pipeline.

Covers:
- Pipeline initialization with ONNX models
- Standalone execution of detection and segmentation
- Frame processing on positive sample images
- Empty detection handling on negative / blank frames
- Binary composite mask generation and shape validation
- Visual rendering and overlay validation
- Robust error handling for corrupt / invalid inputs
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
from ai.pipeline.mirror_pipeline import FitRangeMirrorPipeline


class TestFitRangeMirrorPipeline(unittest.TestCase):
    """Integration test suite for FitRangeMirrorPipeline."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.det_model_path = (
            cls.project_root
            / "models"
            / "human_detection"
            / "human_detection.onnx"
        )
        cls.seg_model_path = (
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
        cls.negative_image_path = (
            cls.project_root
            / "dataset"
            / "human detection dataset"
            / "0"
            / "0.png"
        )
        cls.blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # ----------------------------------------------------
    # 1. Pipeline Initialization & Parameter Validation
    # ----------------------------------------------------

    def test_01_pipeline_initialization(self):
        """Test default pipeline initialization with detection, segmentation, and pose engines."""
        pipeline = FitRangeMirrorPipeline(
            detection_model_path=self.det_model_path,
            segmentation_model_path=self.seg_model_path,
        )
        self.assertIsNotNone(pipeline.detector)
        self.assertIsNotNone(pipeline.segmenter)
        self.assertIsNotNone(pipeline.pose_estimator)
        self.assertTrue(pipeline.enable_detection)
        self.assertTrue(pipeline.enable_segmentation)
        self.assertTrue(pipeline.enable_pose)

    def test_02_pipeline_partial_modes(self):
        """Test initializing pipeline in detection-only, segmentation-only, and pose-only modes."""
        # Detection only
        det_only = FitRangeMirrorPipeline(
            detection_model_path=self.det_model_path,
            enable_detection=True,
            enable_segmentation=False,
            enable_pose=False,
        )
        self.assertIsNotNone(det_only.detector)
        self.assertIsNone(det_only.segmenter)
        self.assertIsNone(det_only.pose_estimator)

        # Segmentation only
        seg_only = FitRangeMirrorPipeline(
            segmentation_model_path=self.seg_model_path,
            enable_detection=False,
            enable_segmentation=True,
            enable_pose=False,
        )
        self.assertIsNone(seg_only.detector)
        self.assertIsNotNone(seg_only.segmenter)
        self.assertIsNone(seg_only.pose_estimator)

        # Pose only
        pose_only = FitRangeMirrorPipeline(
            enable_detection=False,
            enable_segmentation=False,
            enable_pose=True,
        )
        self.assertIsNone(pose_only.detector)
        self.assertIsNone(pose_only.segmenter)
        self.assertIsNotNone(pose_only.pose_estimator)

    def test_03_pipeline_invalid_configuration(self):
        """Test that disabling all three engines raises ValueError."""
        with self.assertRaises(ValueError):
            FitRangeMirrorPipeline(
                enable_detection=False,
                enable_segmentation=False,
                enable_pose=False,
            )

    def test_04_pipeline_missing_model_files(self):
        """Test that non-existent model files raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            FitRangeMirrorPipeline(
                detection_model_path="non_existent_det.onnx"
            )
        with self.assertRaises(FileNotFoundError):
            FitRangeMirrorPipeline(
                segmentation_model_path="non_existent_seg.onnx"
            )
        with self.assertRaises(FileNotFoundError):
            FitRangeMirrorPipeline(
                pose_model_path="non_existent_pose.task"
            )

    # ----------------------------------------------------
    # 2. Positive Sample Image Processing
    # ----------------------------------------------------

    def test_05_process_positive_sample_frame(self):
        """Test frame processing on a confirmed human image."""
        pipeline = FitRangeMirrorPipeline(
            detection_model_path=self.det_model_path,
            segmentation_model_path=self.seg_model_path,
            det_conf=0.35,
            seg_conf=0.35,
        )
        frame = cv2.imread(str(self.positive_image_path))
        self.assertIsNotNone(frame)
        h, w = frame.shape[:2]

        result = pipeline.process_frame(frame)

        # Check top-level flags
        self.assertTrue(result["has_person"])
        self.assertGreaterEqual(result["person_count"], 1)
        self.assertEqual(result["frame_shape"], (h, w))

        # Check detections schema
        self.assertGreater(len(result["detections"]), 0)
        for det in result["detections"]:
            self.assertEqual(len(det["bbox"]), 4)
            self.assertGreaterEqual(det["confidence"], 0.35)
            self.assertEqual(det["class_id"], 0)
            self.assertEqual(det["class_name"], "person")

        # Check segmentations schema
        self.assertGreater(len(result["segmentations"]), 0)
        for seg in result["segmentations"]:
            self.assertEqual(len(seg["bbox"]), 4)
            self.assertIn("polygon", seg)
            self.assertGreater(len(seg["polygon"]), 2)
            self.assertGreater(seg["mask_area"], 0)

        # Check composite mask
        mask = result["composite_mask"]
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(mask.shape, (h, w))
        self.assertEqual(mask.dtype, np.uint8)
        # Should contain person pixels (value 255)
        self.assertGreater((mask == 255).sum(), 0)
        # Should only contain binary values {0, 255}
        unique_vals = set(np.unique(mask).tolist())
        self.assertTrue(unique_vals.issubset({0, 255}))

        # Check pose estimation output
        self.assertIn("pose", result)
        pose_res = result["pose"]
        self.assertTrue(pose_res["has_pose"])
        self.assertEqual(len(pose_res["landmarks"]), 33)
        self.assertEqual(len(pose_res["world_landmarks"]), 33)

    # ----------------------------------------------------
    # 3. Empty / Negative Frame Processing
    # ----------------------------------------------------

    def test_06_process_negative_sample_frame(self):
        """Test frame processing on background image with no human."""
        pipeline = FitRangeMirrorPipeline(
            detection_model_path=self.det_model_path,
            segmentation_model_path=self.seg_model_path,
        )
        frame = cv2.imread(str(self.negative_image_path))
        self.assertIsNotNone(frame)
        h, w = frame.shape[:2]

        result = pipeline.process_frame(frame)

        self.assertFalse(result["has_person"])
        self.assertEqual(result["person_count"], 0)
        self.assertEqual(result["detections"], [])
        self.assertEqual(result["segmentations"], [])
        self.assertFalse(result["pose"]["has_pose"])
        self.assertEqual(result["pose"]["landmarks"], [])

        mask = result["composite_mask"]
        self.assertEqual(mask.shape, (h, w))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual((mask == 255).sum(), 0)

    def test_07_process_blank_array(self):
        """Test handling of an all-black synthetic frame."""
        pipeline = FitRangeMirrorPipeline(
            detection_model_path=self.det_model_path,
            segmentation_model_path=self.seg_model_path,
        )
        result = pipeline.process_frame(self.blank_frame)

        self.assertFalse(result["has_person"])
        self.assertEqual(result["person_count"], 0)
        self.assertEqual(result["detections"], [])
        self.assertEqual(result["segmentations"], [])
        self.assertFalse(result["pose"]["has_pose"])
        self.assertEqual(result["pose"]["landmarks"], [])
        self.assertEqual((result["composite_mask"] == 0).all(), True)

    # ----------------------------------------------------
    # 4. Input Validation & Error Handling
    # ----------------------------------------------------

    def test_08_invalid_frame_inputs(self):
        """Test that invalid frame inputs raise appropriate exceptions."""
        pipeline = FitRangeMirrorPipeline(
            detection_model_path=self.det_model_path,
            segmentation_model_path=self.seg_model_path,
        )

        with self.assertRaises(ValueError):
            pipeline.process_frame(np.array([]))

        with self.assertRaises(TypeError):
            pipeline.process_frame(42)

        with self.assertRaises(FileNotFoundError):
            pipeline.process_frame("missing_image_path.jpg")

    # ----------------------------------------------------
    # 5. Visual Overlay Rendering
    # ----------------------------------------------------

    def test_09_render_visualization_positive(self):
        """Test rendering visual overlays on positive detection frame."""
        pipeline = FitRangeMirrorPipeline(
            detection_model_path=self.det_model_path,
            segmentation_model_path=self.seg_model_path,
        )
        frame = cv2.imread(str(self.positive_image_path))
        result = pipeline.process_frame(frame)

        vis = pipeline.render_visualization(
            frame,
            result,
            show_boxes=True,
            show_masks=True,
            show_polygons=True,
            show_pose=True,
            show_hud=True,
            fps=30.0,
        )

        self.assertIsInstance(vis, np.ndarray)
        self.assertEqual(vis.shape, frame.shape)
        self.assertEqual(vis.dtype, np.uint8)

    def test_10_render_visualization_negative(self):
        """Test rendering visual overlay on negative frame."""
        pipeline = FitRangeMirrorPipeline(
            detection_model_path=self.det_model_path,
            segmentation_model_path=self.seg_model_path,
        )
        result = pipeline.process_frame(self.blank_frame)
        vis = pipeline.render_visualization(self.blank_frame, result, fps=25.0)

        self.assertEqual(vis.shape, self.blank_frame.shape)
        self.assertEqual(vis.dtype, np.uint8)

    def test_11_pipeline_pose_disabled(self):
        """Test pipeline behavior when pose estimation is explicitly disabled."""
        pipeline = FitRangeMirrorPipeline(
            detection_model_path=self.det_model_path,
            segmentation_model_path=self.seg_model_path,
            enable_pose=False,
        )
        self.assertIsNone(pipeline.pose_estimator)
        frame = cv2.imread(str(self.positive_image_path))
        result = pipeline.process_frame(frame)

        self.assertTrue(result["has_person"])
        self.assertIn("pose", result)
        self.assertFalse(result["pose"]["has_pose"])
        self.assertEqual(result["pose"]["landmarks"], [])
        self.assertEqual(result["pose"]["world_landmarks"], [])

    def test_12_single_person_mode_filter(self):
        """Test that single_person_mode isolates only the primary user in multi-person frame."""
        frame = cv2.imread(str(self.positive_image_path))
        multi_frame = np.hstack([frame, frame])

        # Test single_person_mode=True (default)
        single_pipeline = FitRangeMirrorPipeline(
            detection_model_path=self.det_model_path,
            segmentation_model_path=self.seg_model_path,
            enable_pose=False,
            single_person_mode=True,
        )
        single_res = single_pipeline.process_frame(multi_frame)
        self.assertTrue(single_res["has_person"])
        self.assertEqual(single_res["person_count"], 1)
        self.assertEqual(len(single_res["detections"]), 1)
        self.assertEqual(len(single_res["segmentations"]), 1)

        # Test single_person_mode=False (multi-person tracking)
        multi_pipeline = FitRangeMirrorPipeline(
            detection_model_path=self.det_model_path,
            segmentation_model_path=self.seg_model_path,
            enable_pose=False,
            single_person_mode=False,
        )
        multi_res = multi_pipeline.process_frame(multi_frame)
        self.assertTrue(multi_res["has_person"])
        self.assertGreaterEqual(multi_res["person_count"], 2)
        self.assertGreaterEqual(len(multi_res["detections"]), 2)
        self.assertGreaterEqual(len(multi_res["segmentations"]), 2)

    def test_13_render_binary_mask(self):
        """Test rendering pure binary mask (pitch black background with solid white silhouette)."""
        pipeline = FitRangeMirrorPipeline(
            detection_model_path=self.det_model_path,
            segmentation_model_path=self.seg_model_path,
            enable_pose=False,
        )
        frame = cv2.imread(str(self.positive_image_path))
        result = pipeline.process_frame(frame)

        mask_vis = pipeline.render_binary_mask(result, show_card=False)
        self.assertEqual(mask_vis.shape, frame.shape)
        self.assertEqual(mask_vis.dtype, np.uint8)

        # Pixels must be exclusively 0 (black background) or 255 (white silhouette)
        unique_vals = set(np.unique(mask_vis).tolist())
        self.assertTrue(unique_vals.issubset({0, 255}))
        self.assertGreater((mask_vis == 255).sum(), 0)

    def test_14_render_binary_mask_card(self):
        """Test rendering binary mask formatted with UI card banner matching project spec."""
        pipeline = FitRangeMirrorPipeline(
            detection_model_path=self.det_model_path,
            segmentation_model_path=self.seg_model_path,
            enable_pose=False,
        )
        frame = cv2.imread(str(self.positive_image_path))
        result = pipeline.process_frame(frame)

        card = pipeline.render_binary_mask(result, show_card=True)
        self.assertGreater(card.shape[0], frame.shape[0])
        self.assertEqual(card.shape[1], frame.shape[1])
        self.assertEqual(card.shape[2], 3)
        self.assertEqual(card.dtype, np.uint8)

    def test_15_render_visualization_view_modes(self):
        """Test render_visualization across all view modes ('overlay', 'mask', 'mask_card', 'side_by_side')."""
        pipeline = FitRangeMirrorPipeline(
            detection_model_path=self.det_model_path,
            segmentation_model_path=self.seg_model_path,
            enable_pose=False,
        )
        frame = cv2.imread(str(self.positive_image_path))
        result = pipeline.process_frame(frame)

        # 1. Overlay
        vis_overlay = pipeline.render_visualization(frame, result, view_mode="overlay")
        self.assertEqual(vis_overlay.shape, frame.shape)

        # 2. Pure Mask
        vis_mask = pipeline.render_visualization(frame, result, view_mode="mask")
        self.assertEqual(vis_mask.shape, frame.shape)

        # 3. Mask Card
        vis_card = pipeline.render_visualization(frame, result, view_mode="mask_card")
        self.assertGreater(vis_card.shape[0], frame.shape[0])

        # 4. Side-by-side
        vis_sbs = pipeline.render_visualization(frame, result, view_mode="side_by_side")
        self.assertEqual(vis_sbs.shape[0], frame.shape[0])
        self.assertEqual(vis_sbs.shape[1], frame.shape[1] * 2)


if __name__ == "__main__":
    unittest.main()
