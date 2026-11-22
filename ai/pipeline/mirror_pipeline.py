"""FitRange Mirror - Integrated Human Detection, Segmentation & Pose Estimation Pipeline.

This module provides the `FitRangeMirrorPipeline` class, which combines standalone
ONNX Human Detection, ONNX Human Instance Segmentation, and Google MediaPipe Pose Landmarker
into a unified pipeline optimized for real-time edge processing (webcam, video stream, or static frames).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from ai.detection.inference_onnx import HumanDetectorONNX
from ai.pose.estimator import PoseEstimator
from ai.segmentation.inference_onnx import HumanSegmenterONNX

logger = logging.getLogger("FitRangeMirror.Pipeline")


class FitRangeMirrorPipeline:
    """Integrated pipeline managing independent Detection, Segmentation, and Pose engines."""

    def __init__(
        self,
        detection_model_path: Optional[Union[str, Path]] = None,
        segmentation_model_path: Optional[Union[str, Path]] = None,
        pose_model_path: Optional[Union[str, Path]] = None,
        det_conf: float = 0.35,
        seg_conf: float = 0.35,
        pose_conf: float = 0.5,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        enable_detection: bool = True,
        enable_segmentation: bool = True,
        enable_pose: bool = True,
        pose_complexity: int = 1,
        pose_static_mode: bool = False,
        providers: Optional[List[str]] = None,
    ) -> None:
        """Initialize the integrated pipeline.

        Args:
            detection_model_path: Path to detection .onnx model. Defaults to
                models/human_detection/human_detection.onnx
            segmentation_model_path: Path to segmentation .onnx model. Defaults to
                models/human_segmentation/human_segmentation.onnx
            pose_model_path: Path to pose .task model asset. Defaults to
                models/pose/pose_landmarker_full.task
            det_conf: Confidence threshold for human detection.
            seg_conf: Confidence threshold for human segmentation.
            pose_conf: Confidence threshold for MediaPipe Pose Landmarker.
            iou_threshold: NMS IoU threshold.
            imgsz: Inference square dimension (default: 640).
            enable_detection: Whether to execute the detection model.
            enable_segmentation: Whether to execute the segmentation model.
            enable_pose: Whether to execute the pose estimation model.
            pose_complexity: MediaPipe Pose complexity (0: Lite, 1: Full, 2: Heavy).
            pose_static_mode: Whether to treat frames independently (True for static images).
            providers: ONNX Runtime execution providers (default: ['CPUExecutionProvider']).

        Raises:
            FileNotFoundError: If enabled model files do not exist.
            ValueError: If all engines are disabled.
        """
        if not enable_detection and not enable_segmentation and not enable_pose:
            raise ValueError(
                "At least one of enable_detection, enable_segmentation, or enable_pose must be True."
            )

        self.enable_detection = enable_detection
        self.enable_segmentation = enable_segmentation
        self.enable_pose = enable_pose
        self.det_conf = det_conf
        self.seg_conf = seg_conf
        self.pose_conf = pose_conf
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz

        project_root = Path(__file__).resolve().parent.parent.parent

        # 1. Initialize Human Detection engine if enabled
        self.detector: Optional[HumanDetectorONNX] = None
        if self.enable_detection:
            det_path = (
                Path(detection_model_path).resolve()
                if detection_model_path
                else project_root
                / "models"
                / "human_detection"
                / "human_detection.onnx"
            )
            if not det_path.is_file():
                logger.error("Detection ONNX model not found at: %s", det_path)
                raise FileNotFoundError(
                    f"Detection ONNX model file not found at: {det_path}"
                )
            logger.info("Initializing HumanDetectorONNX from: %s", det_path)
            self.detector = HumanDetectorONNX(
                model_path=det_path,
                conf_threshold=det_conf,
                iou_threshold=iou_threshold,
                imgsz=imgsz,
                providers=providers,
            )

        # 2. Initialize Human Segmentation engine if enabled
        self.segmenter: Optional[HumanSegmenterONNX] = None
        if self.enable_segmentation:
            seg_path = (
                Path(segmentation_model_path).resolve()
                if segmentation_model_path
                else project_root
                / "models"
                / "human_segmentation"
                / "human_segmentation.onnx"
            )
            if not seg_path.is_file():
                logger.error("Segmentation ONNX model not found at: %s", seg_path)
                raise FileNotFoundError(
                    f"Segmentation ONNX model file not found at: {seg_path}"
                )
            logger.info("Initializing HumanSegmenterONNX from: %s", seg_path)
            self.segmenter = HumanSegmenterONNX(
                model_path=seg_path,
                conf_threshold=seg_conf,
                iou_threshold=iou_threshold,
                imgsz=imgsz,
                providers=providers,
            )

        # 3. Initialize Google MediaPipe Pose Landmarker if enabled
        self.pose_estimator: Optional[PoseEstimator] = None
        if self.enable_pose:
            logger.info(
                "Initializing MediaPipe Tasks PoseEstimator (complexity=%d, static=%s)",
                pose_complexity,
                pose_static_mode,
            )
            self.pose_estimator = PoseEstimator(
                model_asset_path=pose_model_path,
                static_image_mode=pose_static_mode,
                model_complexity=pose_complexity,
                min_detection_confidence=pose_conf,
                min_tracking_confidence=pose_conf,
            )

        logger.info(
            "FitRangeMirrorPipeline initialized successfully (det=%s, seg=%s, pose=%s, imgsz=%d)",
            self.enable_detection,
            self.enable_segmentation,
            self.enable_pose,
            self.imgsz,
        )

    def process_frame(
        self,
        frame: Union[np.ndarray, str, Path],
        det_conf: Optional[float] = None,
        seg_conf: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Process an input frame through detection, segmentation, and pose models.

        Args:
            frame: Input frame as a NumPy array (BGR uint8) or valid image path.
            det_conf: Optional per-frame confidence override for detection.
            seg_conf: Optional per-frame confidence override for segmentation.

        Returns:
            Dictionary matching the integrated schema:
            {
                "has_person": bool,
                "person_count": int,
                "detections": [
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float,
                        "class_id": 0,
                        "class_name": "person"
                    }
                ],
                "segmentations": [
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float,
                        "class_id": 0,
                        "class_name": "person",
                        "polygon": [[x1, y1], [x2, y2], ...],
                        "mask_area": int
                    }
                ],
                "composite_mask": np.ndarray,  # uint8 binary mask (H, W), values 0 or 255
                "pose": {
                    "has_pose": bool,
                    "landmarks": [
                        {"id": int, "name": str, "x": float, "y": float, "z": float, "visibility": float, "pixel_x": int, "pixel_y": int}
                    ],
                    "world_landmarks": [...]
                },
                "frame_shape": (height, width)
            }

        Raises:
            ValueError: If the frame is invalid, empty, or corrupt.
            TypeError: If frame is of unsupported type.
        """
        img_bgr = self._validate_and_load_frame(frame)
        h, w = img_bgr.shape[:2]

        detections: List[Dict[str, Any]] = []
        segmentations: List[Dict[str, Any]] = []

        # 1. Run detection if enabled
        if self.detector is not None:
            try:
                det_result = self.detector.detect(
                    img_bgr,
                    conf_threshold=det_conf or self.det_conf,
                    iou_threshold=self.iou_threshold,
                )
                detections = det_result.get("detections", [])
            except Exception as e:
                logger.error("Error during detection inference: %s", e, exc_info=True)
                raise

        # 2. Run segmentation if enabled
        composite_mask = np.zeros((h, w), dtype=np.uint8)
        if self.segmenter is not None:
            try:
                seg_result = self.segmenter.segment(
                    img_bgr,
                    conf_threshold=seg_conf or self.seg_conf,
                    iou_threshold=self.iou_threshold,
                )
                segmentations = seg_result.get("segmentations", [])

                # Generate full-frame composite binary mask
                if segmentations:
                    composite_mask = self.segmenter.get_composite_mask(
                        img_bgr, segmentations=segmentations
                    )
            except Exception as e:
                logger.error("Error during segmentation inference: %s", e, exc_info=True)
                raise

        # 3. Run pose estimation if enabled
        pose_data: Dict[str, Any] = {
            "has_pose": False,
            "landmarks": [],
            "world_landmarks": [],
        }
        if self.pose_estimator is not None:
            try:
                pose_data = self.pose_estimator.estimate(img_bgr)
            except Exception as e:
                logger.error("Error during pose estimation: %s", e, exc_info=True)
                raise

        # 4. Determine person count and presence
        if self.detector is not None:
            person_count = len(detections)
        elif self.segmenter is not None:
            person_count = len(segmentations)
        else:
            person_count = 1 if pose_data.get("has_pose", False) else 0

        has_person = person_count > 0 or pose_data.get("has_pose", False)

        return {
            "has_person": has_person,
            "person_count": person_count,
            "detections": detections,
            "segmentations": segmentations,
            "composite_mask": composite_mask,
            "pose": pose_data,
            "frame_shape": (h, w),
        }

    @staticmethod
    def _validate_and_load_frame(frame: Union[np.ndarray, str, Path]) -> np.ndarray:
        """Validate input frame format and convert to BGR uint8 array."""
        if isinstance(frame, (str, Path)):
            path = Path(frame).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Frame file not found: {path}")
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Failed to read/decode image file: {path}")
            return img

        if isinstance(frame, np.ndarray):
            if frame.size == 0:
                raise ValueError("Input frame is empty (size 0).")
            if frame.ndim == 2:
                return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if frame.ndim == 3:
                if frame.shape[2] == 4:
                    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                if frame.shape[2] == 3:
                    return frame
                raise ValueError(
                    f"Unexpected number of channels: {frame.shape[2]}. Expected 3 or 4."
                )
            raise ValueError(
                f"Expected 2D or 3D frame array, got ndim={frame.ndim}"
            )

        raise TypeError(
            f"Unsupported frame type: {type(frame)}. Expected np.ndarray, str, or Path."
        )

    def render_visualization(
        self,
        frame: np.ndarray,
        result: Dict[str, Any],
        show_boxes: bool = True,
        show_masks: bool = True,
        show_polygons: bool = True,
        show_pose: bool = True,
        show_hud: bool = True,
        mask_color: Tuple[int, int, int] = (0, 255, 128),  # Greenish cyan
        bbox_color: Tuple[int, int, int] = (0, 255, 0),    # Pure green
        alpha: float = 0.35,
        fps: Optional[float] = None,
    ) -> np.ndarray:
        """Render detection boxes, segmentation masks, skeleton landmarks, and HUD onto frame.

        Args:
            frame: Input frame array (BGR uint8).
            result: Result dictionary returned by `process_frame`.
            show_boxes: Whether to draw bounding box rectangles and labels.
            show_masks: Whether to alpha-blend the binary segmentation mask.
            show_polygons: Whether to trace mask polygon contour outlines.
            show_pose: Whether to draw skeleton connections and landmark points.
            show_hud: Whether to draw top information HUD bar.
            mask_color: Color tuple (B, G, R) for segmentation fill.
            bbox_color: Color tuple (B, G, R) for bounding box outline.
            alpha: Transparency factor for mask blending (0.0 to 1.0).
            fps: Current frame-per-second metric to display.

        Returns:
            np.ndarray with rendered visual overlays (same shape and dtype as frame).
        """
        vis = frame.copy()
        h, w = vis.shape[:2]

        # 1. Render segmentation mask overlay
        if show_masks and result.get("has_person", False):
            comp_mask = result.get("composite_mask")
            if comp_mask is not None and (comp_mask == 255).any():
                color_mask = np.zeros_like(vis)
                color_mask[comp_mask == 255] = mask_color
                cv2.addWeighted(color_mask, alpha, vis, 1.0, 0, vis)

        # 2. Render polygon outlines
        if show_polygons:
            for seg in result.get("segmentations", []):
                poly = seg.get("polygon", [])
                if len(poly) >= 3:
                    pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(
                        vis, [pts], isClosed=True, color=(255, 255, 0), thickness=2
                    )

        # 3. Render skeleton pose landmarks and connections
        if show_pose and self.pose_estimator is not None:
            pose_res = result.get("pose", {})
            if pose_res.get("has_pose", False):
                vis = self.pose_estimator.draw_pose(
                    vis,
                    pose_res,
                    draw_connections=True,
                    min_visibility=0.5,
                    landmark_color=(0, 255, 255),
                    connection_color=(255, 128, 0),
                    thickness=2,
                    radius=4,
                )

        # 4. Render detection bounding boxes
        if show_boxes:
            items = result.get("detections") or result.get("segmentations", [])
            for index, item in enumerate(items):
                x1, y1, x2, y2 = map(int, item["bbox"])
                cv2.rectangle(vis, (x1, y1), (x2, y2), bbox_color, 2)
                confidence = item.get("confidence", 0.0)
                label = f"Person {index + 1}: {confidence:.2f}"
                (lw, lh), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    vis,
                    (x1, max(y1 - lh - 6, 0)),
                    (x1 + lw + 6, max(y1, lh + 6)),
                    bbox_color,
                    -1,
                )
                cv2.putText(
                    vis,
                    label,
                    (x1 + 3, max(y1 - 4, lh + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

        # 5. Render HUD status bar
        if show_hud:
            person_count = result.get("person_count", 0)
            pose_detected = result.get("pose", {}).get("has_pose", False)
            status_text = f"FitRange Mirror | People: {person_count} | Pose: {'YES' if pose_detected else 'NO'}"
            if fps is not None:
                status_text += f" | FPS: {fps:.1f}"

            hud_bg = vis.copy()
            cv2.rectangle(hud_bg, (0, 0), (w, 36), (20, 20, 20), -1)
            cv2.addWeighted(hud_bg, 0.7, vis, 0.3, 0, vis)

            status_color = (0, 255, 0) if (person_count > 0 or pose_detected) else (120, 120, 255)
            cv2.circle(vis, (20, 18), 6, status_color, -1)
            cv2.putText(
                vis,
                status_text,
                (35, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return vis

    def close(self) -> None:
        """Release underlying pose and inference resources."""
        if self.pose_estimator is not None:
            self.pose_estimator.close()
