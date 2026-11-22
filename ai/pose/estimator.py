"""FitRange Mirror - MediaPipe Pose Landmarker (Tasks API).

This module provides the `PoseEstimator` class wrapping Google MediaPipe Pose Landmarker
from the modern MediaPipe Tasks Vision API (mediapipe.tasks.python.vision) to extract
33 anatomical body keypoints in real time with high precision.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

logger = logging.getLogger("FitRangeMirror.Pose")

# Official 33 MediaPipe Pose Landmark Names
POSE_LANDMARK_NAMES: Dict[int, str] = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}

# Standard 35 MediaPipe Pose Skeleton Connections
# Extracted from vision.PoseLandmarksConnections or canonical BlazePose topology
try:
    POSE_CONNECTIONS: List[Tuple[int, int]] = [
        (c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS
    ]
except Exception:
    POSE_CONNECTIONS: List[Tuple[int, int]] = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
        (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
        (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
    ]


class PoseEstimator:
    """Standalone MediaPipe Pose Landmarker using modern MediaPipe Tasks API."""

    def __init__(
        self,
        model_asset_path: Optional[Union[str, Path]] = None,
        static_image_mode: bool = False,
        running_mode: Optional[vision.RunningMode] = None,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        num_poses: int = 1,
    ) -> None:
        """Initialize the MediaPipe Pose Landmarker.

        Args:
            model_asset_path: Path to the .task model file. Defaults to
                models/pose/pose_landmarker_full.task (or lite if complexity=0).
            static_image_mode: If True, uses RunningMode.IMAGE. If False, uses
                RunningMode.VIDEO with monotonic frame timestamps.
            running_mode: Explicit RunningMode override (IMAGE, VIDEO, or LIVE_STREAM).
            model_complexity: 0 (Lite), 1 (Full), 2 (Heavy). Used for default model selection.
            min_detection_confidence: Minimum confidence [0.0, 1.0] for pose detection.
            min_presence_confidence: Minimum confidence [0.0, 1.0] for landmark presence.
            min_tracking_confidence: Minimum confidence [0.0, 1.0] for landmark tracking.
            num_poses: Maximum number of poses to detect (default: 1).

        Raises:
            FileNotFoundError: If the .task model asset cannot be found.
        """
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.num_poses = num_poses

        # Determine running mode
        if running_mode is not None:
            self.running_mode = running_mode
        elif static_image_mode:
            self.running_mode = vision.RunningMode.IMAGE
        else:
            self.running_mode = vision.RunningMode.VIDEO

        # Resolve model asset path
        resolved_model_path = self._resolve_model_path(model_asset_path, model_complexity)
        if not resolved_model_path.is_file():
            logger.error("Pose Landmarker model asset not found at: %s", resolved_model_path)
            raise FileNotFoundError(
                f"Pose Landmarker model asset not found: {resolved_model_path}\n"
                "Download the official model asset from Google Cloud Storage:\n"
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
            )

        self.model_path = resolved_model_path
        logger.info(
            "Initializing MediaPipe Tasks PoseLandmarker (mode=%s, model=%s)",
            self.running_mode.name,
            self.model_path.name,
        )

        base_options = mp_tasks.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=self.running_mode,
            num_poses=self.num_poses,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_pose_presence_confidence=self.min_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_segmentation_masks=False,
        )

        self._landmarker: Optional[vision.PoseLandmarker] = (
            vision.PoseLandmarker.create_from_options(options)
        )
        self._last_timestamp_ms: int = -1

    @staticmethod
    def _resolve_model_path(
        user_path: Optional[Union[str, Path]],
        complexity: int,
    ) -> Path:
        """Resolve path to the .task model asset."""
        if user_path:
            return Path(user_path).resolve()

        project_root = Path(__file__).resolve().parent.parent.parent
        pose_models_dir = project_root / "models" / "pose"

        if complexity == 0:
            preferred = pose_models_dir / "pose_landmarker_lite.task"
            fallback = pose_models_dir / "pose_landmarker_full.task"
        else:
            preferred = pose_models_dir / "pose_landmarker_full.task"
            fallback = pose_models_dir / "pose_landmarker_lite.task"

        if preferred.is_file():
            return preferred
        if fallback.is_file():
            return fallback
        return preferred

    def estimate(
        self,
        image: Union[np.ndarray, str, Path],
        timestamp_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Estimate 3D body pose landmarks from an input image or video frame.

        Args:
            image: Input image as a NumPy array (BGR uint8) or valid image path.
            timestamp_ms: Monotonically increasing frame timestamp in milliseconds
                (used when running in VIDEO or LIVE_STREAM mode). If None, an auto-
                incrementing timestamp is maintained internally.

        Returns:
            Dictionary matching the standard FitRange Mirror pose schema:
            {
                "has_pose": bool,
                "landmarks": [
                    {
                        "id": int,
                        "name": str,
                        "x": float,          # Normalized [0.0, 1.0]
                        "y": float,          # Normalized [0.0, 1.0]
                        "z": float,          # Relative depth
                        "visibility": float, # Visibility score [0.0, 1.0]
                        "pixel_x": int,      # Pixel coordinate on image
                        "pixel_y": int       # Pixel coordinate on image
                    }
                ],
                "world_landmarks": [         # Real-world 3D coordinates in meters
                    {"id": int, "name": str, "x": float, "y": float, "z": float, "visibility": float}
                ]
            }

        Raises:
            FileNotFoundError: If image file path does not exist.
            ValueError: If image is empty or invalid.
        """
        if self._landmarker is None:
            raise RuntimeError("PoseEstimator has been closed and cannot perform inference.")

        img_bgr = self._validate_and_load(image)
        h, w = img_bgr.shape[:2]

        # Convert OpenCV BGR frame to RGB for MediaPipe Image
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        try:
            if self.running_mode == vision.RunningMode.IMAGE:
                result = self._landmarker.detect(mp_image)
            else:
                # VIDEO mode requires strictly monotonically increasing timestamp
                if timestamp_ms is not None:
                    ts = int(timestamp_ms)
                    if ts <= self._last_timestamp_ms:
                        ts = self._last_timestamp_ms + 1
                else:
                    now_ms = int(time.perf_counter() * 1000)
                    ts = now_ms if now_ms > self._last_timestamp_ms else self._last_timestamp_ms + 1

                self._last_timestamp_ms = ts
                result = self._landmarker.detect_for_video(mp_image, ts)
        except Exception as e:
            logger.error("Error during MediaPipe Tasks pose inference: %s", e, exc_info=True)
            raise

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return {
                "has_pose": False,
                "landmarks": [],
                "world_landmarks": [],
            }

        # Extract primary person landmarks
        raw_landmarks = result.pose_landmarks[0]
        landmarks: List[Dict[str, Any]] = []

        for idx, lm in enumerate(raw_landmarks):
            px = int(np.clip(lm.x * w, 0, w - 1))
            py = int(np.clip(lm.y * h, 0, h - 1))
            name = POSE_LANDMARK_NAMES.get(idx, f"landmark_{idx}")
            vis_score = getattr(lm, "visibility", 1.0)
            if vis_score is None:
                vis_score = 1.0

            landmarks.append(
                {
                    "id": idx,
                    "name": name,
                    "x": round(float(lm.x), 4),
                    "y": round(float(lm.y), 4),
                    "z": round(float(lm.z), 4),
                    "visibility": round(float(vis_score), 4),
                    "pixel_x": px,
                    "pixel_y": py,
                }
            )

        world_landmarks: List[Dict[str, Any]] = []
        if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
            raw_world = result.pose_world_landmarks[0]
            for idx, wlm in enumerate(raw_world):
                name = POSE_LANDMARK_NAMES.get(idx, f"landmark_{idx}")
                vis_score = getattr(wlm, "visibility", 1.0)
                if vis_score is None:
                    vis_score = 1.0

                world_landmarks.append(
                    {
                        "id": idx,
                        "name": name,
                        "x": round(float(wlm.x), 4),
                        "y": round(float(wlm.y), 4),
                        "z": round(float(wlm.z), 4),
                        "visibility": round(float(vis_score), 4),
                    }
                )

        return {
            "has_pose": True,
            "landmarks": landmarks,
            "world_landmarks": world_landmarks,
        }

    # Backward-compatible alias
    estimate_pose = estimate

    def draw_pose(
        self,
        image: np.ndarray,
        pose_result: Dict[str, Any],
        draw_connections: bool = True,
        min_visibility: float = 0.5,
        landmark_color: Tuple[int, int, int] = (0, 255, 255),  # Yellow
        connection_color: Tuple[int, int, int] = (255, 128, 0),  # Cyan-blue
        thickness: int = 2,
        radius: int = 4,
    ) -> np.ndarray:
        """Render skeleton lines and landmark points directly with OpenCV.

        Operates independently of legacy mp.solutions.drawing_utils.

        Args:
            image: Frame to draw upon (BGR uint8).
            pose_result: Output dictionary from `estimate()`.
            draw_connections: Whether to draw connecting skeleton lines between joints.
            min_visibility: Minimum visibility threshold to render a keypoint/line.
            landmark_color: Color tuple (B, G, R) for keypoint circles.
            connection_color: Color tuple (B, G, R) for limb lines.
            thickness: Line thickness in pixels.
            radius: Keypoint circle radius in pixels.

        Returns:
            np.ndarray with rendered pose overlays.
        """
        vis = image.copy()
        if not pose_result.get("has_pose", False):
            return vis

        landmarks = pose_result.get("landmarks", [])
        if not landmarks:
            return vis

        lm_by_id = {lm["id"]: lm for lm in landmarks}

        # 1. Draw skeleton connections
        if draw_connections:
            for start_idx, end_idx in POSE_CONNECTIONS:
                lm1 = lm_by_id.get(start_idx)
                lm2 = lm_by_id.get(end_idx)
                if (
                    lm1
                    and lm2
                    and lm1.get("visibility", 1.0) >= min_visibility
                    and lm2.get("visibility", 1.0) >= min_visibility
                ):
                    pt1 = (lm1["pixel_x"], lm1["pixel_y"])
                    pt2 = (lm2["pixel_x"], lm2["pixel_y"])
                    cv2.line(
                        vis,
                        pt1,
                        pt2,
                        connection_color,
                        thickness=thickness,
                        lineType=cv2.LINE_AA,
                    )

        # 2. Draw keypoint circles
        for lm in landmarks:
            if lm.get("visibility", 1.0) >= min_visibility:
                pt = (lm["pixel_x"], lm["pixel_y"])
                cv2.circle(
                    vis,
                    pt,
                    radius,
                    landmark_color,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )
                cv2.circle(
                    vis,
                    pt,
                    radius + 1,
                    (0, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

        return vis

    def close(self) -> None:
        """Release MediaPipe Tasks resources."""
        if hasattr(self, "_landmarker") and self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None

    def __enter__(self) -> PoseEstimator:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    @staticmethod
    def _validate_and_load(image: Union[np.ndarray, str, Path]) -> np.ndarray:
        """Validate input type and convert to BGR uint8 NumPy array."""
        if isinstance(image, (str, Path)):
            path = Path(image).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found: {path}")
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Failed to read image from: {path}")
            return img

        if isinstance(image, np.ndarray):
            if image.size == 0:
                raise ValueError("Image array is empty (size 0).")
            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if image.ndim == 3:
                if image.shape[2] == 4:
                    return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                if image.shape[2] == 3:
                    return image
                raise ValueError(
                    f"Unexpected number of channels: {image.shape[2]}"
                )
            raise ValueError(f"Expected 2D or 3D array, got ndim={image.ndim}")

        raise TypeError(
            f"Unsupported image type: {type(image)}. Expected np.ndarray, str, or Path."
        )


def main() -> None:
    """CLI runner for standalone pose estimation."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="FitRange Mirror - MediaPipe Tasks Pose Estimation"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image file for pose estimation.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional path to .task model asset.",
    )
    parser.add_argument(
        "--complexity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Model complexity: 0 (Lite), 1 (Full), 2 (Heavy). Default: 1",
    )
    args = parser.parse_args()

    estimator = PoseEstimator(
        model_asset_path=args.model,
        static_image_mode=True,
        model_complexity=args.complexity,
    )
    try:
        output = estimator.estimate(args.image)
        summary = {
            "has_pose": output["has_pose"],
            "landmark_count": len(output["landmarks"]),
            "key_landmarks": {
                lm["name"]: {
                    "pixel": (lm["pixel_x"], lm["pixel_y"]),
                    "visibility": lm["visibility"],
                }
                for lm in output["landmarks"]
                if lm["name"]
                in {
                    "nose",
                    "left_shoulder",
                    "right_shoulder",
                    "left_hip",
                    "right_hip",
                    "left_knee",
                    "right_knee",
                    "left_ankle",
                    "right_ankle",
                }
            },
        }
        print(json.dumps(summary, indent=2))
    finally:
        estimator.close()


if __name__ == "__main__":
    main()
