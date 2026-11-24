"""FitRange Mirror - Standalone Human Detection (PyTorch / YOLO).

This module provides the `HumanDetector` class for detecting human presence in static images
or video frames using Ultralytics YOLO11n. It detects strictly the 'person' class (ID 0)
and does not require webcam access, ByteTrack, or MediaPipe.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

# Ensure Ultralytics settings and runs stay within the project directory to prevent permission errors
_DEFAULT_YOLO_CONFIG = (
    Path(__file__).resolve().parent.parent.parent / "models" / ".ultralytics"
)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_DEFAULT_YOLO_CONFIG))

from ultralytics import YOLO  # noqa: E402


class HumanDetector:
    """Standalone Person Detector using YOLO11n."""

    PERSON_CLASS_ID: int = 0
    PERSON_CLASS_NAME: str = "person"

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        conf_threshold: float = 0.40,
        imgsz: int = 640,
        device: str = "cpu",
    ) -> None:
        """Initialize the HumanDetector.

        Args:
            model_path: Path to YOLO weights (.pt). If None, defaults to
                        models/human_detection/human_detection.pt or fallback models/yolo11n.pt.
            conf_threshold: Default confidence threshold for person detection (0.0 to 1.0).
            imgsz: Default inference image resolution (square size in pixels).
            device: Execution device, e.g. 'cpu', 'cuda', or '0'.

        Raises:
            FileNotFoundError: If the specified model checkpoint file does not exist.
            ValueError: If confidence threshold or image size is invalid.
        """
        if not (0.0 <= conf_threshold <= 1.0):
            raise ValueError(
                f"conf_threshold must be between 0.0 and 1.0, got {conf_threshold}"
            )
        if imgsz <= 0:
            raise ValueError(f"imgsz must be positive, got {imgsz}")

        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.device = device

        resolved_path = self._resolve_model_path(model_path)
        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found at: {resolved_path}"
            )

        self.model_path = resolved_path
        self._model = YOLO(str(self.model_path))

    @staticmethod
    def _resolve_model_path(model_path: Optional[Union[str, Path]]) -> Path:
        """Resolve model checkpoint path with sensible project defaults."""
        if model_path is not None:
            return Path(model_path).resolve()

        project_root = Path(__file__).resolve().parent.parent.parent
        primary = (
            project_root
            / "models"
            / "human_detection"
            / "human_detection.pt"
        )
        if primary.exists():
            return primary

        fallback = project_root / "models" / "yolo11n.pt"
        return fallback

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        conf_threshold: Optional[float] = None,
        imgsz: Optional[int] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Detect persons in an input image.

        Args:
            image: Image file path (str or Path) or loaded image as a NumPy array (BGR/RGB).
            conf_threshold: Optional per-call confidence threshold override.
            imgsz: Optional per-call image size override.

        Returns:
            Dictionary matching the required specification:
            {
                "detections": [
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float,
                        "class_id": 0,
                        "class_name": "person"
                    }
                ]
            }

        Raises:
            FileNotFoundError: If image file path does not exist.
            ValueError: If input image is empty, corrupt, or unsupported.
            TypeError: If input is neither a path nor a NumPy array.
        """
        conf = (
            self.conf_threshold if conf_threshold is None else conf_threshold
        )
        size = self.imgsz if imgsz is None else imgsz

        if not (0.0 <= conf <= 1.0):
            raise ValueError(
                f"conf_threshold must be between 0.0 and 1.0, got {conf}"
            )
        if size <= 0:
            raise ValueError(f"imgsz must be positive, got {size}")

        input_data = self._validate_and_prepare_input(image)

        # Run inference strictly filtering for person class (ID 0)
        results = self._model.predict(
            source=input_data,
            classes=[self.PERSON_CLASS_ID],
            conf=conf,
            imgsz=size,
            device=self.device,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    coords = [round(float(v), 2) for v in box.xyxy[0].tolist()]
                    confidence = round(float(box.conf[0]), 4)
                    class_id = int(box.cls[0].item())

                    detections.append(
                        {
                            "bbox": coords,
                            "confidence": confidence,
                            "class_id": class_id,
                            "class_name": self.PERSON_CLASS_NAME,
                        }
                    )

        return {"detections": detections}

    @staticmethod
    def _validate_and_prepare_input(
        image: Union[str, Path, np.ndarray],
    ) -> Union[str, np.ndarray]:
        """Validate input type, existence, and dimensions."""
        if isinstance(image, (str, Path)):
            path = Path(image).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found: {path}")

            # Verify it can be decoded
            test_read = cv2.imread(str(path))
            if test_read is None:
                raise ValueError(
                    f"Unable to read/decode image file from: {path}"
                )
            return str(path)

        if isinstance(image, np.ndarray):
            if image.size == 0:
                raise ValueError("Input image array is empty (size 0).")
            if image.ndim not in (2, 3):
                raise ValueError(
                    f"Expected 2D or 3D image array, got ndim={image.ndim}"
                )
            return image

        raise TypeError(
            f"Unsupported image input type: {type(image)}. Expected str, Path, or np.ndarray."
        )


def main() -> None:
    """CLI test runner for standalone human detection."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="FitRange Mirror - Standalone Human Detection"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image file for detection.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to YOLO weights (.pt). Default: models/human_detection/human_detection.pt",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Confidence threshold (default: 0.40).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image resolution (default: 640).",
    )
    args = parser.parse_args()

    detector = HumanDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        imgsz=args.imgsz,
    )
    output = detector.detect(args.image)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
