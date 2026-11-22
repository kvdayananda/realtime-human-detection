"""FitRange Mirror - Standalone Human Instance Segmentation (PyTorch / YOLO).

This module provides the `HumanSegmenter` class for detecting and segmenting humans
in static images or video frames using Ultralytics YOLO11n-seg. It isolates strictly
the 'person' class (ID 0) and outputs instance bounding boxes and pixel-level boundary
polygon contours.
"""

from __future__ import annotations

import argparse
import json
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


class HumanSegmenter:
    """Standalone Person Instance Segmenter using YOLO11n-seg."""

    PERSON_CLASS_ID: int = 0
    PERSON_CLASS_NAME: str = "person"

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        conf_threshold: float = 0.35,
        imgsz: int = 640,
        device: str = "cpu",
    ) -> None:
        """Initialize the HumanSegmenter.

        Args:
            model_path: Path to YOLO segmentation weights (.pt). If None, defaults to
                        models/human_segmentation/human_segmentation.pt.
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
                f"Segmentation checkpoint not found at: {resolved_path}"
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
            / "human_segmentation"
            / "human_segmentation.pt"
        )
        return primary

    def segment(
        self,
        image: Union[str, Path, np.ndarray],
        conf_threshold: Optional[float] = None,
        imgsz: Optional[int] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Segment persons in an input image.

        Args:
            image: Image file path (str or Path) or loaded image as a NumPy array (BGR/RGB).
            conf_threshold: Optional per-call confidence threshold override.
            imgsz: Optional per-call image size override.

        Returns:
            Dictionary containing segmentation instances:
            {
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
                "detections": [...]  # Aliased for backward compatibility
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

        # Run segmentation inference strictly filtering for person class (ID 0)
        results = self._model.predict(
            source=input_data,
            classes=[self.PERSON_CLASS_ID],
            conf=conf,
            imgsz=size,
            device=self.device,
            verbose=False,
        )

        segmentations: List[Dict[str, Any]] = []

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                has_masks = (
                    result.masks is not None and len(result.masks.data) > 0
                )

                for index, box in enumerate(result.boxes):
                    coords = [round(float(v), 2) for v in box.xyxy[0].tolist()]
                    confidence = round(float(box.conf[0]), 4)
                    class_id = int(box.cls[0].item())

                    # Extract polygon contour points
                    polygon: List[List[float]] = []
                    mask_area: int = 0

                    if has_masks and index < len(result.masks.xy):
                        poly_array = result.masks.xy[index]
                        if len(poly_array) >= 3:
                            polygon = [
                                [round(float(pt[0]), 2), round(float(pt[1]), 2)]
                                for pt in poly_array
                            ]
                            mask_area = int(
                                cv2.contourArea(poly_array.astype(np.float32))
                            )

                    segmentations.append(
                        {
                            "bbox": coords,
                            "confidence": confidence,
                            "class_id": class_id,
                            "class_name": self.PERSON_CLASS_NAME,
                            "polygon": polygon,
                            "mask_area": mask_area,
                        }
                    )

        return {
            "segmentations": segmentations,
            "detections": segmentations,
        }

    def get_composite_mask(
        self,
        image: Union[str, Path, np.ndarray],
        segmentations: Optional[List[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        """Create a full-resolution single-channel binary mask (0 or 255) for all segmented persons.

        Args:
            image: Original image or path (used to determine canvas dimensions).
            segmentations: Pre-computed list of segmentation items. If None, inference is executed.

        Returns:
            np.ndarray of shape (H, W) with dtype uint8 (255 for person, 0 for background).
        """
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Cannot read image from: {image}")
            h, w = img.shape[:2]
        elif isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        if segmentations is None:
            res = self.segment(image)
            segmentations = res["segmentations"]

        composite = np.zeros((h, w), dtype=np.uint8)

        for item in segmentations:
            poly = item.get("polygon", [])
            if len(poly) >= 3:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(composite, [pts], color=255)

        return composite

    @staticmethod
    def _validate_and_prepare_input(
        image: Union[str, Path, np.ndarray],
    ) -> Union[str, np.ndarray]:
        """Validate input type, existence, and dimensions."""
        if isinstance(image, (str, Path)):
            path = Path(image).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found: {path}")

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
    """CLI runner for standalone human segmentation."""
    parser = argparse.ArgumentParser(
        description="FitRange Mirror - Standalone Human Segmentation"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image file for segmentation.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to YOLO segmentation weights (.pt). Default: models/human_segmentation/human_segmentation.pt",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold (default: 0.35).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image resolution (default: 640).",
    )
    args = parser.parse_args()

    segmenter = HumanSegmenter(
        model_path=args.model,
        conf_threshold=args.conf,
        imgsz=args.imgsz,
    )
    output = segmenter.segment(args.image)
    # Output compact polygon preview in CLI to avoid flooding terminal
    cli_output = {
        "segmentations": [
            {
                "bbox": s["bbox"],
                "confidence": s["confidence"],
                "class_id": s["class_id"],
                "class_name": s["class_name"],
                "polygon_points_count": len(s["polygon"]),
                "polygon_sample": s["polygon"][:3],
                "mask_area": s["mask_area"],
            }
            for s in output["segmentations"]
        ]
    }
    print(json.dumps(cli_output, indent=2))


if __name__ == "__main__":
    main()
