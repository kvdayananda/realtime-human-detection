"""FitRange Mirror - ONNX Runtime Inference for Human Detection.

This module provides the `HumanDetectorONNX` class, which performs high-performance
person detection inference directly via ONNX Runtime without requiring PyTorch or
Ultralytics at runtime.

Preprocessing & Coordinate Mapping:
1. Letterbox Preprocessing:
   - The input image is resized to fit inside the model input dimension (default 640x640)
     while strictly preserving its original aspect ratio.
   - Scale factor: r = min(imgsz / orig_w, imgsz / orig_h)
   - New unpadded dimensions: nw = round(orig_w * r), nh = round(orig_h * r)
   - Symmetric padding: dx = (imgsz - nw) / 2, dy = (imgsz - nh) / 2
   - Padding fill color: neutral gray (114, 114, 114).
2. Tensor Normalization:
   - Pixel values are converted from BGR to RGB, normalized to [0.0, 1.0], and
     transposed to NCHW layout: (1, 3, imgsz, imgsz).
3. Output Parsing & Non-Maximum Suppression (NMS):
   - Model output shape is (1, 84, 8400) for YOLO11.
   - Transposed to (8400, 84) where each row contains:
     [center_x, center_y, width, height, class_0_score, ..., class_79_score].
   - Filtered for Class 0 ('person') score >= conf_threshold.
   - Bounding boxes are converted from center (xc, yc, w, h) to corner (x1, y1, w, h)
     and processed through OpenCV DNN NMS.
4. Inverse Coordinate Mapping:
   - To project bounding-box coordinates back onto the original unpadded image:
     orig_x1 = (x1 - dx) / r
     orig_y1 = (y1 - dy) / r
     orig_x2 = (x1 + w - dx) / r
     orig_y2 = (y1 + h - dy) / r
   - Clamped to [0, orig_w] and [0, orig_h].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort


class HumanDetectorONNX:
    """Standalone Person Detector using ONNX Runtime."""

    PERSON_CLASS_ID: int = 0
    PERSON_CLASS_NAME: str = "person"

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        conf_threshold: float = 0.40,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        providers: Optional[List[str]] = None,
    ) -> None:
        """Initialize the ONNX Runtime Person Detector.

        Args:
            model_path: Path to ONNX model file (.onnx). If None, defaults to
                        models/human_detection/human_detection.onnx.
            conf_threshold: Default confidence threshold for detection (0.0 to 1.0).
            iou_threshold: Intersection-over-Union threshold for NMS.
            imgsz: Input resolution in pixels (default: 640).
            providers: List of execution providers (e.g. ['CPUExecutionProvider']).

        Raises:
            FileNotFoundError: If ONNX model file does not exist.
            ValueError: If thresholds or dimensions are invalid.
        """
        if not (0.0 <= conf_threshold <= 1.0):
            raise ValueError(
                f"conf_threshold must be between 0.0 and 1.0, got {conf_threshold}"
            )
        if not (0.0 <= iou_threshold <= 1.0):
            raise ValueError(
                f"iou_threshold must be between 0.0 and 1.0, got {iou_threshold}"
            )
        if imgsz <= 0:
            raise ValueError(f"imgsz must be positive, got {imgsz}")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz

        resolved_path = self._resolve_model_path(model_path)
        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"ONNX model file not found at: {resolved_path}"
            )

        self.model_path = resolved_path

        # Configure ONNX Runtime session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        if providers is None:
            providers = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=providers,
        )

        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    @staticmethod
    def _resolve_model_path(model_path: Optional[Union[str, Path]]) -> Path:
        """Resolve ONNX model path with project defaults."""
        if model_path is not None:
            return Path(model_path).resolve()

        project_root = Path(__file__).resolve().parent.parent.parent
        return (
            project_root
            / "models"
            / "human_detection"
            / "human_detection.onnx"
        )

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Detect persons in an input image using ONNX Runtime.

        Args:
            image: Image file path (str or Path) or loaded image array (BGR uint8).
            conf_threshold: Optional confidence threshold override.
            iou_threshold: Optional NMS IoU threshold override.

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
            ValueError: If input image is empty or invalid.
            TypeError: If input is neither a path nor a NumPy array.
        """
        conf = (
            self.conf_threshold if conf_threshold is None else conf_threshold
        )
        iou = self.iou_threshold if iou_threshold is None else iou_threshold

        if not (0.0 <= conf <= 1.0):
            raise ValueError(
                f"conf_threshold must be between 0.0 and 1.0, got {conf}"
            )
        if not (0.0 <= iou <= 1.0):
            raise ValueError(
                f"iou_threshold must be between 0.0 and 1.0, got {iou}"
            )

        img_bgr = self._load_image(image)
        orig_h, orig_w = img_bgr.shape[:2]

        # 1. Preprocess: letterbox resize & normalize
        blob, ratio, pad = self._preprocess(img_bgr, self.imgsz)

        # 2. Inference: ONNX Runtime forward pass
        outputs = self._session.run(
            [self._output_name],
            {self._input_name: blob},
        )[0]

        # 3. Postprocess: parse detections, NMS, inverse coordinate mapping
        detections = self._postprocess(
            outputs=outputs,
            ratio=ratio,
            pad=pad,
            orig_shape=(orig_w, orig_h),
            conf_threshold=conf,
            iou_threshold=iou,
        )

        return {"detections": detections}

    @staticmethod
    def _load_image(image: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Load and validate image into standard BGR NumPy array."""
        if isinstance(image, (str, Path)):
            path = Path(image).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found: {path}")
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(
                    f"Unable to read/decode image file from: {path}"
                )
            return img

        if isinstance(image, np.ndarray):
            if image.size == 0:
                raise ValueError("Input image array is empty.")
            if image.ndim == 2:
                # Convert grayscale to BGR
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if image.ndim == 3:
                if image.shape[2] == 4:
                    return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                return image
            raise ValueError(
                f"Expected 2D or 3D image array, got ndim={image.ndim}"
            )

        raise TypeError(
            f"Unsupported image input type: {type(image)}. Expected str, Path, or np.ndarray."
        )

    @staticmethod
    def _preprocess(
        image: np.ndarray,
        imgsz: int,
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Letterbox resize with neutral gray padding to square imgsz x imgsz."""
        orig_h, orig_w = image.shape[:2]
        ratio = min(imgsz / orig_w, imgsz / orig_h)
        new_w = int(round(orig_w * ratio))
        new_h = int(round(orig_h * ratio))

        resized = cv2.resize(
            image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )

        padded = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        dx = (imgsz - new_w) // 2
        dy = (imgsz - new_h) // 2
        padded[dy : dy + new_h, dx : dx + new_w] = resized

        # Convert BGR -> RGB, scale to [0, 1], transpose to NCHW
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, :]
        return blob, ratio, (dx, dy)

    def _postprocess(
        self,
        outputs: np.ndarray,
        ratio: float,
        pad: Tuple[int, int],
        orig_shape: Tuple[int, int],
        conf_threshold: float,
        iou_threshold: float,
    ) -> List[Dict[str, Any]]:
        """Parse raw ONNX output, filter class 0, apply NMS, and unpad coordinates."""
        orig_w, orig_h = orig_shape
        dx, dy = pad

        # outputs shape: (1, 84, 8400) -> transpose to (8400, 84)
        predictions = outputs[0].T

        boxes: List[List[float]] = []
        scores: List[float] = []

        # In YOLO11 detection output:
        # columns 0:4 are [xc, yc, w, h] in model input coordinates (640x640)
        # column 4 is score for class 0 ('person')
        for pred in predictions:
            person_score = float(pred[4])
            if person_score >= conf_threshold:
                xc = float(pred[0])
                yc = float(pred[1])
                w = float(pred[2])
                h = float(pred[3])

                x1 = xc - w / 2.0
                y1 = yc - h / 2.0

                boxes.append([x1, y1, w, h])
                scores.append(person_score)

        if not boxes:
            return []

        # OpenCV DNN NMS expects boxes as [x, y, w, h]
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes,
            scores=scores,
            score_threshold=conf_threshold,
            nms_threshold=iou_threshold,
        )

        detections: List[Dict[str, Any]] = []

        if len(indices) > 0:
            flat_indices = (
                indices.flatten() if hasattr(indices, "flatten") else indices
            )
            for idx in flat_indices:
                x1, y1, w, h = boxes[idx]
                confidence = scores[idx]

                # Map back from padded 640x640 space to original image coordinates
                orig_x1 = max(0.0, min(float(orig_w), (x1 - dx) / ratio))
                orig_y1 = max(0.0, min(float(orig_h), (y1 - dy) / ratio))
                orig_x2 = max(0.0, min(float(orig_w), (x1 + w - dx) / ratio))
                orig_y2 = max(0.0, min(float(orig_h), (y1 + h - dy) / ratio))

                # Sanity check: ensure valid box
                if orig_x2 > orig_x1 and orig_y2 > orig_y1:
                    detections.append(
                        {
                            "bbox": [
                                round(orig_x1, 2),
                                round(orig_y1, 2),
                                round(orig_x2, 2),
                                round(orig_y2, 2),
                            ],
                            "confidence": round(confidence, 4),
                            "class_id": self.PERSON_CLASS_ID,
                            "class_name": self.PERSON_CLASS_NAME,
                        }
                    )

        # Sort detections by confidence descending
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections


def main() -> None:
    """CLI runner for ONNX Runtime human detection."""
    parser = argparse.ArgumentParser(
        description="FitRange Mirror - ONNX Runtime Human Detection"
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
        help="Path to ONNX model. Default: models/human_detection/human_detection.onnx",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Confidence threshold (default: 0.40).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (default: 0.45).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image resolution (default: 640).",
    )
    args = parser.parse_args()

    detector = HumanDetectorONNX(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
    )
    output = detector.detect(args.image)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
