"""FitRange Mirror - ONNX Runtime Inference for Human Instance Segmentation.

This module provides the `HumanSegmenterONNX` class, which performs high-performance
person instance segmentation directly via ONNX Runtime without requiring PyTorch or
Ultralytics at runtime.

Preprocessing, Postprocessing & Mask Reconstruction:
1. Letterbox Preprocessing:
   - Resizes image to fit inside 640x640 preserving original aspect ratio with neutral
     gray padding (114, 114, 114).
   - Converts BGR to RGB, scales to [0.0, 1.0], and transposes to NCHW format.
2. Dual Output Parsing:
   - Output 0: (1, 116, 8400) -> Transposed to (8400, 116).
     Columns 0:4 are [xc, yc, w, h], column 4 is score for class 0 ('person'),
     and columns 84:116 are the 32 mask coefficients.
   - Output 1: (1, 32, 160, 160) -> 32 prototype mask feature maps.
3. Mask Synthesis:
   - For each detected person after NMS:
     a. Compute linear combination of the 32 prototype masks with the detection's coefficients:
        mask_raw = sum(coeff_i * proto_i)
     b. Apply sigmoid activation to obtain probability map:
        mask_prob = 1.0 / (1.0 + exp(-mask_raw))
     c. Bilinearly upscale mask from 160x160 to 640x640 letterbox space.
     d. Crop mask strictly to the bounding box.
     e. Crop away letterbox padding and resize back to original image dimensions (W_orig, H_orig).
     f. Threshold at 0.5 to form binary mask (0 or 1).
     g. Extract external contour polygon using OpenCV findContours.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort


class HumanSegmenterONNX:
    """Standalone Person Instance Segmenter using ONNX Runtime."""

    PERSON_CLASS_ID: int = 0
    PERSON_CLASS_NAME: str = "person"

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        providers: Optional[List[str]] = None,
    ) -> None:
        """Initialize the ONNX Runtime Person Segmenter.

        Args:
            model_path: Path to ONNX segmentation model (.onnx). If None, defaults to
                        models/human_segmentation/human_segmentation.onnx.
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
                f"ONNX segmentation model not found at: {resolved_path}"
            )

        self.model_path = resolved_path

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
        output_names = [out.name for out in self._session.get_outputs()]
        self._output_boxes_name = output_names[0]
        self._output_protos_name = output_names[1]

    @staticmethod
    def _resolve_model_path(model_path: Optional[Union[str, Path]]) -> Path:
        """Resolve ONNX model path with project defaults."""
        if model_path is not None:
            return Path(model_path).resolve()

        project_root = Path(__file__).resolve().parent.parent.parent
        return (
            project_root
            / "models"
            / "human_segmentation"
            / "human_segmentation.onnx"
        )

    def segment(
        self,
        image: Union[str, Path, np.ndarray],
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Segment persons in an input image using ONNX Runtime.

        Args:
            image: Image file path (str or Path) or loaded image array (BGR uint8).
            conf_threshold: Optional confidence threshold override.
            iou_threshold: Optional NMS IoU threshold override.

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
                "detections": [...]
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

        # 2. Forward pass via ONNX Runtime
        outputs = self._session.run(
            [self._output_boxes_name, self._output_protos_name],
            {self._input_name: blob},
        )
        boxes_output, protos_output = outputs[0], outputs[1]

        # 3. Postprocess detections and reconstruct masks
        segmentations = self._postprocess(
            boxes_output=boxes_output,
            protos_output=protos_output,
            ratio=ratio,
            pad=pad,
            orig_shape=(orig_w, orig_h),
            conf_threshold=conf,
            iou_threshold=iou,
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
            image: Original image or path.
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

        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, :]
        return blob, ratio, (dx, dy)

    def _postprocess(
        self,
        boxes_output: np.ndarray,
        protos_output: np.ndarray,
        ratio: float,
        pad: Tuple[int, int],
        orig_shape: Tuple[int, int],
        conf_threshold: float,
        iou_threshold: float,
    ) -> List[Dict[str, Any]]:
        """Parse raw segmentation output, synthesize masks, and extract contours."""
        orig_w, orig_h = orig_shape
        dx, dy = pad

        # boxes_output: (1, 116, 8400) -> (8400, 116)
        preds = boxes_output[0].T
        # protos_output: (1, 32, 160, 160) -> (32, 160, 160)
        protos = protos_output[0]

        boxes: List[List[float]] = []
        scores: List[float] = []
        mask_coeffs: List[np.ndarray] = []

        # Filter person detections (class 0)
        for pred in preds:
            person_score = float(pred[4])
            if person_score >= conf_threshold:
                xc, yc, w, h = (
                    float(pred[0]),
                    float(pred[1]),
                    float(pred[2]),
                    float(pred[3]),
                )
                x1 = xc - w / 2.0
                y1 = yc - h / 2.0
                boxes.append([x1, y1, w, h])
                scores.append(person_score)
                mask_coeffs.append(pred[84:116])

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes,
            scores=scores,
            score_threshold=conf_threshold,
            nms_threshold=iou_threshold,
        )

        segmentations: List[Dict[str, Any]] = []

        if len(indices) > 0:
            flat_indices = (
                indices.flatten() if hasattr(indices, "flatten") else indices
            )

            for idx in flat_indices:
                x1, y1, w, h = boxes[idx]
                confidence = scores[idx]
                coeff = mask_coeffs[idx]

                # Map bounding box back to original unpadded frame
                orig_x1 = max(0.0, min(float(orig_w), (x1 - dx) / ratio))
                orig_y1 = max(0.0, min(float(orig_h), (y1 - dy) / ratio))
                orig_x2 = max(0.0, min(float(orig_w), (x1 + w - dx) / ratio))
                orig_y2 = max(0.0, min(float(orig_h), (y1 + h - dy) / ratio))

                if orig_x2 <= orig_x1 or orig_y2 <= orig_y1:
                    continue

                # Synthesize instance mask:
                # 1. Linear combination of 32 prototype masks: (160, 160)
                mask_160 = np.sum(coeff[:, None, None] * protos, axis=0)
                # 2. Sigmoid activation
                mask_prob = 1.0 / (1.0 + np.exp(-mask_160))
                # 3. Bilinear resize to 640x640 letterbox coordinates
                mask_640 = cv2.resize(
                    mask_prob, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR
                )

                # 4. Crop to detection bounding box in letterbox space
                bx1, by1 = int(max(0, x1)), int(max(0, y1))
                bx2, by2 = int(min(self.imgsz, x1 + w)), int(min(self.imgsz, y1 + h))
                cropped = np.zeros_like(mask_640)
                cropped[by1:by2, bx1:bx2] = mask_640[by1:by2, bx1:bx2]

                # 5. Unpad and resize to original image coordinates
                pad_w = self.imgsz - 2 * dx
                pad_h = self.imgsz - 2 * dy
                unpadded = cropped[dy : dy + pad_h, dx : dx + pad_w]
                mask_full = cv2.resize(
                    unpadded, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
                )

                # 6. Binary threshold at 0.5
                bin_mask = (mask_full > 0.5).astype(np.uint8)
                mask_area = int(bin_mask.sum())

                # 7. Extract external boundary polygon contour
                contours, _ = cv2.findContours(
                    bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                polygon: List[List[float]] = []
                if contours:
                    # Select largest contour
                    largest = max(contours, key=cv2.contourArea)
                    polygon = [
                        [round(float(pt[0][0]), 2), round(float(pt[0][1]), 2)]
                        for pt in largest
                    ]

                segmentations.append(
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
                        "polygon": polygon,
                        "mask_area": mask_area,
                    }
                )

        segmentations.sort(key=lambda d: d["confidence"], reverse=True)
        return segmentations


def main() -> None:
    """CLI runner for ONNX Runtime human segmentation."""
    parser = argparse.ArgumentParser(
        description="FitRange Mirror - ONNX Runtime Human Segmentation"
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
        help="Path to ONNX model. Default: models/human_segmentation/human_segmentation.onnx",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold (default: 0.35).",
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
    parser.add_argument(
        "--save-mask",
        type=str,
        default=None,
        help="Optional path to save full-resolution binary mask image (0 or 255).",
    )
    args = parser.parse_args()

    segmenter = HumanSegmenterONNX(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
    )
    output = segmenter.segment(args.image)

    if args.save_mask:
        mask = segmenter.get_composite_mask(args.image, segmentations=output["segmentations"])
        cv2.imwrite(args.save_mask, mask)
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
