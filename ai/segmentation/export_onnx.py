"""FitRange Mirror - YOLO to ONNX Exporter for Human Instance Segmentation.

This script exports the PyTorch YOLO11n-seg human segmentation model checkpoint
to an optimized ONNX format for deployment with ONNX Runtime.

Usage:
    python -m ai.segmentation.export_onnx
    python ai/segmentation/export_onnx.py --weights models/human_segmentation/human_segmentation.pt --output models/human_segmentation/human_segmentation.onnx --imgsz 640
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Union

# Ensure Ultralytics settings and runs stay within the project directory to prevent permission errors
_DEFAULT_YOLO_CONFIG = (
    Path(__file__).resolve().parent.parent.parent / "models" / ".ultralytics"
)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_DEFAULT_YOLO_CONFIG))

from ultralytics import YOLO  # noqa: E402


def export_onnx(
    model_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    imgsz: int = 640,
    opset: int = 12,
    dynamic: bool = False,
) -> Path:
    """Export YOLO human segmentation checkpoint to ONNX format.

    Args:
        model_path: Path to input PyTorch weights (.pt). Default: models/human_segmentation/human_segmentation.pt
        output_path: Target path for the exported .onnx file. Default: models/human_segmentation/human_segmentation.onnx
        imgsz: Square input dimension in pixels (default: 640).
        opset: ONNX operator set version (default: 12).
        dynamic: Whether to support dynamic batch/image axes (default: False).

    Returns:
        Path to the verified exported ONNX model file.

    Raises:
        FileNotFoundError: If input weights file does not exist.
        RuntimeError: If ONNX export fails or output file is not generated.
    """
    project_root = Path(__file__).resolve().parent.parent.parent

    # 1. Resolve source checkpoint
    if model_path is None:
        source_pt = (
            project_root
            / "models"
            / "human_segmentation"
            / "human_segmentation.pt"
        )
    else:
        source_pt = Path(model_path).resolve()

    if not source_pt.is_file():
        raise FileNotFoundError(f"Source model weights not found at: {source_pt}")

    # 2. Resolve destination path
    if output_path is None:
        target_onnx = (
            project_root
            / "models"
            / "human_segmentation"
            / "human_segmentation.onnx"
        )
    else:
        target_onnx = Path(output_path).resolve()

    target_onnx.parent.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("FitRange Mirror - Human Segmentation ONNX Export")
    print("==================================================")
    print(f"Source PyTorch Model: {source_pt}")
    print(f"Target ONNX Model:    {target_onnx}")
    print(f"Input Resolution:     {imgsz}x{imgsz}")
    print(f"ONNX Opset:           {opset}")
    print(f"Dynamic Axes:         {dynamic}")
    print("--------------------------------------------------")

    # 3. Load YOLO segmentation model
    model = YOLO(str(source_pt))

    # 4. Perform export
    exported_result = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        dynamic=dynamic,
        simplify=False,
    )

    exported_file = Path(exported_result).resolve()

    # Move to target destination if placed elsewhere by Ultralytics
    if exported_file != target_onnx:
        if target_onnx.exists():
            target_onnx.unlink()
        exported_file.replace(target_onnx)
        final_path = target_onnx
    else:
        final_path = exported_file

    # 5. Verify exported file exists and has non-zero size
    if not final_path.is_file() or final_path.stat().st_size == 0:
        raise RuntimeError(
            f"ONNX export failed: Output file missing or empty at {final_path}"
        )

    file_size_mb = final_path.stat().st_size / (1024 * 1024)
    print(f" Export succeeded!")
    print(f"Verified ONNX file: {final_path} ({file_size_mb:.2f} MB)")
    print("==================================================")
    return final_path


def main() -> None:
    """CLI entry point for segmentation ONNX export."""
    parser = argparse.ArgumentParser(
        description="FitRange Mirror - Export Human Segmentation Model to ONNX"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to source PyTorch weights (.pt). Default: models/human_segmentation/human_segmentation.pt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for exported ONNX file. Default: models/human_segmentation/human_segmentation.onnx",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Square input resolution in pixels (default: 640).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX operator set version (default: 12).",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic input axes.",
    )
    args = parser.parse_args()

    export_onnx(
        model_path=args.weights,
        output_path=args.output,
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
    )


if __name__ == "__main__":
    main()
