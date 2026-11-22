"""Validation script for FitRange Mirror Independent Human Annotations.

Enforces strict data integrity:
- Checks for missing detection .txt files or segmentation .png masks.
- Validates normalized YOLO coordinate ranges [0.0, 1.0], positive w, h, and class_id == 0.
- Verifies mask dimensions match source images exactly.
- Verifies mask pixel values are strictly {0, 255}.
- Verifies negative controls have empty detection files and all-zero masks.
- Verifies positive controls have valid non-empty detection files and non-empty masks.
- Validates manifest.csv entries against file inventory.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

def validate_annotations(
    base_dir: Path | str,
    strict_positives: bool = True,
) -> Dict[str, Any]:
    """Validate annotation files against image files and integrity rules.

    Args:
        base_dir: Path to dataset/independent_validation directory.
        strict_positives: If True, fails if positive images lack completed annotations.
                         If False, reports positive pending status without failing negative verification.

    Returns:
        Structured report dictionary with pass/fail status and detailed issue logs.
    """
    root = Path(base_dir).resolve()
    img_dir = root / "images"
    det_dir = root / "annotations" / "detection"
    seg_dir = root / "annotations" / "segmentation"
    manifest_file = root / "manifest.csv"

    if not img_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {img_dir}")
    if not det_dir.is_dir():
        raise FileNotFoundError(f"Detection annotations directory not found: {det_dir}")
    if not seg_dir.is_dir():
        raise FileNotFoundError(f"Segmentation annotations directory not found: {seg_dir}")

    # Load manifest if available
    manifest_records: Dict[str, Dict[str, str]] = {}
    if manifest_file.is_file():
        with open(manifest_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                manifest_records[row["image_id"]] = row

    images = sorted(list(img_dir.glob("*.png")))
    if not images:
        raise ValueError(f"No PNG images found in {img_dir}")

    errors: List[str] = []
    warnings: List[str] = []
    pending_positives: List[str] = []
    validated_negatives: List[str] = []
    validated_positives: List[str] = []

    for img_path in images:
        stem = img_path.stem
        is_negative = stem.startswith("val_neg")
        det_file = det_dir / f"{stem}.txt"
        seg_file = seg_dir / f"{stem}.png"

        # 1. Image loading and dimensions
        im = cv2.imread(str(img_path))
        if im is None:
            errors.append(f"[{stem}] Failed to read image file: {img_path.name}")
            continue
        h, w = im.shape[:2]

        # 2. Manifest check
        if stem in manifest_records:
            rec = manifest_records[stem]
            if int(rec["width"]) != w or int(rec["height"]) != h:
                errors.append(
                    f"[{stem}] Manifest resolution ({rec['width']}x{rec['height']}) "
                    f"does not match actual image resolution ({w}x{h})"
                )

        # 3. Detection file check
        if not det_file.is_file():
            if is_negative:
                errors.append(f"[{stem}] Negative control missing detection file: {det_file.name}")
            else:
                pending_positives.append(stem)
                if strict_positives:
                    errors.append(f"[{stem}] Positive sample missing detection annotation: {det_file.name}")
                else:
                    warnings.append(f"[{stem}] Positive sample pending detection annotation: {det_file.name}")
                continue
        else:
            det_content = det_file.read_text(encoding="utf-8").strip()
            if is_negative:
                if len(det_content) > 0:
                    errors.append(
                        f"[{stem}] Negative control detection file must be empty, but contains: {det_content[:50]}"
                    )
            else:
                if len(det_content) == 0:
                    pending_positives.append(stem)
                    if strict_positives:
                        errors.append(f"[{stem}] Positive sample has empty detection file: {det_file.name}")
                    else:
                        warnings.append(f"[{stem}] Positive sample pending annotation in: {det_file.name}")
                    continue
                else:
                    # Validate YOLO coordinates
                    lines = det_content.splitlines()
                    for line_idx, line in enumerate(lines, start=1):
                        parts = line.strip().split()
                        if len(parts) < 5:
                            errors.append(
                                f"[{stem}] Detection line {line_idx} has < 5 tokens: '{line}'"
                            )
                            continue
                        try:
                            cls_id = int(parts[0])
                            xc, yc, bw, bh = map(float, parts[1:5])
                        except ValueError as ve:
                            errors.append(
                                f"[{stem}] Detection line {line_idx} non-numeric token: {ve}"
                            )
                            continue

                        if cls_id != 0:
                            errors.append(
                                f"[{stem}] Line {line_idx} invalid class_id {cls_id} (must be 0 for person)"
                            )
                        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                            errors.append(
                                f"[{stem}] Line {line_idx} center ({xc}, {yc}) out of bounds [0, 1]"
                            )
                        if not (0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                            errors.append(
                                f"[{stem}] Line {line_idx} dimension ({bw}, {bh}) out of bounds (0, 1]"
                            )

        # 4. Segmentation mask check
        if not seg_file.is_file():
            if is_negative:
                errors.append(f"[{stem}] Negative control missing segmentation mask: {seg_file.name}")
            else:
                if strict_positives:
                    errors.append(f"[{stem}] Positive sample missing segmentation mask: {seg_file.name}")
                else:
                    warnings.append(f"[{stem}] Positive sample pending segmentation mask: {seg_file.name}")
        else:
            mask = cv2.imread(str(seg_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                errors.append(f"[{stem}] Failed to read mask file as grayscale: {seg_file.name}")
            else:
                if mask.shape != (h, w):
                    errors.append(
                        f"[{stem}] Mask dimensions {mask.shape} do not match image dimensions {(h, w)}"
                    )

                unique_vals = set(np.unique(mask).tolist())
                if not unique_vals.issubset({0, 255}):
                    errors.append(
                        f"[{stem}] Mask contains invalid pixel values {unique_vals}. Must be only {{0, 255}}"
                    )

                person_pixels = int((mask == 255).sum())

                if is_negative:
                    if person_pixels > 0:
                        errors.append(
                            f"[{stem}] Negative control mask must have 0 person pixels, but has {person_pixels}"
                        )
                    else:
                        validated_negatives.append(stem)
                else:
                    if person_pixels == 0:
                        if strict_positives:
                            errors.append(f"[{stem}] Positive sample mask has 0 person pixels")
                    else:
                        validated_positives.append(stem)

    passed = len(errors) == 0
    return {
        "passed": passed,
        "total_images": len(images),
        "total_errors": len(errors),
        "total_warnings": len(warnings),
        "validated_negatives": len(validated_negatives),
        "validated_positives": len(validated_positives),
        "pending_positives": len(pending_positives),
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate independent annotation integrity.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Path to dataset/independent_validation directory.",
    )
    parser.add_argument(
        "--allow-pending-positives",
        action="store_true",
        help="Permit positive images marked as pending human annotation without failing verification.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    print("=" * 60)
    print("      FITRANGE MIRROR - ANNOTATION VALIDATOR")
    print("=" * 60)
    print(f"Target Directory: {data_dir}")
    print(f"Mode: {'ALLOW PENDING POSITIVES' if args.allow_pending_positives else 'STRICT (ALL ANNOTATED)'}")
    print("-" * 60)

    try:
        report = validate_annotations(
            data_dir,
            strict_positives=not args.allow_pending_positives,
        )
    except Exception as e:
        print(f"FATAL: Validation failed to execute: {e}")
        sys.exit(1)

    print(f"Total Images Checked:       {report['total_images']}")
    print(f"Validated Negatives:        {report['validated_negatives']} / 30")
    print(f"Validated Positives:        {report['validated_positives']} / 100")
    print(f"Pending Positive Samples:   {report['pending_positives']} / 100")
    print(f"Total Errors:               {report['total_errors']}")
    print(f"Total Warnings:             {report['total_warnings']}")
    print("-" * 60)

    if report["errors"]:
        print("ERRORS DETECTED:")
        for err in report["errors"][:15]:
            print(f"  [ERROR] {err}")
        if len(report["errors"]) > 15:
            print(f"  ... and {len(report['errors']) - 15} more errors.")
        print("-" * 60)

    if report["warnings"]:
        print("STATUS / WARNINGS:")
        for w in report["warnings"][:10]:
            print(f"  [INFO] {w}")
        if len(report["warnings"]) > 10:
            print(f"  ... and {len(report['warnings']) - 10} more pending items.")
        print("-" * 60)

    if report["passed"]:
        print("VALIDATION RESULT: PASSED")
        sys.exit(0)
    else:
        print("VALIDATION RESULT: FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
