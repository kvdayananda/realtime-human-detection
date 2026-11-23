"""Tool to convert Labelme JSON annotations into FitRange Mirror Ground Truth.

Converts:
1. Polygons labeled 'person' into YOLO normalized bounding boxes (class 0).
2. Polygons labeled 'person' into single-channel 8-bit PNG binary masks (0/255).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def convert_labelme_file(
    json_path: Path,
    out_det_dir: Path,
    out_seg_dir: Path,
    target_label: str = "person",
) -> Tuple[int, int]:
    """Convert a single Labelme JSON file into YOLO box text file and binary mask PNG.

    Args:
        json_path: Path to Labelme JSON file.
        out_det_dir: Directory to write YOLO .txt file.
        out_seg_dir: Directory to write single-channel PNG mask.
        target_label: Label name for human subjects (default: 'person').

    Returns:
        Tuple of (boxes_count, person_pixel_count).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    h = data.get("imageHeight")
    w = data.get("imageWidth")
    stem = json_path.stem

    if h is None or w is None:
        raise ValueError(f"Missing image dimensions in {json_path}")

    boxes = []
    mask = np.zeros((h, w), dtype=np.uint8)

    for shape in data.get("shapes", []):
        if shape.get("label", "").strip().lower() != target_label.lower():
            continue

        shape_type = shape.get("shape_type", "polygon")
        raw_points = shape.get("points", [])

        if shape_type == "rectangle" and len(raw_points) == 2:
            # Labelme rectangle format: [[x1, y1], [x2, y2]]
            x1, y1 = float(raw_points[0][0]), float(raw_points[0][1])
            x2, y2 = float(raw_points[1][0]), float(raw_points[1][1])
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            points = np.array(
                [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
                dtype=np.int32,
            )
        elif len(raw_points) >= 3:
            # Polygon format: at least 3 vertices
            points = np.array(raw_points, dtype=np.int32)
            x_min = float(points[:, 0].min())
            y_min = float(points[:, 1].min())
            x_max = float(points[:, 0].max())
            y_max = float(points[:, 1].max())
        else:
            continue

        # Rasterize polygon/rectangle into binary mask
        cv2.fillPoly(mask, [points], color=255)

        # Compute bounding box
        x_min = max(0.0, min(float(w), x_min))
        y_min = max(0.0, min(float(h), y_min))
        x_max = max(0.0, min(float(w), x_max))
        y_max = max(0.0, min(float(h), y_max))

        bw = x_max - x_min
        bh = y_max - y_min
        if bw <= 0 or bh <= 0:
            continue

        xc = (x_min + x_max) / (2.0 * w)
        yc = (y_min + y_max) / (2.0 * h)
        norm_w = bw / float(w)
        norm_h = bh / float(h)

        boxes.append(f"0 {xc:.6f} {yc:.6f} {norm_w:.6f} {norm_h:.6f}")

    # Write YOLO detection file
    det_out_path = out_det_dir / f"{stem}.txt"
    det_out_path.write_text("\n".join(boxes) + ("\n" if boxes else ""), encoding="utf-8")

    # Write binary mask PNG
    seg_out_path = out_seg_dir / f"{stem}.png"
    cv2.imwrite(str(seg_out_path), mask)

    return len(boxes), int((mask == 255).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Labelme JSON to ground-truth boxes and masks.")
    parser.add_argument("--json-dir", type=str, required=True, help="Directory containing Labelme .json files.")
    parser.add_argument("--out-det", type=str, required=True, help="Directory for YOLO .txt label files.")
    parser.add_argument("--out-seg", type=str, required=True, help="Directory for binary PNG masks.")
    parser.add_argument("--label", type=str, default="person", help="Target subject label (default: person).")
    args = parser.parse_args()

    json_dir = Path(args.json_dir).resolve()
    out_det = Path(args.out_det).resolve()
    out_seg = Path(args.out_seg).resolve()

    out_det.mkdir(parents=True, exist_ok=True)
    out_seg.mkdir(parents=True, exist_ok=True)

    json_files = sorted(list(json_dir.glob("*.json")))
    print(f"Found {len(json_files)} Labelme JSON files in {json_dir}")

    total_boxes = 0
    converted_count = 0
    for jf in json_files:
        if jf.stem.startswith("val_neg_"):
            print(f"[SKIPPED] {jf.name} is a negative control sample. Negative controls must remain empty (0 detections, all-black mask).")
            continue
        boxes_cnt, mask_pixels = convert_labelme_file(jf, out_det, out_seg, target_label=args.label)
        total_boxes += boxes_cnt
        converted_count += 1
        print(f"Converted {jf.name} -> {boxes_cnt} boxes, {mask_pixels} person pixels")

    print(f"Conversion complete. Converted {converted_count} positive files. Total boxes: {total_boxes}")


if __name__ == "__main__":
    main()
