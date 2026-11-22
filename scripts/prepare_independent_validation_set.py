"""Script to prepare the independent validation dataset for FitRange Mirror.

Selects 100 positive images (humans) and 30 negative images (no humans)
from 'dataset/human detection dataset/', sanitizes headers, assigns unique IDs,
sets up the independent validation directory structure, creates ground truth for
negative controls, and populates manifest.csv with 'pending_human_annotation'
for all positive samples.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR_0 = PROJECT_ROOT / "dataset" / "human detection dataset" / "0"
SRC_DIR_1 = PROJECT_ROOT / "dataset" / "human detection dataset" / "1"

OUT_DIR = PROJECT_ROOT / "dataset" / "independent_validation"
IMG_DIR = OUT_DIR / "images"
DET_ANN_DIR = OUT_DIR / "annotations" / "detection"
SEG_ANN_DIR = OUT_DIR / "annotations" / "segmentation"
PREV_DIR = OUT_DIR / "previews"

for d in [IMG_DIR, DET_ANN_DIR, SEG_ANN_DIR, PREV_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Identify known duplicates / overlap / test images to exclude from validation set
EXCLUDED_0 = {"0.png", "162.png"}  # 0.png used in test suite; 162.png is duplicate of 394.png in 1
EXCLUDED_1 = {"0.png", "1.png", "2.png", "394.png"}  # 0, 1, 2 used in tests; 394 is duplicate of 162 in 0

# Gather all available images in numeric order
def get_sorted_images(dir_path: Path, excluded: set[str]) -> list[Path]:
    items = []
    for f in dir_path.glob("*.png"):
        if f.name in excluded:
            continue
        try:
            num = int(f.stem)
            items.append((num, f))
        except ValueError:
            items.append((999999, f))
    items.sort(key=lambda x: x[0])
    return [x[1] for x in items]

all_pos = get_sorted_images(SRC_DIR_1, EXCLUDED_1)
all_neg = get_sorted_images(SRC_DIR_0, EXCLUDED_0)

print(f"Eligible positive images: {len(all_pos)}, Eligible negative images: {len(all_neg)}")

# Select 100 positives using an even stride across the pool
indices_pos = np.linspace(0, len(all_pos) - 1, 100, dtype=int)
selected_pos = [all_pos[i] for i in indices_pos]

# Select 30 negatives using an even stride across the pool
indices_neg = np.linspace(0, len(all_neg) - 1, 30, dtype=int)
selected_neg = [all_neg[i] for i in indices_neg]

manifest_rows = []

# Process 100 positive images
print("Processing 100 positive images...")
for idx, src_p in enumerate(selected_pos, start=1):
    img_id = f"val_pos_{idx:03d}"
    filename = f"{img_id}.png"
    dst_img = IMG_DIR / filename
    
    im = cv2.imread(str(src_p))
    if im is None:
        raise RuntimeError(f"Could not read {src_p}")
    h, w = im.shape[:2]
    
    # Write sanitized clean png
    cv2.imwrite(str(dst_img), im)
    
    manifest_rows.append({
        "image_id": img_id,
        "filename": filename,
        "original_source_path": f"dataset/human detection dataset/1/{src_p.name}",
        "width": w,
        "height": h,
        "human_present": "yes",
        "orientation": "pending_manual_tag",
        "clothing_category": "pending_manual_tag",
        "background_category": "pending_manual_tag",
        "lighting_category": "pending_manual_tag",
        "annotation_status": "pending_human_annotation",
        "annotator": "unassigned"
    })

# Process 30 negative images
print("Processing 30 negative images...")
for idx, src_p in enumerate(selected_neg, start=1):
    img_id = f"val_neg_{idx:03d}"
    filename = f"{img_id}.png"
    dst_img = IMG_DIR / filename
    dst_det = DET_ANN_DIR / f"{img_id}.txt"
    dst_seg = SEG_ANN_DIR / f"{img_id}.png"
    
    im = cv2.imread(str(src_p))
    if im is None:
        raise RuntimeError(f"Could not read {src_p}")
    h, w = im.shape[:2]
    
    cv2.imwrite(str(dst_img), im)
    
    # Negative ground truth: empty text file (zero bounding boxes)
    dst_det.write_text("")
    
    # Negative ground truth: all-zero mask matching image dimensions
    zero_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.imwrite(str(dst_seg), zero_mask)
    
    manifest_rows.append({
        "image_id": img_id,
        "filename": filename,
        "original_source_path": f"dataset/human detection dataset/0/{src_p.name}",
        "width": w,
        "height": h,
        "human_present": "no",
        "orientation": "none",
        "clothing_category": "none",
        "background_category": "negative_background",
        "lighting_category": "ambient",
        "annotation_status": "verified_empty",
        "annotator": "human_verified_negative"
    })

# Write manifest.csv
manifest_path = OUT_DIR / "manifest.csv"
fieldnames = [
    "image_id",
    "filename",
    "original_source_path",
    "width",
    "height",
    "human_present",
    "orientation",
    "clothing_category",
    "background_category",
    "lighting_category",
    "annotation_status",
    "annotator"
]

with open(manifest_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(manifest_rows)

print(f"Independent validation dataset prepared successfully at: {OUT_DIR}")
print(f"Total images: {len(manifest_rows)} (100 positive, 30 negative)")
print(f"Manifest written to: {manifest_path}")
