import cv2
import numpy as np
from pathlib import Path
from ai.pipeline.mirror_pipeline import FitRangeMirrorPipeline

project_root = Path(__file__).resolve().parent.parent
user_img_path = Path(r"C:\Users\dayan\.gemini\antigravity\brain\562e4ed7-27b0-47f2-bfee-f3103c29381b\.user_uploaded\media_1795509684702_76241b46.png")
if user_img_path.is_file():
    user_img = cv2.imread(str(user_img_path))
    print(f"User image shape: {user_img.shape}")

# Initialize pipeline with YOLO11s models (single person mode enabled)
pipeline = FitRangeMirrorPipeline(
    det_conf=0.40,
    seg_conf=0.40,
    single_person_mode=True,
    enable_detection=True,
    enable_segmentation=True,
    enable_pose=True,
)

sample_img_path = project_root / "dataset" / "val_annotated" / "images" / "front_facing_1.png"
if not sample_img_path.is_file():
    sample_img_path = project_root / "dataset" / "val" / "images" / "0.png"

im = cv2.imread(str(sample_img_path))
print(f"Sample input image shape: {im.shape}")

res = pipeline.process_frame(im)
print(f"Pipeline processed: has_person={res['has_person']}, person_count={res['person_count']}")
if res['detections']:
    print(f"Detection confidence: {res['detections'][0]['confidence']:.4f}")
if res['segmentations']:
    print(f"Segmentation confidence: {res['segmentations'][0]['confidence']:.4f}, mask_area: {res['segmentations'][0]['mask_area']}")

# Render the binary mask card
card = pipeline.render_binary_mask(res, show_card=True)
mask_raw = pipeline.render_binary_mask(res, show_card=False)

# Save card and mask to runs/sample_predictions
out_dir = project_root / "runs" / "sample_predictions"
out_dir.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out_dir / "yolo11s_binary_mask_card.png"), card)
cv2.imwrite(str(out_dir / "yolo11s_binary_mask_raw.png"), mask_raw)

# Save also into brain artifacts directory
artifact_dir = Path(r"C:\Users\dayan\.gemini\antigravity\brain\562e4ed7-27b0-47f2-bfee-f3103c29381b")
cv2.imwrite(str(artifact_dir / "fitrange_yolo11s_binary_mask_card.png"), card)
print(f"Saved binary mask card successfully to {out_dir / 'yolo11s_binary_mask_card.png'} and artifact dir.")
