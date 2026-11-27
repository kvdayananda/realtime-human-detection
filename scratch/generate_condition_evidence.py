import json
import cv2
import numpy as np
from pathlib import Path
from ai.pipeline.mirror_pipeline import FitRangeMirrorPipeline

project_root = Path(__file__).resolve().parent.parent
img_dir = project_root / "dataset" / "val_annotated" / "images"
masks_dir = project_root / "dataset" / "val_annotated" / "masks"
out_dir = project_root / "runs" / "condition_evidence"
out_dir.mkdir(parents=True, exist_ok=True)
artifact_dir = Path(r"C:\Users\dayan\.gemini\antigravity\brain\562e4ed7-27b0-47f2-bfee-f3103c29381b")

pipeline = FitRangeMirrorPipeline(
    det_conf=0.35,
    seg_conf=0.35,
    pose_conf=0.40,
    single_person_mode=True,
    enable_detection=True,
    enable_segmentation=True,
    enable_pose=True,
)

conditions_to_test = [
    {
        "id": "front_facing",
        "condition_name": "Front-facing person",
        "files": ["front_facing_1.png", "front_facing_2.png"],
        "description": "Standing human directly facing camera with full frontal visibility of head, torso, arms, and legs.",
        "expected_human": True,
    },
    {
        "id": "side_facing",
        "condition_name": "Side-facing person",
        "files": ["side_facing_1.png"],
        "description": "Standing human in 90-degree lateral profile view with contralateral limb occlusion.",
        "expected_human": True,
    },
    {
        "id": "different_clothing",
        "condition_name": "Different clothing",
        "files": ["athletic_clothing.png", "casual_clothing.png"],
        "description": "Form-fitting athletic sportswear (shorts, training top) vs loose casual garments (jacket, relaxed trousers).",
        "expected_human": True,
    },
    {
        "id": "different_body_shapes",
        "condition_name": "Different body shapes",
        "files": ["athletic_clothing.png", "front_facing_1.png", "casual_clothing.png", "side_facing_1.png"],
        "description": "Athletic muscular build, standard lean stature, broader casual profile, and lateral slim profile.",
        "expected_human": True,
    },
    {
        "id": "different_backgrounds",
        "condition_name": "Different backgrounds",
        "files": ["front_facing_1.png", "cluttered_background.png", "outdoor_daylight.png", "negative_room.png", "negative_furniture.png"],
        "description": "Plain white wall, cluttered home gym interior, outdoor landscape, and human-absent negative control rooms.",
        "expected_human": None, # contains both positive and negative
    },
    {
        "id": "different_lighting",
        "condition_name": "Different lighting conditions",
        "files": ["outdoor_daylight.png", "front_facing_1.png", "indoor_lighting.png"],
        "description": "High-dynamic-range direct sunlight, even neutral indoor illumination, and warm ambient indoor lighting with shadows.",
        "expected_human": True,
    },
]

evidence_results = {}

for cond in conditions_to_test:
    cid = cond["id"]
    evidence_results[cid] = {
        "condition_name": cond["condition_name"],
        "description": cond["description"],
        "samples": []
    }
    for fname in cond["files"]:
        fpath = img_dir / fname
        if not fpath.is_file():
            print(f"File not found: {fname}")
            continue
        im = cv2.imread(str(fpath))
        h, w = im.shape[:2]

        res = pipeline.process_frame(im)
        has_person = res["has_person"]
        p_count = res["person_count"]

        det_conf = res["detections"][0]["confidence"] if res["detections"] else 0.0
        det_bbox = res["detections"][0]["bbox"] if res["detections"] else []
        seg_conf = res["segmentations"][0]["confidence"] if res["segmentations"] else 0.0
        mask_area = res["segmentations"][0]["mask_area"] if res["segmentations"] else 0
        comp_mask = res["composite_mask"]
        comp_mask_px = int((comp_mask == 255).sum())

        has_pose = res["pose"]["has_pose"]
        landmark_count = len(res["pose"]["landmarks"])

        # Ground truth comparison if mask file exists
        gt_mask_path = masks_dir / fname
        iou = None
        dice = None
        if gt_mask_path.is_file():
            gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
            gt_bin = (gt_mask > 127).astype(np.uint8)
            pred_bin = (comp_mask == 255).astype(np.uint8)

            intersection = np.logical_and(gt_bin, pred_bin).sum()
            union = np.logical_or(gt_bin, pred_bin).sum()
            if union == 0:
                iou = 1.0 if (gt_bin.sum() == 0 and pred_bin.sum() == 0) else 0.0
            else:
                iou = float(intersection / union)

            denom = gt_bin.sum() + pred_bin.sum()
            if denom == 0:
                dice = 1.0
            else:
                dice = float(2.0 * intersection / denom)

        # Generate visualization artifacts
        # 1. Binary mask card
        mask_card = pipeline.render_binary_mask(res, show_card=True)
        card_out_name = f"{cid}_{Path(fname).stem}_card.png"
        cv2.imwrite(str(out_dir / card_out_name), mask_card)

        # 2. Side-by-side or combined overlay
        comb_vis = pipeline.render_visualization(
            im, res, show_boxes=True, show_masks=True, show_polygons=True, show_pose=True, fps=30.0
        )
        comb_out_name = f"{cid}_{Path(fname).stem}_overlay.png"
        cv2.imwrite(str(out_dir / comb_out_name), comb_vis)

        # Save sample data
        sample_record = {
            "filename": fname,
            "dimensions": f"{w}x{h}",
            "has_person": has_person,
            "person_count": p_count,
            "detection_confidence": round(float(det_conf), 4),
            "bounding_box": [round(x, 1) for x in det_bbox],
            "segmentation_confidence": round(float(seg_conf), 4),
            "mask_pixel_area": mask_area,
            "composite_mask_pixels": comp_mask_px,
            "has_pose": has_pose,
            "pose_landmarks": landmark_count,
            "iou_vs_ground_truth": round(iou, 4) if iou is not None else "N/A",
            "dice_vs_ground_truth": round(dice, 4) if dice is not None else "N/A",
            "card_image": card_out_name,
            "overlay_image": comb_out_name
        }
        evidence_results[cid]["samples"].append(sample_record)
        print(f"Evaluated [{cond['condition_name']}] file={fname}: person={has_person}, det_conf={det_conf:.3f}, seg_conf={seg_conf:.3f}, iou={iou}")

# Save JSON results
with open(out_dir / "condition_evidence_report.json", "w", encoding="utf-8") as f:
    json.dump(evidence_results, f, indent=2)

print("\nSaved condition evidence report to:", out_dir / "condition_evidence_report.json")
