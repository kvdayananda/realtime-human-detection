import cv2
import json
import shutil
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parent.parent
from ai.pipeline.mirror_pipeline import FitRangeMirrorPipeline
from ai.detection import HumanDetectorONNX

out_dir = project_root / "runs" / "sample_predictions"
out_dir.mkdir(parents=True, exist_ok=True)

pipeline = FitRangeMirrorPipeline(
    det_conf=0.35,
    seg_conf=0.35,
    pose_conf=0.40,
    enable_detection=True,
    enable_segmentation=True,
    enable_pose=True,
)
det_onnx = HumanDetectorONNX(conf_threshold=0.35)

scenarios = [
    {
        "id": "front_facing_standing",
        "src": "dataset/val_annotated/images/front_facing_1.png",
        "title": "Front-Facing Person (Standing, Clear Lighting)",
    },
    {
        "id": "side_facing_profile",
        "src": "dataset/val_annotated/images/side_facing_1.png",
        "title": "Side-Facing Person (Lateral Profile)",
    },
    {
        "id": "athletic_clothing_workout",
        "src": "dataset/val_annotated/images/athletic_clothing.png",
        "title": "Athletic Clothing & Active Workout Movement",
    },
    {
        "id": "cluttered_indoor_background",
        "src": "dataset/val_annotated/images/cluttered_background.png",
        "title": "Cluttered Indoor Background Environment",
    },
    {
        "id": "negative_room_control",
        "src": "dataset/val_annotated/images/negative_room.png",
        "title": "Negative Control Scene (No Human Present)",
    },
]

manifest = []

for sc in scenarios:
    sc_id = sc["id"]
    src_path = project_root / sc["src"]
    im = cv2.imread(str(src_path))
    h, w = im.shape[:2]

    # 1. Save original image
    orig_path = out_dir / f"{sc_id}_original.png"
    cv2.imwrite(str(orig_path), im)

    # 2. Run detection-only and draw bounding boxes
    det_res = det_onnx.detect(im)
    det_vis = im.copy()
    for d in det_res["detections"]:
        x1, y1, x2, y2 = map(int, d["bbox"])
        conf = d["confidence"]
        cv2.rectangle(det_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"person {conf:.2f}"
        cv2.putText(
            det_vis, label, (x1, max(y1 - 6, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
    det_path = out_dir / f"{sc_id}_detection.png"
    cv2.imwrite(str(det_path), det_vis)

    # 3. Run full pipeline
    pipe_res = pipeline.process_frame(im)

    # 4. Save binary segmentation mask
    mask = pipe_res["composite_mask"]
    mask_path = out_dir / f"{sc_id}_mask.png"
    cv2.imwrite(str(mask_path), mask)

    # 5. Save combined overlay visualization
    combined_vis = pipeline.render_visualization(
        im, pipe_res, show_boxes=True, show_masks=True, show_polygons=True, show_pose=True, fps=30.0
    )
    comb_path = out_dir / f"{sc_id}_combined.png"
    cv2.imwrite(str(comb_path), combined_vis)

    record = {
        "scenario_id": sc_id,
        "title": sc["title"],
        "source": sc["src"],
        "resolution": [w, h],
        "has_person": pipe_res["has_person"],
        "person_count": pipe_res["person_count"],
        "detection_count": len(pipe_res["detections"]),
        "segmentation_count": len(pipe_res["segmentations"]),
        "has_pose": pipe_res["pose"]["has_pose"],
        "mask_person_pixels": int((mask == 255).sum()),
        "files": {
            "original": f"{sc_id}_original.png",
            "detection": f"{sc_id}_detection.png",
            "mask": f"{sc_id}_mask.png",
            "combined": f"{sc_id}_combined.png",
        },
    }
    manifest.append(record)
    print(f"Generated sample predictions for {sc_id} (people={pipe_res['person_count']}, pose={pipe_res['pose']['has_pose']})")

manifest_path = out_dir / "predictions_manifest.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"Sample prediction outputs and manifest saved successfully to: {out_dir}")
