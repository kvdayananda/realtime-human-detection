"""FitRange Mirror - Completion Audit Scenario Evaluation Script.

Tests detection and segmentation across diverse scenarios:
- Front-facing vs side/profile poses
- Different clothing types and contrast
- Indoor, outdoor, and cluttered backgrounds
- Variable lighting (bright, dim, shadow)
- Different body postures
- Negative background controls
"""

import json
import os
import pathlib
import sys
import cv2
import numpy as np

# Set local ultralytics directory and pythonpath
project_root = pathlib.Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.environ["YOLO_CONFIG_DIR"] = str((project_root / "models" / ".ultralytics").resolve())

from ai.detection import HumanDetector, HumanDetectorONNX
from ai.segmentation import HumanSegmenter, HumanSegmenterONNX

det_pt = HumanDetector()
det_onnx = HumanDetectorONNX()
seg_pt = HumanSegmenter()
seg_onnx = HumanSegmenterONNX()

out_dir = project_root / "runs" / "audit_samples"
out_dir.mkdir(parents=True, exist_ok=True)

# Select diverse candidate images from the benchmark dataset
candidates = [
    ("dataset/human detection dataset/1/0.png", "Front-facing standing, indoor clear lighting"),
    ("dataset/human detection dataset/1/5.png", "Partial / side angle, indoor background"),
    ("dataset/human detection dataset/1/10.png", "Outdoor background, natural daylight"),
    ("dataset/human detection dataset/1/25.png", "Dynamic posture / athletic clothing"),
    ("dataset/human detection dataset/1/50.png", "Challenging contrast / indoor lighting"),
    ("dataset/human detection dataset/1/100.png", "Complex cluttered background, distinctive pose"),
    ("dataset/human detection dataset/0/0.png", "Negative background control (no human present)"),
]

audit_results = []

for rel_path, scenario in candidates:
    img_path = project_root / rel_path
    if not img_path.exists():
        continue

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    # 1. Detection inference
    res_det_pt = det_pt.detect(img)
    res_det_onnx = det_onnx.detect(img)

    # 2. Segmentation inference
    res_seg_pt = seg_pt.segment(img)
    res_seg_onnx = seg_onnx.segment(img)

    item = {
        "image": rel_path,
        "resolution": f"{w}x{h}",
        "scenario": scenario,
        "det_pt_count": len(res_det_pt["detections"]),
        "det_onnx_count": len(res_det_onnx["detections"]),
        "seg_pt_count": len(res_seg_pt["segmentations"]),
        "seg_onnx_count": len(res_seg_onnx["segmentations"]),
        "det_pt_conf": [round(d["confidence"], 3) for d in res_det_pt["detections"]],
        "det_onnx_conf": [round(d["confidence"], 3) for d in res_det_onnx["detections"]],
        "seg_mask_area": [s["mask_area"] for s in res_seg_pt["segmentations"]],
    }
    audit_results.append(item)

    # Generate visual overlay combining segmentation mask and detection bounding box
    vis = img.copy()

    # Draw segmentation masks & polygons (blue/cyan fill)
    for s in res_seg_pt["segmentations"]:
        poly = s.get("polygon", [])
        if len(poly) >= 3:
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            overlay = vis.copy()
            cv2.fillPoly(overlay, [pts], color=(255, 128, 0))
            cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)
            cv2.polylines(vis, [pts], isClosed=True, color=(255, 200, 0), thickness=2)

    # Draw detection bounding boxes (green)
    for d in res_det_pt["detections"]:
        x1, y1, x2, y2 = map(int, d["bbox"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        conf_val = d["confidence"]
        label = f"person {conf_val:.2f}"
        cv2.putText(vis, label, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    prefix = "neg_" if "0/0.png" in rel_path else "pos_"
    save_name = out_dir / f"{prefix}{img_path.stem}_annotated.jpg"
    cv2.imwrite(str(save_name), vis)

summary_json = out_dir / "audit_scenario_results.json"
with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(audit_results, f, indent=2)

print("AUDIT_RESULTS_START")
print(json.dumps(audit_results, indent=2))
print("AUDIT_RESULTS_END")
print(f"Visual sample overlays saved to: {out_dir}")
