import cv2
import json
import shutil
from pathlib import Path
from ai.detection.detect import HumanDetector
from ai.detection.inference_onnx import HumanDetectorONNX

project_root = Path(__file__).resolve().parent.parent
artifact_dir = Path(r"C:\Users\dayan\.gemini\antigravity\brain\562e4ed7-27b0-47f2-bfee-f3103c29381b")
out_dir = project_root / "runs" / "model1_outputs"
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Initialize PyTorch Detector & ONNX Runtime Detector
print("Loading Model 1 (Human Detection) Engines...")
detector_pt = HumanDetector(conf_threshold=0.40)
detector_onnx = HumanDetectorONNX(conf_threshold=0.40)

test_cases = [
    {
        "name": "Front-Facing Person",
        "path": project_root / "dataset" / "val_annotated" / "images" / "front_facing_1.png",
        "prefix": "front_facing"
    },
    {
        "name": "Side-Facing Person",
        "path": project_root / "dataset" / "val_annotated" / "images" / "side_facing_1.png",
        "prefix": "side_facing"
    }
]

results_summary = {}

for tc in test_cases:
    img_path = tc["path"]
    img_name = tc["name"]
    prefix = tc["prefix"]
    print(f"\n==========================================")
    print(f"Running Model 1 Detection on: {img_name}")
    print(f"Source: {img_path}")
    print(f"==========================================")

    im_bgr = cv2.imread(str(img_path))
    h, w = im_bgr.shape[:2]

    # Run PyTorch inference
    res_pt = detector_pt.detect(im_bgr)
    print(f"--- PyTorch Detection Output ---")
    print(json.dumps(res_pt, indent=2))

    # Run ONNX Runtime inference
    res_onnx = detector_onnx.detect(im_bgr)
    print(f"--- ONNX Runtime Detection Output ---")
    print(json.dumps(res_onnx, indent=2))

    # Draw visual annotations for ONNX result
    vis_img = im_bgr.copy()
    for det in res_onnx.get("detections", []):
        x1, y1, x2, y2 = map(int, det["bbox"])
        conf = det["confidence"]
        label = f"person {conf:.1%}"

        # Draw bounding box
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label banner
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis_img, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), (0, 200, 0), -1)
        cv2.putText(vis_img, label, (x1 + 4, max(y1 - 4, th + 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Save visualization
    vis_file = out_dir / f"model1_detection_{prefix}.png"
    cv2.imwrite(str(vis_file), vis_img)
    shutil.copy(vis_file, artifact_dir / f"model1_detection_{prefix}.png")
    print(f"Saved visual detection result to: {vis_file}")

    results_summary[prefix] = {
        "image": img_name,
        "resolution": f"{w}x{h}",
        "pytorch": res_pt["detections"],
        "onnx": res_onnx["detections"],
        "vis_image": f"model1_detection_{prefix}.png"
    }

# Save JSON results
summary_json = out_dir / "model1_run_summary.json"
with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(results_summary, f, indent=2)

print("\nModel 1 run complete. Summary saved to:", summary_json)
