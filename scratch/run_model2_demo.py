import cv2
import json
import shutil
from pathlib import Path
import numpy as np

from ai.segmentation.segment import HumanSegmenter
from ai.segmentation.inference_onnx import HumanSegmenterONNX

project_root = Path(__file__).resolve().parent.parent
artifact_dir = Path(r"C:\Users\dayan\.gemini\antigravity\brain\562e4ed7-27b0-47f2-bfee-f3103c29381b")
out_dir = project_root / "runs" / "model2_outputs"
out_dir.mkdir(parents=True, exist_ok=True)

print("Loading Model 2 (Human Segmentation) Engines...")
segmenter_pt = HumanSegmenter(conf_threshold=0.40)
segmenter_onnx = HumanSegmenterONNX(conf_threshold=0.40)

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

def render_binary_mask_card(comp_mask, title="Human Segmentation", subtitle="(Binary Mask)"):
    h, w = comp_mask.shape[:2]
    mask_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if (comp_mask == 255).any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        smoothed = cv2.morphologyEx(comp_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(smoothed, contours, -1, color=255, thickness=-1)
        mask_canvas[smoothed == 255] = (255, 255, 255)

    header_h = max(int(h * 0.14), 54)
    card = np.zeros((h + header_h, w, 3), dtype=np.uint8)
    card[:header_h, :] = (48, 30, 20)
    cv2.line(card, (0, header_h), (w, header_h), (70, 50, 40), 1)
    card[header_h:, :] = mask_canvas

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale_title = max(w / 640.0 * 0.7, 0.45)
    scale_sub = max(w / 640.0 * 0.55, 0.38)
    (tw1, th1), _ = cv2.getTextSize(title, font, scale_title, 2)
    (tw2, th2), _ = cv2.getTextSize(subtitle, font, scale_sub, 1)

    y_title = max(int(header_h * 0.42), th1 + 6)
    y_sub = max(int(header_h * 0.82), y_title + th2 + 10)

    cv2.putText(card, title, ((w - tw1) // 2, y_title), font, scale_title, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(card, subtitle, ((w - tw2) // 2, y_sub), font, scale_sub, (230, 230, 230), 1, cv2.LINE_AA)
    return card

results_summary = {}

for tc in test_cases:
    img_path = tc["path"]
    img_name = tc["name"]
    prefix = tc["prefix"]
    print(f"\n==========================================")
    print(f"Running Model 2 Segmentation on: {img_name}")
    print(f"Source: {img_path}")
    print(f"==========================================")

    im_bgr = cv2.imread(str(img_path))
    h, w = im_bgr.shape[:2]

    # Run PyTorch inference
    res_pt = segmenter_pt.segment(im_bgr)
    mask_pt = segmenter_pt.get_composite_mask(im_bgr, res_pt["segmentations"])
    print(f"--- PyTorch Segmentation Count: {len(res_pt['segmentations'])} ---")
    if res_pt["segmentations"]:
        first = res_pt["segmentations"][0]
        print(f"Confidence: {first['confidence']:.4f}, Mask Area: {first['mask_area']} px, Polygon Points: {len(first['polygon'])}")

    # Run ONNX Runtime inference
    res_onnx = segmenter_onnx.segment(im_bgr)
    mask_onnx = segmenter_onnx.get_composite_mask(im_bgr, res_onnx["segmentations"])
    print(f"--- ONNX Runtime Segmentation Count: {len(res_onnx['segmentations'])} ---")
    if res_onnx["segmentations"]:
        first = res_onnx["segmentations"][0]
        print(f"Confidence: {first['confidence']:.4f}, Mask Area: {first['mask_area']} px, Polygon Points: {len(first['polygon'])}")

    # Save raw binary mask and styled card
    card = render_binary_mask_card(mask_onnx)
    card_path = out_dir / f"model2_segmentation_{prefix}_card.png"
    mask_path = out_dir / f"model2_segmentation_{prefix}_binary_mask.png"
    cv2.imwrite(str(card_path), card)
    cv2.imwrite(str(mask_path), mask_onnx)

    shutil.copy(card_path, artifact_dir / f"model2_segmentation_{prefix}_card.png")
    shutil.copy(mask_path, artifact_dir / f"model2_segmentation_{prefix}_binary_mask.png")
    print(f"Saved binary mask card to: {card_path}")

    # Overlay visualization
    overlay = im_bgr.copy()
    color_mask = np.zeros_like(im_bgr)
    color_mask[mask_onnx == 255] = (0, 255, 128) # green-cyan fill
    overlay = cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0)
    for seg in res_onnx["segmentations"]:
        poly = np.array(seg["polygon"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [poly], isClosed=True, color=(0, 255, 255), thickness=2)
    overlay_path = out_dir / f"model2_segmentation_{prefix}_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)
    shutil.copy(overlay_path, artifact_dir / f"model2_segmentation_{prefix}_overlay.png")

    results_summary[prefix] = {
        "image": img_name,
        "resolution": f"{w}x{h}",
        "pytorch_confidence": res_pt["segmentations"][0]["confidence"] if res_pt["segmentations"] else 0.0,
        "pytorch_mask_area": res_pt["segmentations"][0]["mask_area"] if res_pt["segmentations"] else 0,
        "onnx_confidence": res_onnx["segmentations"][0]["confidence"] if res_onnx["segmentations"] else 0.0,
        "onnx_mask_area": res_onnx["segmentations"][0]["mask_area"] if res_onnx["segmentations"] else 0,
        "card_image": f"model2_segmentation_{prefix}_card.png",
        "binary_mask_image": f"model2_segmentation_{prefix}_binary_mask.png",
        "overlay_image": f"model2_segmentation_{prefix}_overlay.png",
    }

# Save summary JSON
with open(out_dir / "model2_run_summary.json", "w", encoding="utf-8") as f:
    json.dump(results_summary, f, indent=2)

print("\nModel 2 run complete. Summary saved to:", out_dir / "model2_run_summary.json")
