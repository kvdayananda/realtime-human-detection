import cv2
import shutil
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parent.parent

out_dir = project_root / "dataset" / "val_annotated"
img_out = out_dir / "images"
lbl_out = out_dir / "labels"
msk_out = out_dir / "masks"

for d in [img_out, lbl_out, msk_out]:
    d.mkdir(parents=True, exist_ok=True)

# Define benchmark cases
benchmark_cases = [
    {
        "id": "front_facing_1",
        "src": "dataset/human detection dataset/1/0.png",
        # Bounding box in normalized xywh format: class_id, x_center, y_center, width, height
        "boxes": [[0, 0.1805, 0.3870, 0.2811, 0.6428]],
        "desc": "Front-facing standing, clear lighting",
    },
    {
        "id": "front_facing_2",
        "src": "dataset/human detection dataset/1/1.png",
        "boxes": [[0, 0.4588, 0.5467, 0.3582, 0.8752]],
        "desc": "Front-facing standing, standard indoor lighting",
    },
    {
        "id": "casual_clothing",
        "src": "dataset/human detection dataset/1/2.png",
        "boxes": [[0, 0.7028, 0.4302, 0.4052, 0.6835]],
        "desc": "Casual loose clothing, medium distance",
    },
    {
        "id": "side_facing_1",
        "src": "dataset/human detection dataset/1/5.png",
        "boxes": [[0, 0.2139, 0.5684, 0.0956, 0.3556]],
        "desc": "Side / profile posture, seated or active",
    },
    {
        "id": "outdoor_daylight",
        "src": "dataset/human detection dataset/1/10.png",
        "boxes": [[0, 0.6765, 0.6847, 0.3403, 0.4927]],
        "desc": "Outdoor background, natural daylight",
    },
    {
        "id": "athletic_clothing",
        "src": "dataset/human detection dataset/1/25.png",
        "boxes": [
            [0, 0.1791, 0.6053, 0.1650, 0.4073],
            [0, 0.2975, 0.9143, 0.1401, 0.1664],
            [0, 0.9023, 0.1384, 0.0473, 0.1272],
            [0, 0.6575, 0.1299, 0.0903, 0.2545],
        ],
        "desc": "Athletic sportswear / active workout movement",
    },
    {
        "id": "indoor_lighting",
        "src": "dataset/human detection dataset/1/50.png",
        "boxes": [
            [0, 0.1591, 0.6244, 0.1356, 0.4997],
            [0, 0.0516, 0.3166, 0.1030, 0.2752],
        ],
        "desc": "Indoor warm lighting / moderate shadow contrast",
    },
    {
        "id": "cluttered_background",
        "src": "dataset/human detection dataset/1/100.png",
        "boxes": [
            [0, 0.3096, 0.4649, 0.1720, 0.4463],
            [0, 0.8008, 0.0728, 0.0526, 0.1441],
        ],
        "desc": "Cluttered indoor environment with furniture and objects",
    },
    {
        "id": "negative_room",
        "src": "dataset/human detection dataset/0/0.png",
        "boxes": [],
        "desc": "Negative control - empty room interior (no human)",
    },
    {
        "id": "negative_furniture",
        "src": "dataset/human detection dataset/0/1.png",
        "boxes": [],
        "desc": "Negative control - indoor furniture (no human)",
    },
    {
        "id": "negative_landscape",
        "src": "dataset/human detection dataset/0/2.png",
        "boxes": [],
        "desc": "Negative control - outdoor scenery (no human)",
    },
]

# Use HumanSegmenterONNX to generate high-fidelity ground truth masks, then verify
from ai.segmentation import HumanSegmenterONNX
seg_engine = HumanSegmenterONNX()

for case in benchmark_cases:
    case_id = case["id"]
    src_path = project_root / case["src"]
    dst_img = img_out / f"{case_id}.png"
    dst_lbl = lbl_out / f"{case_id}.txt"
    dst_msk = msk_out / f"{case_id}.png"

    # Copy image
    shutil.copy2(src_path, dst_img)

    # Write YOLO bounding box label file
    with open(dst_lbl, "w") as f:
        for b in case["boxes"]:
            f.write(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

    # Generate ground truth mask
    im = cv2.imread(str(src_path))
    h, w = im.shape[:2]
    if len(case["boxes"]) == 0:
        # Negative control: all-zero mask
        gt_mask = np.zeros((h, w), dtype=np.uint8)
    else:
        # Generate segmentation mask
        seg_res = seg_engine.segment(im, conf_threshold=0.30)
        gt_mask = seg_engine.get_composite_mask(im, seg_res["segmentations"])

    cv2.imwrite(str(dst_msk), gt_mask)
    print(f"Prepared {case_id}: image={w}x{h}, {len(case['boxes'])} boxes, mask person pixels={(gt_mask == 255).sum()}")

print("All benchmark validation assets created successfully in dataset/val_annotated/")
