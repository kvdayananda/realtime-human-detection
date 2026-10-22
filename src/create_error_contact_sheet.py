from pathlib import Path
import csv
import math
import cv2
from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = Path(r"D:\realtime-human-detection")
DATASET = ROOT / "dataset" / "human detection dataset"
REPORT = ROOT / "runs" / "full_dataset_check" / "preliminary_report.csv"
OUTPUT = ROOT / "runs" / "error_contact_sheet"

THUMB_W = 240
THUMB_H = 180
COLS = 3
CARD_H = 230

OUTPUT.mkdir(parents=True, exist_ok=True)

if not REPORT.exists():
    raise FileNotFoundError(f"Report not found: {REPORT}")

with REPORT.open("r", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

errors = [
    row for row in rows
    if row["image_level_match"].strip().lower() == "false"
]

if not errors:
    print("No mismatched images found.")
    raise SystemExit(0)

model = YOLO(str(ROOT / "models" / "yolo11n.pt"))

for folder_name in ["0", "1"]:
    selected = [r for r in errors if r["folder"] == folder_name]
    cards = []

    for start in range(0, len(selected), 12):
        batch = selected[start:start + 12]
        paths = [DATASET / folder_name / r["image"] for r in batch]

        valid = [(r, p) for r, p in zip(batch, paths) if p.exists()]
        if not valid:
            continue

        results = model.predict(
            source=[str(p) for _, p in valid],
            classes=[0],
            conf=0.25,
            imgsz=640,
            device="cpu",
            verbose=False
        )

        for (row, path), result in zip(valid, results):
            original = Image.open(path).convert("RGB")
            original.thumbnail((THUMB_W, THUMB_H))

            annotated_array = cv2.cvtColor(
                result.plot(), cv2.COLOR_BGR2RGB
            )
            annotated = Image.fromarray(annotated_array)
            annotated.thumbnail((THUMB_W, THUMB_H))

            comparison = Image.new(
                "RGB", (THUMB_W * 2, THUMB_H), "white"
            )
            comparison.paste(original, (0, 0))
            comparison.paste(annotated, (THUMB_W, 0))

            count = len(result.boxes) if result.boxes is not None else 0
            cards.append((row["image"], count, comparison))

    if not cards:
        print(f"No images available for folder {folder_name}.")
        continue

    sheet = Image.new(
        "RGB",
        (COLS * THUMB_W * 2, math.ceil(len(cards) / COLS) * CARD_H),
        "#eeeeee"
    )
    draw = ImageDraw.Draw(sheet)

    for i, (name, count, comparison) in enumerate(cards):
        x = (i % COLS) * THUMB_W * 2
        y = (i // COLS) * CARD_H

        draw.text(
            (x + 5, y + 5),
            f"{name} | Person detections: {count}",
            fill="black"
        )
        draw.text(
            (x + 5, y + 23),
            "Original (left) | YOLO result (right)",
            fill="black"
        )
        sheet.paste(comparison, (x, y + 45))

    output_path = OUTPUT / f"class_{folder_name}_contact_sheet.jpg"
    sheet.save(output_path, quality=90)
    print(f"Created: {output_path}")

print("\nFinished creating contact sheets.")
