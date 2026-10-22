from pathlib import Path
import csv
import shutil

from ultralytics import YOLO

ROOT = Path(r"D:\realtime-human-detection")

MODEL_PATH = ROOT / "models" / "yolo11n.pt"
DATASET = ROOT / "dataset" / "human detection dataset"
REPORT = ROOT / "runs" / "full_dataset_check" / "preliminary_report.csv"

OUTPUT = ROOT / "runs" / "error_review"

if not REPORT.exists():
    raise FileNotFoundError(f"Report not found: {REPORT}")

model = YOLO(str(MODEL_PATH))

with REPORT.open("r", newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

errors = [
    row for row in rows
    if row["image_level_match"].strip().lower() == "false"
]

print(f"Reviewing {len(errors)} mismatched images...")

for folder_name in ["0", "1"]:
    selected = [
        row for row in errors
        if row["folder"] == folder_name
    ]

    if not selected:
        continue

    source_images = [
        DATASET / folder_name / row["image"]
        for row in selected
    ]

    result_dir = OUTPUT / f"class_{folder_name}"

    results = model.predict(
        source=[str(path) for path in source_images],
        classes=[0],
        conf=0.25,
        imgsz=640,
        device="cpu",
        save=True,
        project=str(OUTPUT),
        name=f"class_{folder_name}",
        exist_ok=True,
        verbose=False
    )

    for source_path, result in zip(source_images, results):
        if not source_path.exists():
            print(f"Missing image: {source_path}")
            continue

        print(
            f"Folder {folder_name}: {source_path.name} - "
            f"{len(result.boxes) if result.boxes is not None else 0} detections"
        )

print("\nError review completed.")
print(f"Folder 0 results: {OUTPUT / 'class_0'}")
print(f"Folder 1 results: {OUTPUT / 'class_1'}")