from pathlib import Path
from ultralytics import YOLO
import csv

ROOT = Path(r"D:\realtime-human-detection")
MODEL_PATH = ROOT / "models" / "yolo11n.pt"
DATASET = ROOT / "dataset" / "human detection dataset"
OUTPUT = ROOT / "runs" / "full_dataset_check"

model = YOLO(str(MODEL_PATH))

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
report = []
total = 0

for folder_name, expected_people in [("0", False), ("1", True)]:
    folder = DATASET / folder_name

    images = sorted(
        [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ],
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem
    )

    print(f"\nTesting folder {folder_name}: {len(images)} images")

    correct_image_predictions = 0
    total_images = len(images)

    for start in range(0, total_images, 16):
        batch = images[start:start + 16]

        results = model.predict(
            source=[str(p) for p in batch],
            classes=[0],
            conf=0.25,
            imgsz=640,
            device="cpu",
            verbose=False
        )

        for image_path, result in zip(batch, results):
            detections = len(result.boxes) if result.boxes is not None else 0
            predicted_people = detections > 0

            correct = predicted_people == expected_people
            correct_image_predictions += int(correct)
            total += 1

            report.append({
                "folder": folder_name,
                "image": image_path.name,
                "detections": detections,
                "expected_people": expected_people,
                "image_level_match": correct
            })

        print(
            f"\rProcessed {min(start + len(batch), total_images)}"
            f"/{total_images}",
            end=""
        )

    print(
        f"\nFolder {folder_name} image-level agreement: "
        f"{correct_image_predictions}/{total_images}"
    )

OUTPUT.mkdir(parents=True, exist_ok=True)

report_path = OUTPUT / "preliminary_report.csv"

with report_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=report[0].keys())
    writer.writeheader()
    writer.writerows(report)

overall_matches = sum(row["image_level_match"] for row in report)

print("\n========== SUMMARY ==========")
print(f"Images processed: {total}")
print(f"Image-level matches: {overall_matches}/{total}")
print(f"Image-level agreement: {overall_matches / total * 100:.2f}%")
print(f"CSV report: {report_path}")
print("\nNote: This is image-level agreement, not object-detection mAP.")