import csv
from pathlib import Path

report_path = (
    Path(r"D:\realtime-human-detection")
    / "runs"
    / "full_dataset_check"
    / "preliminary_report.csv"
)

with report_path.open("r", newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

errors = [row for row in rows if row["image_level_match"].lower() == "false"]

print(f"Total mismatched images: {len(errors)}")
print("\nFolder 0: Unexpected person detections")
print("-" * 50)

for row in errors:
    if row["folder"] == "0":
        print(f'{row["image"]} | Detections: {row["detections"]}')

print("\nFolder 1: People not detected")
print("-" * 50)

for row in errors:
    if row["folder"] == "1":
        print(f'{row["image"]} | Detections: {row["detections"]}')
