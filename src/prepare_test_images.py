from pathlib import Path
import random
import shutil

ROOT = Path(r"D:\realtime-human-detection")
SOURCE = ROOT / "dataset" / "human detection dataset"
DEST = ROOT / "dataset" / "person_detection_test" / "images"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SAMPLES_PER_CLASS = 40

random.seed(42)
DEST.mkdir(parents=True, exist_ok=True)

# Remove only previously copied test images.
for path in DEST.iterdir():
    if path.is_file() and path.suffix.lower() in EXTENSIONS:
        path.unlink()

for class_name in ["0", "1"]:
    source_dir = SOURCE / class_name

    images = [
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]

    if len(images) < SAMPLES_PER_CLASS:
        raise RuntimeError(f"Not enough images in {source_dir}")

    selected = random.sample(images, SAMPLES_PER_CLASS)

    for image in selected:
        destination_name = f"class{class_name}_{image.name}"
        shutil.copy2(image, DEST / destination_name)

    print(f"Copied {SAMPLES_PER_CLASS} images from class {class_name}")

print(f"\nTest images created: {len(list(DEST.iterdir()))}")
print(f"Location: {DEST}")
print("Original dataset files were not moved or deleted.")
