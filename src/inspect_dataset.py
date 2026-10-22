from pathlib import Path
from PIL import Image

DATASET = Path("dataset/human detection dataset")

for class_name in ["0", "1"]:
    folder = DATASET / class_name

    print(f"\nClass {class_name}")
    print("-" * 40)

    images = list(folder.glob("*.png"))

    for image_path in images[:5]:
        try:
            with Image.open(image_path) as image:
                print(
                    f"File: {image_path.name} | "
                    f"Size: {image.size} | "
                    f"Mode: {image.mode}"
                )
        except Exception as error:
            print(f"Could not read {image_path.name}: {error}")