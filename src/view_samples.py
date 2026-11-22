from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import matplotlib.pyplot as plt

DATASET = Path("dataset/human detection dataset")

fig, axes = plt.subplots(2, 5, figsize=(15, 7))

for row, class_name in enumerate(["0", "1"]):
    folder = DATASET / class_name
    images = sorted(folder.glob("*.png"))[:5]

    for col, image_path in enumerate(images):
        with Image.open(image_path) as image:
            axes[row, col].imshow(image)

        axes[row, col].set_title(
            f"Class {class_name}: {image_path.name}"
        )
        axes[row, col].axis("off")

plt.tight_layout()
plt.show()