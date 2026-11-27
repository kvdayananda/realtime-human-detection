import cv2
import numpy as np
from pathlib import Path
import shutil

project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / "runs" / "condition_evidence"
artifact_dir = Path(r"C:\Users\dayan\.gemini\antigravity\brain\562e4ed7-27b0-47f2-bfee-f3103c29381b")
art_cond_dir = artifact_dir / "condition_evidence"
art_cond_dir.mkdir(parents=True, exist_ok=True)

# Copy all files to artifact directory
for f in src_dir.glob("*.png"):
    shutil.copy(f, art_cond_dir / f.name)
print(f"Copied all {len(list(src_dir.glob('*.png')))} evidence images to artifact folder.")

# Build 6-panel condition evidence summary grid
grid_items = [
    ("1. Front-Facing Person (IoU: 97.9%)", src_dir / "front_facing_front_facing_1_overlay.png"),
    ("2. Side-Facing Person (IoU: 94.2%)", src_dir / "side_facing_side_facing_1_overlay.png"),
    ("3. Different Clothing (Sportswear)", src_dir / "different_clothing_athletic_clothing_overlay.png"),
    ("4. Different Body Shapes (Casual)", src_dir / "different_body_shapes_casual_clothing_overlay.png"),
    ("5. Cluttered Indoor Background", src_dir / "different_backgrounds_cluttered_background_overlay.png"),
    ("6. Different Lighting (Outdoor Daylight)", src_dir / "different_lighting_outdoor_daylight_overlay.png"),
]

cell_w, cell_h = 360, 360
banner_h = 40
processed_cells = []

for title, path in grid_items:
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    # Resize keeping aspect ratio, centered in cell
    scale = min((cell_h - banner_h) / h, cell_w / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8) + 20
    # Add title banner
    cv2.rectangle(cell, (0, 0), (cell_w, banner_h), (45, 30, 20), -1)
    cv2.putText(cell, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)
    # Place image centered below banner
    y_off = banner_h + (cell_h - banner_h - nh) // 2
    x_off = (cell_w - nw) // 2
    cell[y_off:y_off+nh, x_off:x_off+nw] = resized
    processed_cells.append(cell)

row1 = np.hstack(processed_cells[:3])
row2 = np.hstack(processed_cells[3:])
grid = np.vstack([row1, row2])

# Add top main title
main_banner_h = 60
main_grid = np.zeros((grid.shape[0] + main_banner_h, grid.shape[1], 3), dtype=np.uint8)
main_grid[:main_banner_h, :] = (35, 20, 15)
main_grid[main_banner_h:, :] = grid

cv2.putText(main_grid, "FITRANGE MIRROR - EMPIRICAL CONDITION VALIDATION MATRIX", (cell_w * 3 // 2 - 380, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(main_grid, "Empirical visual proofs across front/side-facing, clothing, body shapes, backgrounds, and lighting",
            (cell_w * 3 // 2 - 340, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1, cv2.LINE_AA)

matrix_path = artifact_dir / "condition_evidence_matrix.png"
cv2.imwrite(str(matrix_path), main_grid)
print("Saved condition evidence matrix to:", matrix_path)
