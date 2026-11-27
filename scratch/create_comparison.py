import cv2
import numpy as np
from pathlib import Path

user_path = Path(r"C:\Users\dayan\.gemini\antigravity\brain\562e4ed7-27b0-47f2-bfee-f3103c29381b\.user_uploaded\media_1795509684702_76241b46.png")
card_path = Path(r"C:\Users\dayan\.gemini\antigravity\brain\562e4ed7-27b0-47f2-bfee-f3103c29381b\fitrange_yolo11s_binary_mask_card.png")
artifact_dir = Path(r"C:\Users\dayan\.gemini\antigravity\brain\562e4ed7-27b0-47f2-bfee-f3103c29381b")

img_user = cv2.imread(str(user_path))
img_card = cv2.imread(str(card_path))

target_h = 420
w_user = int(img_user.shape[1] * (target_h / img_user.shape[0]))
r_user = cv2.resize(img_user, (w_user, target_h), interpolation=cv2.INTER_AREA)

w_card = int(img_card.shape[1] * (target_h / img_card.shape[0]))
r_card = cv2.resize(img_card, (w_card, target_h), interpolation=cv2.INTER_AREA)

# Border divider
divider = np.zeros((target_h, 8, 3), dtype=np.uint8) + 60

comparison = np.hstack([r_user, divider, r_card])
cv2.imwrite(str(artifact_dir / "fitrange_mask_comparison.png"), comparison)
print("Saved comparison image successfully.")
