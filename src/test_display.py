import cv2
import numpy as np

image = np.zeros((480, 640, 3), dtype=np.uint8)

cv2.putText(
    image,
    "OpenCV Display Test",
    (100, 240),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2,
)

cv2.imshow("Display Test", image)

print("Press Q to close the window.")

while True:
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()