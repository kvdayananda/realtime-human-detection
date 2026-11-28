from pathlib import Path
import cv2
from ultralytics import YOLO


class PersonDetector:
    def __init__(self, confidence=0.25, image_size=640):
        model_path = Path(__file__).resolve().parent / "person_detection.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.image_size = image_size

    def detect(self, frame):
        results = self.model.predict(
            source=frame,
            classes=[0],
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False
        )

        result = results[0]
        annotated_frame = result.plot()
        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].cpu().tolist()
                )

                detections.append({
                    "class": "person",
                    "confidence": float(box.conf[0].cpu()),
                    "bbox": [x1, y1, x2, y2]
                })

        return annotated_frame, detections


def main():
    detector = PersonDetector()
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "Cannot open webcam. Check camera connection and permissions."
        )

    print("Model 1: Person Detection started. Press Q to exit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Failed to read webcam frame.")
                break

            annotated_frame, detections = detector.detect(frame)

            cv2.putText(
                annotated_frame,
                f"People detected: {len(detections)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow("Model 1 - Person Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
