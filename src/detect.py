import cv2
import time
from pathlib import Path
from ultralytics import YOLO
import mediapipe as mp


def main():
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "models" / "yolo11n.pt"

    print("Loading YOLO person detector...")
    model = YOLO(str(model_path))

    # Initialize MediaPipe face detector
    mp_face = mp.solutions.face_detection
    face_detector = mp_face.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Cannot access webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Human and face detection started.")
    print("Press Q in the webcam window to quit.")

    previous_time = time.perf_counter()

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print("Failed to capture webcam frame.")
                break

            # Detect and track multiple people
            results = model.track(
    		source=frame,
    		persist=True,
    		classes=[0],
    		conf=0.35,
    		imgsz=640,
    		device="cpu",
    		tracker="bytetrack.yaml",
    		max_det=100,
    		verbose=False
	)

            person_count = 0

            for result in results:
                if result.boxes is None:
                    continue

                boxes = result.boxes

                for index, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )

                    confidence = float(box.conf[0])
                    person_count += 1

                    if boxes.id is not None:
                        person_id = int(boxes.id[index].item())
                        label = f"Person ID: {person_id}"
                    else:
                        label = f"Person {person_count}"

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"{label} {confidence:.2f}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2
                    )

            # Detect visible faces
            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            face_results = face_detector.process(rgb_frame)
            face_count = 0

            if face_results.detections:
                height, width = frame.shape[:2]

                for detection in face_results.detections:
                    bbox = detection.location_data.relative_bounding_box

                    x1 = max(0, int(bbox.xmin * width))
                    y1 = max(0, int(bbox.ymin * height))
                    x2 = min(width, int((bbox.xmin + bbox.width) * width))
                    y2 = min(height, int((bbox.ymin + bbox.height) * height))

                    face_count += 1

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 180, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Face",
                        (x1, max(y1 - 5, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 180, 0),
                        2
                    )

            # Calculate FPS
            current_time = time.perf_counter()
            fps = 1.0 / max(
                current_time - previous_time, 1e-6
            )
            previous_time = current_time

            # Display statistics
            cv2.putText(
                frame,
                f"People: {person_count} | Faces: {face_count} | FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "Human and Face Detection",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        face_detector.close()
        cv2.destroyAllWindows()
        print("Detection stopped.")


if __name__ == "__main__":
    main()