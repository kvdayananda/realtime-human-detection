"""FitRange Mirror - Real-Time Webcam / Video Application.

Runs the integrated Human Detection, Human Segmentation, and Pose Estimation pipeline
on a live webcam feed, video stream, or static test input.

Usage:
    # Live webcam (default index 0)
    python -m ai.pipeline.run_mirror

    # Video file or image
    python -m ai.pipeline.run_mirror --source path/to/video.mp4
    python -m ai.pipeline.run_mirror --source "dataset/human detection dataset/1/0.png"

    # Selective component execution
    python -m ai.pipeline.run_mirror --no-seg
    python -m ai.pipeline.run_mirror --no-det
    python -m ai.pipeline.run_mirror --no-pose
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from ai.pipeline.mirror_pipeline import FitRangeMirrorPipeline

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("FitRangeMirror.App")


def run_pipeline(
    source: str = "0",
    det_conf: float = 0.35,
    seg_conf: float = 0.35,
    enable_detection: bool = True,
    enable_segmentation: bool = True,
    enable_pose: bool = True,
    pose_model_path: str | None = None,
    display: bool = True,
    save_output: str | None = None,
    max_frames: int | None = None,
) -> None:
    """Run the integrated FitRange Mirror pipeline on webcam or file source."""
    logger.info("Initializing FitRange Mirror Pipeline...")
    pipeline = FitRangeMirrorPipeline(
        det_conf=det_conf,
        seg_conf=seg_conf,
        pose_model_path=pose_model_path,
        enable_detection=enable_detection,
        enable_segmentation=enable_segmentation,
        enable_pose=enable_pose,
    )

    # Determine if source is webcam index, image, or video file
    is_image = False
    source_is_cam = False

    try:
        cam_idx = int(source)
        source_is_cam = True
    except ValueError:
        cam_idx = None

    if source_is_cam:
        logger.info("Opening webcam at index %d...", cam_idx)
        # Try DirectShow on Windows, then default
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(cam_idx)

        if not cap.isOpened():
            logger.error("Cannot access camera at index %d.", cam_idx)
            raise RuntimeError(
                f"Cannot access webcam at index {cam_idx}. Check camera connection and permissions."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        src_path = Path(source).resolve()
        if not src_path.exists():
            logger.error("Source file does not exist: %s", src_path)
            raise FileNotFoundError(f"Input source file not found: {src_path}")

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        if src_path.suffix.lower() in image_extensions:
            is_image = True
            logger.info("Processing static image source: %s", src_path)
            frame = cv2.imread(str(src_path))
            if frame is None:
                raise ValueError(f"Failed to read image from: {src_path}")
            cap = None
        else:
            logger.info("Opening video source: %s", src_path)
            cap = cv2.VideoCapture(str(src_path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video source: {src_path}")

    # Handle static image input
    if is_image:
        t0 = time.perf_counter()
        result = pipeline.process_frame(frame)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        fps = 1000.0 / max(latency_ms, 1e-3)

        logger.info(
            "Detection, Segmentation & Pose complete: %d person(s) found, pose=%s in %.1f ms",
            result["person_count"],
            result["pose"]["has_pose"],
            latency_ms,
        )

        vis = pipeline.render_visualization(frame, result, fps=fps)

        if save_output:
            cv2.imwrite(save_output, vis)
            logger.info("Saved annotated image to: %s", save_output)

        if display:
            window_name = "FitRange Mirror - Static Preview"
            cv2.imshow(window_name, vis)
            logger.info("Displaying image window. Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # Handle stream / video loop
    writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        writer = cv2.VideoWriter(save_output, fourcc, fps_in, (fw, fh))
        logger.info("Saving output video stream to: %s", save_output)

    window_name = "FitRange Mirror - Real-Time Pipeline"
    if display:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    prev_time = time.perf_counter()
    frame_count = 0

    logger.info("Starting real-time processing loop. Press 'q' in window to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream or capture failed.")
                break

            t_start = time.perf_counter()
            result = pipeline.process_frame(frame)
            t_infer = time.perf_counter() - t_start

            curr_time = time.perf_counter()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            vis = pipeline.render_visualization(frame, result, fps=fps)

            if writer is not None:
                writer.write(vis)

            if display:
                cv2.imshow(window_name, vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User requested exit ('q' pressed).")
                    break

            frame_count += 1
            if max_frames and frame_count >= max_frames:
                logger.info("Reached maximum frame limit (%d). Exiting.", max_frames)
                break

    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        logger.info("Pipeline processing finished. Processed %d frames.", frame_count)


def main() -> None:
    """CLI entry point for FitRange Mirror."""
    parser = argparse.ArgumentParser(
        description="FitRange Mirror - Real-Time Human Detection & Instance Segmentation"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source: camera index (default: '0'), or path to video/image file.",
    )
    parser.add_argument(
        "--det-conf",
        type=float,
        default=0.35,
        help="Confidence threshold for human detection (default: 0.35).",
    )
    parser.add_argument(
        "--seg-conf",
        type=float,
        default=0.35,
        help="Confidence threshold for human segmentation (default: 0.35).",
    )
    parser.add_argument(
        "--no-det",
        action="store_true",
        help="Disable detection model (run segmentation only).",
    )
    parser.add_argument(
        "--no-seg",
        action="store_true",
        help="Disable segmentation model (run detection only).",
    )
    parser.add_argument(
        "--no-pose",
        action="store_true",
        help="Disable pose estimation model (BlazePose).",
    )
    parser.add_argument(
        "--pose-model",
        type=str,
        default=None,
        help="Optional path to MediaPipe Pose Landmarker .task model asset.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without displaying OpenCV GUI window.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save annotated output image or video.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process.",
    )
    args = parser.parse_args()

    run_pipeline(
        source=args.source,
        det_conf=args.det_conf,
        seg_conf=args.seg_conf,
        enable_detection=not args.no_det,
        enable_segmentation=not args.no_seg,
        enable_pose=not args.no_pose,
        pose_model_path=args.pose_model,
        display=not args.no_display,
        save_output=args.save,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
