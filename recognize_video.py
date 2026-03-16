#!/usr/bin/env python3
"""Real-time face recognition using a webcam.

Usage
-----
    python recognize_video.py
    python recognize_video.py --camera 0 --model models/face_encodings.pkl
    python recognize_video.py --scale 0.5   # process at half resolution for speed
"""

from __future__ import annotations

import argparse
import logging
import sys

import cv2
import numpy as np

from src.face_recognizer import FaceRecognizer
from src.utils import draw_bounding_box, frame_to_rgb, rgb_to_bgr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time face recognition from a webcam stream."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0).",
    )
    parser.add_argument(
        "--model",
        default="models/face_encodings.pkl",
        help="Path to the trained model file.",
    )
    parser.add_argument(
        "--detection-model",
        choices=["hog", "cnn"],
        default="hog",
        help="Face detection model (default: hog).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.6,
        help="Match tolerance (default: 0.6).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Resize factor before detection – smaller is faster (default: 0.5).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def run(
    camera: int = 0,
    model: str = "models/face_encodings.pkl",
    detection_model: str = "hog",
    tolerance: float = 0.6,
    scale: float = 0.5,
    verbose: bool = False,
) -> int:
    """Run real-time face recognition on a webcam feed.

    Parameters
    ----------
    camera:
        Camera device index passed to ``cv2.VideoCapture``.
    model:
        Path to the trained model (pickle) file.
    detection_model:
        ``"hog"`` or ``"cnn"``.
    tolerance:
        Euclidean-distance threshold for a positive match.
    scale:
        Downscale factor applied to each frame before face detection.
    verbose:
        Enable debug logging when ``True``.

    Returns
    -------
    Exit code: ``0`` on success, ``1`` on error.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Load model
    try:
        recognizer = FaceRecognizer(
            model_path=model,
            detection_model=detection_model,
            tolerance=tolerance,
        )
    except FileNotFoundError as exc:
        logging.error("%s\nRun train.py first to generate the model.", exc)
        return 1

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        logging.error("Cannot open camera %d.", camera)
        return 1

    print("Press 'q' to quit\u2026")

    process_this_frame = True
    cached_results: list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to grab frame – stopping.")
            break

        if process_this_frame:
            # Resize for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            rgb_small = frame_to_rgb(small_frame)

            raw_results = recognizer.recognize(rgb_small)

            # Scale bounding boxes back to original frame size
            inv = 1.0 / scale
            cached_results = []
            for name, conf, (top, right, bottom, left) in raw_results:
                scaled_box = (
                    int(top * inv),
                    int(right * inv),
                    int(bottom * inv),
                    int(left * inv),
                )
                cached_results.append((name, conf, scaled_box))

        process_this_frame = not process_this_frame  # alternate frames

        # Annotate and display
        rgb_frame = frame_to_rgb(frame)
        for name, conf, bbox in cached_results:
            draw_bounding_box(rgb_frame, bbox, name, conf)

        cv2.imshow("Face Recognition \u2013 press q to quit", rgb_to_bgr(rgb_frame))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    args = parse_args()
    return run(
        camera=args.camera,
        model=args.model,
        detection_model=args.detection_model,
        tolerance=args.tolerance,
        scale=args.scale,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
