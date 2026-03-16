#!/usr/bin/env python3
"""CLI entry point for the Face Recognition project.

Commands
--------
    python main.py train      – encode faces from the dataset
    python main.py image      – recognise faces in a static image
    python main.py video      – real-time recognition from a webcam
"""

from __future__ import annotations

import argparse
import logging
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="face-recognition",
        description="Full face recognition system powered by ML.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ── train ──────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Encode faces from a labelled dataset.")
    p_train.add_argument("--dataset", default="dataset", help="Dataset directory.")
    p_train.add_argument(
        "--model", default="models/face_encodings.pkl", help="Output model path."
    )
    p_train.add_argument(
        "--detection-model", choices=["hog", "cnn"], default="hog",
        help="Detection model: hog (fast) or cnn (accurate).",
    )
    p_train.add_argument("--tolerance", type=float, default=0.6)
    p_train.add_argument("-v", "--verbose", action="store_true")

    # ── image ──────────────────────────────────────────────────────────────
    p_image = sub.add_parser("image", help="Recognize faces in a static image.")
    p_image.add_argument("--image", required=True, help="Path to the input image.")
    p_image.add_argument("--output", default=None, help="Save annotated image here.")
    p_image.add_argument(
        "--model", default="models/face_encodings.pkl", help="Trained model path."
    )
    p_image.add_argument(
        "--detection-model", choices=["hog", "cnn"], default="hog"
    )
    p_image.add_argument("--tolerance", type=float, default=0.6)
    p_image.add_argument("-v", "--verbose", action="store_true")

    # ── video ──────────────────────────────────────────────────────────────
    p_video = sub.add_parser("video", help="Real-time recognition from a webcam.")
    p_video.add_argument("--camera", type=int, default=0)
    p_video.add_argument(
        "--model", default="models/face_encodings.pkl", help="Trained model path."
    )
    p_video.add_argument(
        "--detection-model", choices=["hog", "cnn"], default="hog"
    )
    p_video.add_argument("--tolerance", type=float, default=0.6)
    p_video.add_argument("--scale", type=float, default=0.5)
    p_video.add_argument("-v", "--verbose", action="store_true")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        from src.face_recognizer import FaceRecognizer

        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        recognizer = FaceRecognizer(
            model_path=args.model,
            detection_model=args.detection_model,
            tolerance=args.tolerance,
        )
        try:
            recognizer.train_from_directory(args.dataset)
        except (FileNotFoundError, ValueError) as exc:
            logging.error("Training failed: %s", exc)
            return 1
        print(f"\n\u2713 Model saved to '{args.model}'")
        print(f"  Persons  : {', '.join(recognizer.known_names)}")
        print(f"  Encodings: {len(recognizer)}")

    elif args.command == "image":
        from recognize_image import run as run_image
        return run_image(
            image=args.image,
            model=args.model,
            output=args.output,
            detection_model=args.detection_model,
            tolerance=args.tolerance,
            verbose=args.verbose,
        )

    elif args.command == "video":
        from recognize_video import run as run_video
        return run_video(
            camera=args.camera,
            model=args.model,
            detection_model=args.detection_model,
            tolerance=args.tolerance,
            scale=args.scale,
            verbose=args.verbose,
        )

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
