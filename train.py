#!/usr/bin/env python3
"""Train the face recognition model from a labelled dataset.

Usage
-----
    python train.py                          # uses defaults
    python train.py --dataset dataset/ --model models/face_encodings.pkl
    python train.py --dataset dataset/ --detection-model cnn
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.face_recognizer import FaceRecognizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode faces from a labelled dataset and save the model."
    )
    parser.add_argument(
        "--dataset",
        default="dataset",
        help="Path to the dataset directory (default: dataset/).",
    )
    parser.add_argument(
        "--model",
        default="models/face_encodings.pkl",
        help="Output path for the serialised encodings (default: models/face_encodings.pkl).",
    )
    parser.add_argument(
        "--detection-model",
        choices=["hog", "cnn"],
        default="hog",
        help="Face detection model: 'hog' (fast, CPU) or 'cnn' (accurate, GPU). Default: hog.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.6,
        help="Match tolerance – lower is stricter (default: 0.6).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

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

    print(f"\n✓ Model saved to '{args.model}'")
    print(f"  Persons : {', '.join(recognizer.known_names)}")
    print(f"  Encodings: {len(recognizer)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
