#!/usr/bin/env python3
"""Recognize faces in a static image.

Usage
-----
    python recognize_image.py --image path/to/photo.jpg
    python recognize_image.py --image photo.jpg --output annotated.jpg
    python recognize_image.py --image photo.jpg --model models/face_encodings.pkl
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.face_recognizer import FaceRecognizer
from src.utils import draw_bounding_box, load_image, save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and identify faces in a static image."
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the annotated output image.",
    )
    parser.add_argument(
        "--model",
        default="models/face_encodings.pkl",
        help="Path to the trained model file (default: models/face_encodings.pkl).",
    )
    parser.add_argument(
        "--detection-model",
        choices=["hog", "cnn"],
        default="hog",
        help="Face detection model: 'hog' (fast, CPU) or 'cnn' (GPU). Default: hog.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.6,
        help="Match tolerance (default: 0.6).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def run(
    image: str,
    model: str = "models/face_encodings.pkl",
    output: str | None = None,
    detection_model: str = "hog",
    tolerance: float = 0.6,
    verbose: bool = False,
) -> int:
    """Recognize faces in *image* and optionally save the annotated result.

    Parameters
    ----------
    image:
        Path to the input image file.
    model:
        Path to the trained model (pickle) file.
    output:
        Optional path to save the annotated image. If ``None``, an OpenCV
        window is opened (falls back gracefully in headless environments).
    detection_model:
        ``"hog"`` or ``"cnn"``.
    tolerance:
        Euclidean-distance threshold for a positive match.
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

    # Load image
    try:
        rgb_image = load_image(image)
    except (FileNotFoundError, ValueError) as exc:
        logging.error("Cannot load image: %s", exc)
        return 1

    # Recognize
    results = recognizer.recognize(rgb_image)

    if not results:
        print("No faces detected in the image.")
        return 0

    print(f"\nDetected {len(results)} face(s):")
    for name, confidence, bbox in results:
        print(f"  \u2022 {name}  (confidence: {confidence:.1%})")
        draw_bounding_box(rgb_image, bbox, name, confidence)

    # Save / display
    if output:
        save_image(rgb_image, output)
        print(f"\nAnnotated image saved to '{output}'")
    else:
        # Try to display with OpenCV; gracefully fail in headless environments
        try:
            import cv2
            from src.utils import rgb_to_bgr
            cv2.imshow("Face Recognition", rgb_to_bgr(rgb_image))
            print("\nPress any key to close the window\u2026")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:  # noqa: BLE001
            print("\nNote: GUI display not available. Use --output to save the result.")

    return 0


def main() -> int:
    args = parse_args()
    return run(
        image=args.image,
        model=args.model,
        output=args.output,
        detection_model=args.detection_model,
        tolerance=args.tolerance,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())

