"""Shared utility functions for image I/O and visualisation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# OpenCV uses BGR channel order: (Blue, Green, Red)
_COLOUR_KNOWN = (0, 200, 0)       # green in BGR
_COLOUR_UNKNOWN = (0, 0, 200)     # red in BGR
_FONT = cv2.FONT_HERSHEY_DUPLEX
_FONT_SCALE = 0.6
_FONT_THICKNESS = 1
_BOX_THICKNESS = 2


def load_image(path: str) -> np.ndarray:
    """Load an image from *path* and return it as an **RGB** NumPy array.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If OpenCV cannot decode the file.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    bgr = cv2.imread(str(p))
    if bgr is None:
        raise ValueError(f"Could not decode image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_image(rgb_image: np.ndarray, path: str) -> None:
    """Save an RGB *rgb_image* to *path*."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), bgr)


def draw_bounding_box(
    rgb_image: np.ndarray,
    bounding_box: Tuple[int, int, int, int],
    label: str,
    confidence: Optional[float] = None,
) -> np.ndarray:
    """Draw a labelled bounding box on *rgb_image* (in-place) and return it.

    Parameters
    ----------
    rgb_image:
        NumPy array ``(H, W, 3)`` in **RGB** order.
    bounding_box:
        ``(top, right, bottom, left)`` in pixels (format returned by
        ``face_recognition``).
    label:
        Person name or ``"Unknown"``.
    confidence:
        Optional confidence score in ``[0.0, 1.0]``.

    Returns
    -------
    The *same* array with annotations drawn on it.
    """
    top, right, bottom, left = bounding_box
    colour = _COLOUR_KNOWN if label != "Unknown" else _COLOUR_UNKNOWN

    # Rectangle
    cv2.rectangle(rgb_image, (left, top), (right, bottom), colour, _BOX_THICKNESS)

    # Label background
    text = label if confidence is None else f"{label} ({confidence:.0%})"
    (tw, th), baseline = cv2.getTextSize(text, _FONT, _FONT_SCALE, _FONT_THICKNESS)
    cv2.rectangle(
        rgb_image,
        (left, bottom - th - baseline - 4),
        (left + tw + 4, bottom),
        colour,
        cv2.FILLED,
    )
    # Label text
    cv2.putText(
        rgb_image,
        text,
        (left + 2, bottom - baseline - 2),
        _FONT,
        _FONT_SCALE,
        (255, 255, 255),
        _FONT_THICKNESS,
    )
    return rgb_image


def frame_to_rgb(bgr_frame: np.ndarray) -> np.ndarray:
    """Convert a BGR OpenCV frame to RGB."""
    return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(rgb_image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to BGR for display with OpenCV."""
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
