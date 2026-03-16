"""Face detection utilities using dlib's HOG-based detector (via face_recognition)."""

from __future__ import annotations

import logging
from typing import List, Tuple

import face_recognition
import numpy as np

logger = logging.getLogger(__name__)

# Type alias: bounding box as (top, right, bottom, left) pixel coordinates
BoundingBox = Tuple[int, int, int, int]


class FaceDetector:
    """Detect face locations inside an RGB image array.

    Parameters
    ----------
    model:
        Detection model to use.  ``"hog"`` (default) is fast and works well on
        CPU.  ``"cnn"`` is more accurate but requires a GPU (or is slow on CPU).
    upsamples:
        How many times to upsample the image before detecting faces.  A higher
        number helps detect smaller faces at the cost of speed.
    """

    def __init__(self, model: str = "hog", upsamples: int = 1) -> None:
        if model not in {"hog", "cnn"}:
            raise ValueError("model must be 'hog' or 'cnn'")
        self.model = model
        self.upsamples = upsamples

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, rgb_image: np.ndarray) -> List[BoundingBox]:
        """Return bounding boxes for every face found in *rgb_image*.

        Parameters
        ----------
        rgb_image:
            A NumPy array with shape ``(H, W, 3)`` in **RGB** order (not BGR).

        Returns
        -------
        list of (top, right, bottom, left) tuples
        """
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError("rgb_image must be an (H, W, 3) RGB array")

        locations = face_recognition.face_locations(
            rgb_image, number_of_times_to_upsample=self.upsamples, model=self.model
        )
        logger.debug("Detected %d face(s)", len(locations))
        return locations

    def count(self, rgb_image: np.ndarray) -> int:
        """Return the number of faces detected in *rgb_image*."""
        return len(self.detect(rgb_image))
