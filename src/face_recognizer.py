"""Core face recognition engine – encoding, storage, and matching."""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import face_recognition
import numpy as np
from tqdm import tqdm

from .face_detector import BoundingBox, FaceDetector

logger = logging.getLogger(__name__)

# 128-dimensional encoding produced by face_recognition
Encoding = np.ndarray


class FaceRecognizer:
    """Encode known faces from a dataset and recognise them in new images.

    Parameters
    ----------
    model_path:
        Path to the pickle file used to persist encodings between sessions.
        If the file exists it is loaded automatically on construction.
    detection_model:
        ``"hog"`` (fast, CPU) or ``"cnn"`` (accurate, GPU preferred).
    tolerance:
        Maximum Euclidean distance between encodings to consider a match.
        Lower values are stricter.  ``0.6`` is the recommended default.
    """

    SUPPORTED_EXTENSIONS: Tuple[str, ...] = (
        ".jpg", ".jpeg", ".png", ".bmp", ".webp",
    )

    def __init__(
        self,
        model_path: str = "models/face_encodings.pkl",
        detection_model: str = "hog",
        tolerance: float = 0.6,
    ) -> None:
        self.model_path = Path(model_path)
        self.tolerance = tolerance
        self._detector = FaceDetector(model=detection_model)

        # Known encodings and labels
        self._encodings: List[Encoding] = []
        self._names: List[str] = []

        if self.model_path.exists():
            self.load()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def known_names(self) -> List[str]:
        """Sorted, de-duplicated list of names in the model."""
        return sorted(set(self._names))

    def __len__(self) -> int:
        return len(self._encodings)

    # ------------------------------------------------------------------
    # Training / building the model
    # ------------------------------------------------------------------

    def train_from_directory(self, dataset_dir: str) -> None:
        """Encode all faces found in *dataset_dir* and update the model.

        The directory must follow the layout::

            dataset_dir/
            ├── PersonA/
            │   ├── img1.jpg
            │   └── img2.jpg
            └── PersonB/
                └── img1.jpg

        Sub-directory names are used as person identifiers.

        Parameters
        ----------
        dataset_dir:
            Path to the dataset root directory.
        """
        dataset_path = Path(dataset_dir)
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        new_encodings: List[Encoding] = []
        new_names: List[str] = []

        person_dirs = sorted(
            p for p in dataset_path.iterdir() if p.is_dir()
        )
        if not person_dirs:
            raise ValueError(f"No person sub-directories found in {dataset_dir}")

        logger.info("Found %d person(s) to encode.", len(person_dirs))

        for person_dir in tqdm(person_dirs, desc="Encoding persons", unit="person"):
            name = person_dir.name
            image_paths = [
                p for p in sorted(person_dir.iterdir())
                if p.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ]
            if not image_paths:
                logger.warning("No images found for '%s' – skipping.", name)
                continue

            person_encodings = self._encode_images(image_paths, name)
            new_encodings.extend(person_encodings)
            new_names.extend([name] * len(person_encodings))
            logger.info(
                "  %s: encoded %d face(s) from %d image(s).",
                name, len(person_encodings), len(image_paths),
            )

        if not new_encodings:
            raise ValueError("No valid face encodings were produced from the dataset.")

        self._encodings = new_encodings
        self._names = new_names
        logger.info(
            "Training complete – %d encoding(s) for %d person(s).",
            len(self._encodings), len(self.known_names),
        )
        self.save()

    def add_person(self, name: str, image_paths: List[str]) -> int:
        """Encode faces from *image_paths* and add them under *name*.

        Returns the number of new encodings added.
        """
        paths = [Path(p) for p in image_paths]
        new_encodings = self._encode_images(paths, name)
        self._encodings.extend(new_encodings)
        self._names.extend([name] * len(new_encodings))
        self.save()
        return len(new_encodings)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def recognize(
        self, rgb_image: np.ndarray
    ) -> List[Tuple[str, float, BoundingBox]]:
        """Detect and identify all faces in *rgb_image*.

        Parameters
        ----------
        rgb_image:
            NumPy array ``(H, W, 3)`` in **RGB** order.

        Returns
        -------
        list of ``(name, confidence, bounding_box)`` tuples.
        ``name`` is ``"Unknown"`` when no match is found.
        ``confidence`` is the similarity score in ``[0.0, 1.0]``; higher is better.
        ``bounding_box`` is ``(top, right, bottom, left)`` in pixels.
        """
        if not self._encodings:
            raise RuntimeError(
                "No encodings loaded.  Run train_from_directory() or load() first."
            )

        locations = self._detector.detect(rgb_image)
        if not locations:
            return []

        face_encodings = face_recognition.face_encodings(rgb_image, locations)

        results: List[Tuple[str, float, BoundingBox]] = []
        for encoding, location in zip(face_encodings, locations):
            name, confidence = self._match(encoding)
            results.append((name, confidence, location))

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist encodings and labels to *self.model_path*."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"encodings": self._encodings, "names": self._names}
        with open(self.model_path, "wb") as fh:
            pickle.dump(payload, fh)
        logger.info("Model saved to %s", self.model_path)

    def load(self) -> None:
        """Load encodings and labels from *self.model_path*."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        with open(self.model_path, "rb") as fh:
            payload = pickle.load(fh)  # noqa: S301 – trusted local file
        self._encodings = payload["encodings"]
        self._names = payload["names"]
        logger.info(
            "Loaded %d encoding(s) for %d person(s) from %s",
            len(self._encodings), len(self.known_names), self.model_path,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_images(
        self, paths: List[Path], person_name: str
    ) -> List[Encoding]:
        """Return face encodings extracted from a list of image files."""
        encodings: List[Encoding] = []
        for path in paths:
            try:
                image = face_recognition.load_image_file(str(path))
                locs = self._detector.detect(image)
                if not locs:
                    logger.warning(
                        "No face detected in %s – skipping.", path.name
                    )
                    continue
                enc = face_recognition.face_encodings(image, locs)
                encodings.extend(enc)
            except Exception as exc:  # pragma: no cover
                logger.error("Error processing %s: %s", path, exc)
        return encodings

    def _match(self, encoding: Encoding) -> Tuple[str, float]:
        """Find the best matching name for *encoding*.

        Returns ``(name, confidence)`` where confidence is in ``[0.0, 1.0]``.
        """
        if not self._encodings:
            return "Unknown", 0.0

        distances: np.ndarray = face_recognition.face_distance(
            self._encodings, encoding
        )
        best_idx: int = int(np.argmin(distances))
        best_distance: float = float(distances[best_idx])

        # confidence: closer distance → higher confidence
        confidence: float = max(0.0, 1.0 - best_distance)

        if best_distance <= self.tolerance:
            return self._names[best_idx], confidence
        return "Unknown", confidence
