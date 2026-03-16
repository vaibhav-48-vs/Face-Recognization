"""Unit tests for the face recognition project.

These tests are designed to run without a physical camera or a pre-trained
model – they mock heavy dependencies where needed so the CI can run them
without GPU or large model downloads.
"""

from __future__ import annotations

import pickle
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgb_image(h: int = 120, w: int = 120) -> np.ndarray:
    """Return a blank RGB image array."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_encoding() -> np.ndarray:
    """Return a fake 128-d face encoding."""
    rng = np.random.default_rng(42)
    return rng.random(128).astype(np.float64)


# ---------------------------------------------------------------------------
# FaceDetector
# ---------------------------------------------------------------------------

class TestFaceDetector:
    def test_invalid_model_raises(self):
        from src.face_detector import FaceDetector
        with pytest.raises(ValueError, match="model must be"):
            FaceDetector(model="sift")

    def test_detect_returns_list(self):
        from src.face_detector import FaceDetector
        detector = FaceDetector()
        img = _make_rgb_image()
        with patch("face_recognition.face_locations", return_value=[]) as mock_fl:
            result = detector.detect(img)
        mock_fl.assert_called_once()
        assert isinstance(result, list)

    def test_detect_invalid_image_raises(self):
        from src.face_detector import FaceDetector
        detector = FaceDetector()
        bad_img = np.zeros((100, 100), dtype=np.uint8)  # 2-D (greyscale)
        with pytest.raises(ValueError):
            detector.detect(bad_img)

    def test_count_delegates_to_detect(self):
        from src.face_detector import FaceDetector
        detector = FaceDetector()
        fake_locations = [(0, 60, 60, 0), (0, 120, 60, 60)]
        with patch.object(detector, "detect", return_value=fake_locations):
            assert detector.count(_make_rgb_image()) == 2


# ---------------------------------------------------------------------------
# FaceRecognizer
# ---------------------------------------------------------------------------

class TestFaceRecognizer:
    def test_empty_recognizer_has_zero_len(self, tmp_path):
        from src.face_recognizer import FaceRecognizer
        r = FaceRecognizer(model_path=str(tmp_path / "model.pkl"))
        assert len(r) == 0
        assert r.known_names == []

    def test_save_and_load(self, tmp_path):
        from src.face_recognizer import FaceRecognizer
        model_file = tmp_path / "model.pkl"
        r = FaceRecognizer(model_path=str(model_file))
        enc = _make_encoding()
        r._encodings = [enc]
        r._names = ["Alice"]
        r.save()

        r2 = FaceRecognizer(model_path=str(model_file))
        assert len(r2) == 1
        assert r2.known_names == ["Alice"]
        np.testing.assert_array_equal(r2._encodings[0], enc)

    def test_load_nonexistent_raises(self, tmp_path):
        from src.face_recognizer import FaceRecognizer
        r = FaceRecognizer(model_path=str(tmp_path / "ghost.pkl"))
        with pytest.raises(FileNotFoundError):
            r.load()

    def test_recognize_raises_without_encodings(self, tmp_path):
        from src.face_recognizer import FaceRecognizer
        r = FaceRecognizer(model_path=str(tmp_path / "model.pkl"))
        with pytest.raises(RuntimeError, match="No encodings loaded"):
            r.recognize(_make_rgb_image())

    def test_recognize_returns_results(self, tmp_path):
        from src.face_recognizer import FaceRecognizer
        model_file = tmp_path / "model.pkl"
        r = FaceRecognizer(model_path=str(model_file))
        enc = _make_encoding()
        r._encodings = [enc]
        r._names = ["Bob"]

        fake_loc = [(10, 60, 60, 10)]
        with (
            patch("face_recognition.face_locations", return_value=fake_loc),
            patch("face_recognition.face_encodings", return_value=[enc]),
            patch("face_recognition.face_distance", return_value=np.array([0.0])),
        ):
            results = r.recognize(_make_rgb_image())

        assert len(results) == 1
        name, confidence, bbox = results[0]
        assert name == "Bob"
        assert confidence == pytest.approx(1.0)
        assert bbox == (10, 60, 60, 10)

    def test_recognize_unknown_face(self, tmp_path):
        from src.face_recognizer import FaceRecognizer
        model_file = tmp_path / "model.pkl"
        r = FaceRecognizer(model_path=str(model_file))
        known_enc = _make_encoding()
        r._encodings = [known_enc]
        r._names = ["Charlie"]

        query_enc = _make_encoding() + 1.0  # very different

        fake_loc = [(10, 60, 60, 10)]
        with (
            patch("face_recognition.face_locations", return_value=fake_loc),
            patch("face_recognition.face_encodings", return_value=[query_enc]),
            patch("face_recognition.face_distance", return_value=np.array([0.9])),
        ):
            results = r.recognize(_make_rgb_image())

        assert results[0][0] == "Unknown"

    def test_train_from_directory_missing_dir(self, tmp_path):
        from src.face_recognizer import FaceRecognizer
        r = FaceRecognizer(model_path=str(tmp_path / "model.pkl"))
        with pytest.raises(FileNotFoundError):
            r.train_from_directory(str(tmp_path / "nonexistent"))

    def test_train_from_directory_empty_dir(self, tmp_path):
        from src.face_recognizer import FaceRecognizer
        r = FaceRecognizer(model_path=str(tmp_path / "model.pkl"))
        with pytest.raises(ValueError, match="No person sub-directories"):
            r.train_from_directory(str(tmp_path))

    def test_train_from_directory_encodes_images(self, tmp_path):
        from src.face_recognizer import FaceRecognizer

        # Create fake directory structure
        alice_dir = tmp_path / "Alice"
        alice_dir.mkdir()
        (alice_dir / "img1.jpg").write_bytes(b"fake")

        enc = _make_encoding()
        model_file = tmp_path / "model.pkl"
        r = FaceRecognizer(model_path=str(model_file))

        with (
            patch("face_recognition.load_image_file", return_value=_make_rgb_image()),
            patch("face_recognition.face_locations", return_value=[(0, 50, 50, 0)]),
            patch("face_recognition.face_encodings", return_value=[enc]),
        ):
            r.train_from_directory(str(tmp_path))

        assert len(r) == 1
        assert r.known_names == ["Alice"]
        assert model_file.exists()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class TestUtils:
    def test_load_image_missing_raises(self, tmp_path):
        from src.utils import load_image
        with pytest.raises(FileNotFoundError):
            load_image(str(tmp_path / "ghost.jpg"))

    def test_save_and_reload(self, tmp_path):
        from src.utils import load_image, save_image
        import cv2

        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        out_path = str(tmp_path / "out.png")
        save_image(img, out_path)

        loaded = load_image(out_path)
        assert loaded.shape == (50, 50, 3)

    def test_draw_bounding_box_returns_image(self):
        from src.utils import draw_bounding_box
        img = _make_rgb_image(200, 200)
        result = draw_bounding_box(img, (10, 100, 100, 10), "Alice", 0.95)
        assert result is img  # in-place

    def test_draw_bounding_box_unknown(self):
        from src.utils import draw_bounding_box
        img = _make_rgb_image(200, 200)
        # Should not raise for Unknown label
        draw_bounding_box(img, (10, 100, 100, 10), "Unknown")

    def test_frame_to_rgb_and_back(self):
        from src.utils import frame_to_rgb, rgb_to_bgr
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255  # blue channel
        rgb = frame_to_rgb(bgr)
        assert rgb[0, 0, 2] == 255  # should now be in red channel
        restored = rgb_to_bgr(rgb)
        np.testing.assert_array_equal(restored, bgr)
