"""MediaPipe Face Detection (Tasks API) による顔検出"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from automosaic.detector.base import BoundingBox, FaceDetector

_MODEL_DIR = Path(__file__).parent.parent / "models"
_MODEL_PATH = _MODEL_DIR / "blaze_face_short_range.tflite"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/latest/"
    "blaze_face_short_range.tflite"
)

logger = logging.getLogger("automosaic")


def _ensure_model() -> Path:
    """モデルファイルが無ければダウンロードする。"""
    if _MODEL_PATH.exists():
        return _MODEL_PATH
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("モデルをダウンロード中: %s", _MODEL_URL)
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    logger.info("モデルを保存しました: %s", _MODEL_PATH)
    return _MODEL_PATH


class MediaPipeFaceDetector(FaceDetector):
    def __init__(self, confidence: float = 0.5) -> None:
        model_path = _ensure_model()
        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(
                model_asset_path=str(model_path),
            ),
            min_detection_confidence=confidence,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def detect(self, frame: np.ndarray) -> list[BoundingBox]:
        rgb = frame[:, :, ::-1]
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=rgb.copy(),
        )
        result = self._detector.detect(mp_image)

        if not result.detections:
            return []

        boxes: list[BoundingBox] = []
        for detection in result.detections:
            bb = detection.bounding_box
            boxes.append(
                BoundingBox(
                    x=bb.origin_x,
                    y=bb.origin_y,
                    width=bb.width,
                    height=bb.height,
                    confidence=detection.categories[0].score,
                )
            )
        return boxes
