"""MediaPipe Face Detection (Tasks API) による顔検出"""

from __future__ import annotations

from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from automosaic.detector.base import BoundingBox, FaceDetector

_MODEL_PATH = Path(__file__).parent.parent / "models" / "blaze_face_short_range.tflite"


class MediaPipeFaceDetector(FaceDetector):
    def __init__(self, confidence: float = 0.5) -> None:
        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(
                model_asset_path=str(_MODEL_PATH),
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

        h, w = frame.shape[:2]
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
