"""YOLOv8 による顔検出"""

import numpy as np
from ultralytics import YOLO

from automosaic.detector.base import BoundingBox, FaceDetector


class YOLOFaceDetector(FaceDetector):
    def __init__(self, confidence: float = 0.5, use_gpu: bool = False) -> None:
        self.confidence = confidence
        self.model = YOLO("yolov8n-face.pt")
        self.model.to("cuda" if use_gpu else "cpu")

    def detect(self, frame: np.ndarray) -> list[BoundingBox]:
        results = self.model(frame, conf=self.confidence, verbose=False)

        boxes: list[BoundingBox] = []
        for det in results[0].boxes:
            x1, y1, x2, y2 = det.xyxy[0].tolist()
            boxes.append(
                BoundingBox(
                    x=int(x1),
                    y=int(y1),
                    width=int(x2 - x1),
                    height=int(y2 - y1),
                    confidence=float(det.conf[0]),
                )
            )
        return boxes
