"""顔検出の基底クラスとデータ型"""

from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np


class BoundingBox(NamedTuple):
    x: int
    y: int
    width: int
    height: int
    confidence: float


class FaceDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[BoundingBox]:
        """フレーム(BGR画像)から顔を検出しBoundingBoxリストを返す"""
        ...
