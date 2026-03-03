"""ホワイトリスト管理 - 参照顔との照合でモザイク除外を判定する"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from automosaic.detector.base import BoundingBox


class WhitelistManager:
    """ホワイトリスト画像から参照顔を読み込み、検出顔との照合を行う"""

    def __init__(self, whitelist_dir: Path, threshold: float = 0.6) -> None:
        self.whitelist_dir = whitelist_dir
        self.threshold = threshold
        self.logger = logging.getLogger("automosaic")
        self._reference_encodings: list[np.ndarray] = []
        self._load_reference_faces()

    def _load_reference_faces(self) -> None:
        """ホワイトリストディレクトリから参照顔の特徴量を読み込む"""
        try:
            import face_recognition
        except ImportError:
            self.logger.warning(
                "face_recognition がインストールされていないため"
                "ホワイトリスト機能は無効です"
            )
            return

        image_extensions = (".jpg", ".jpeg", ".png")
        for path in sorted(self.whitelist_dir.iterdir()):
            if path.suffix.lower() not in image_extensions:
                continue

            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)

            if not encodings:
                self.logger.warning(
                    "ホワイトリスト画像に顔がありません: %s", path.name
                )
                continue

            self._reference_encodings.extend(encodings)

        self.logger.info(
            "ホワイトリスト参照顔を %d 件読み込みました",
            len(self._reference_encodings),
        )

    def is_whitelisted(self, frame: np.ndarray, bbox: BoundingBox) -> bool:
        """検出された顔がホワイトリストに一致するか判定する"""
        if not self._reference_encodings:
            return False

        import face_recognition

        face_roi = frame[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]
        rgb_face = face_roi[:, :, ::-1]

        encodings = face_recognition.face_encodings(rgb_face)
        if not encodings:
            return False

        distances = face_recognition.face_distance(
            self._reference_encodings, encodings[0]
        )
        return float(np.min(distances)) <= (1 - self.threshold)

    @property
    def has_references(self) -> bool:
        """参照顔が読み込まれているかどうか"""
        return len(self._reference_encodings) > 0
