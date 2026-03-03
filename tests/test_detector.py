"""detector モジュールのテスト"""

import pytest

from automosaic.detector.base import BoundingBox, FaceDetector


class TestBoundingBox:
    def test_bounding_box_fields(self) -> None:
        """BoundingBox の各フィールドにアクセスできること。"""
        box = BoundingBox(
            x=10, y=20, width=100, height=80, confidence=0.95,
        )
        assert box.x == 10
        assert box.y == 20
        assert box.width == 100
        assert box.height == 80
        assert box.confidence == 0.95


class TestFaceDetector:
    def test_face_detector_is_abstract(self) -> None:
        """FaceDetector を直接インスタンス化すると TypeError が発生すること。"""
        with pytest.raises(TypeError):
            FaceDetector()  # type: ignore[abstract]
