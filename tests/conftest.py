"""テスト用の共通フィクスチャ"""

from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from automosaic.detector.base import BoundingBox, FaceDetector


@pytest.fixture()
def sample_image(tmp_path: Path) -> Path:
    """100x100 の BGR 画像を作成して保存する。"""
    image: np.ndarray = np.zeros((100, 100, 3), dtype=np.uint8)
    path = tmp_path / "sample.jpg"
    cv2.imwrite(str(path), image)
    return path


@pytest.fixture()
def sample_image_with_face(tmp_path: Path) -> Path:
    """200x200 画像で (50,50)-(150,150) に肌色の楕円を描画して保存する。"""
    image: np.ndarray = np.zeros((200, 200, 3), dtype=np.uint8)
    # 肌色 (BGR: 135, 184, 222)
    cv2.ellipse(image, (100, 100), (50, 50), 0, 0, 360, (135, 184, 222), -1)
    path = tmp_path / "face.jpg"
    cv2.imwrite(str(path), image)
    return path


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    """出力ディレクトリを作成して返す。"""
    out = tmp_path / "output"
    out.mkdir()
    return out


@pytest.fixture()
def mock_detector() -> MagicMock:
    """FaceDetector のモック。detect() が固定の BoundingBox を返す。"""
    detector = MagicMock(spec=FaceDetector)
    detector.detect.return_value = [BoundingBox(10, 10, 50, 50, 0.9)]
    return detector
