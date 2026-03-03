"""processor モジュールのテスト"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from automosaic.mosaic.styles import PixelateMosaic
from automosaic.processor.image import ImageProcessor


class TestImageProcessor:
    def test_image_processor_creates_output(
        self, sample_image: Path, output_dir: Path, mock_detector: MagicMock
    ) -> None:
        """mock_detector と PixelateMosaic で処理し、出力ファイルが生成されること。"""
        output_path: Path = output_dir / "out.jpg"
        processor = ImageProcessor(
            detector=mock_detector,
            mosaic_style=PixelateMosaic(),
            strength=10,
            padding=0.2,
        )
        processor.process(sample_image, output_path)
        assert output_path.exists()

    def test_image_processor_returns_face_count(
        self, sample_image: Path, output_dir: Path, mock_detector: MagicMock
    ) -> None:
        """検出数と一致すること。"""
        output_path: Path = output_dir / "out.jpg"
        processor = ImageProcessor(
            detector=mock_detector,
            mosaic_style=PixelateMosaic(),
            strength=10,
            padding=0.2,
        )
        count: int = processor.process(sample_image, output_path)
        assert count == 1

    def test_image_processor_file_not_found(
        self, output_dir: Path, mock_detector: MagicMock
    ) -> None:
        """存在しないパスで FileNotFoundError が発生すること。"""
        processor = ImageProcessor(
            detector=mock_detector,
            mosaic_style=PixelateMosaic(),
            strength=10,
            padding=0.2,
        )
        with pytest.raises(FileNotFoundError):
            processor.process(Path("/nonexistent/image.jpg"), output_dir / "out.jpg")
