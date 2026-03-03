"""mosaic モジュールのテスト"""

import numpy as np
import pytest

from automosaic.mosaic.styles import (
    BlurMosaic,
    PixelateMosaic,
    apply_padding,
    get_mosaic_style,
)


class TestPixelateMosaic:
    def test_pixelate_mosaic_size_unchanged(self) -> None:
        """適用後の画像サイズが変わらないこと。"""
        image: np.ndarray = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mosaic = PixelateMosaic()
        result: np.ndarray = mosaic.apply(image.copy(), 10, 10, 50, 50, 10)
        assert result.shape == (100, 100, 3)

    def test_pixelate_mosaic_modifies_region(self) -> None:
        """対象領域のピクセル値が変わっていること。"""
        image: np.ndarray = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        original_region: np.ndarray = image[10:60, 10:60].copy()
        mosaic = PixelateMosaic()
        result: np.ndarray = mosaic.apply(image, 10, 10, 50, 50, 10)
        assert not np.array_equal(result[10:60, 10:60], original_region)


class TestBlurMosaic:
    def test_blur_mosaic_size_unchanged(self) -> None:
        """適用後の画像サイズが変わらないこと。"""
        image: np.ndarray = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mosaic = BlurMosaic()
        result: np.ndarray = mosaic.apply(image.copy(), 10, 10, 50, 50, 15)
        assert result.shape == (100, 100, 3)

    def test_blur_mosaic_modifies_region(self) -> None:
        """対象領域のピクセル値が変わっていること。"""
        image: np.ndarray = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        original_region: np.ndarray = image[10:60, 10:60].copy()
        mosaic = BlurMosaic()
        result: np.ndarray = mosaic.apply(image, 10, 10, 50, 50, 15)
        assert not np.array_equal(result[10:60, 10:60], original_region)


class TestGetMosaicStyle:
    def test_get_mosaic_style_valid(self) -> None:
        """'pixelate' と 'blur' が正しいインスタンスを返すこと。"""
        assert isinstance(get_mosaic_style("pixelate"), PixelateMosaic)
        assert isinstance(get_mosaic_style("blur"), BlurMosaic)

    def test_get_mosaic_style_invalid(self) -> None:
        """不明な名前で ValueError が発生すること。"""
        with pytest.raises(ValueError, match="Unknown mosaic style"):
            get_mosaic_style("unknown_style")


class TestApplyPadding:
    def test_apply_padding(self) -> None:
        """padding 適用の計算が正しいこと。"""
        x, y, w, h = apply_padding(50, 50, 100, 100, 0.2, 500, 500)
        assert x == 30
        assert y == 30
        assert w == 140
        assert h == 140

    def test_apply_padding_clipping(self) -> None:
        """フレーム境界でクリッピングされること。"""
        x, y, w, h = apply_padding(0, 0, 100, 100, 0.5, 120, 120)
        # x = max(0, 0 - 50) = 0
        # y = max(0, 0 - 50) = 0
        # w = min(120, 0 + 100 + 50) - 0 = 120
        # h = min(120, 0 + 100 + 50) - 0 = 120
        assert x == 0
        assert y == 0
        assert w == 120
        assert h == 120
