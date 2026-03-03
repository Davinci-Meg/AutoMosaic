from abc import ABC, abstractmethod

import cv2
import numpy as np


def apply_padding(
    x: int, y: int, w: int, h: int, padding: float, frame_h: int, frame_w: int
) -> tuple[int, int, int, int]:
    """バウンディングボックスに padding 率で余白を追加する。

    padding=0.2 なら幅・高さをそれぞれ 20% ずつ拡張し、
    フレーム境界を超えないようクリッピングする。
    """
    pad_w = int(w * padding)
    pad_h = int(h * padding)

    new_x = max(0, x - pad_w)
    new_y = max(0, y - pad_h)
    new_w = min(frame_w, x + w + pad_w) - new_x
    new_h = min(frame_h, y + h + pad_h) - new_y

    return new_x, new_y, new_w, new_h


class MosaicStyle(ABC):
    """モザイクスタイルの抽象基底クラス。"""

    @abstractmethod
    def apply(
        self, image: np.ndarray, x: int, y: int, w: int, h: int, strength: int
    ) -> np.ndarray:
        """image の (x, y, w, h) 領域にモザイクを適用し、変更後の image を返す。"""
        ...


class PixelateMosaic(MosaicStyle):
    """ピクセル化によるモザイク処理。"""

    def apply(
        self, image: np.ndarray, x: int, y: int, w: int, h: int, strength: int
    ) -> np.ndarray:
        if strength <= 0 or strength >= w or strength >= h:
            return image

        region = image[y : y + h, x : x + w]
        small = cv2.resize(
            region, (w // strength, h // strength), interpolation=cv2.INTER_LINEAR
        )
        mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        image[y : y + h, x : x + w] = mosaic
        return image


class BlurMosaic(MosaicStyle):
    """ガウシアンブラーによるモザイク処理。"""

    def apply(
        self, image: np.ndarray, x: int, y: int, w: int, h: int, strength: int
    ) -> np.ndarray:
        region = image[y : y + h, x : x + w]
        ksize = strength if strength % 2 == 1 else strength + 1
        blurred = cv2.GaussianBlur(region, (ksize, ksize), 0)
        image[y : y + h, x : x + w] = blurred
        return image


def get_mosaic_style(name: str) -> MosaicStyle:
    """名前からモザイクスタイルのインスタンスを取得するファクトリ関数。"""
    styles: dict[str, type[MosaicStyle]] = {
        "pixelate": PixelateMosaic,
        "blur": BlurMosaic,
    }
    if name not in styles:
        msg = f"Unknown mosaic style: {name!r}. Choose from {list(styles)}"
        raise ValueError(msg)
    return styles[name]()
