"""画像のモザイク処理"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from automosaic.detector.base import FaceDetector
from automosaic.mosaic.styles import MosaicStyle, apply_padding

logger = logging.getLogger("automosaic")


class ImageProcessor:
    """画像ファイルに対して顔検出・モザイク処理を行う。"""

    def __init__(
        self,
        detector: FaceDetector,
        mosaic_style: MosaicStyle,
        strength: int = 15,
        padding: float = 0.2,
        whitelist_manager: Any = None,
    ) -> None:
        self.detector = detector
        self.mosaic_style = mosaic_style
        self.strength = strength
        self.padding = padding
        self.whitelist_manager = whitelist_manager

    def process(self, input_path: Path, output_path: Path) -> int:
        """画像を読み込み、顔にモザイクを適用して保存する。

        Args:
            input_path: 入力画像のパス。
            output_path: 出力画像のパス。

        Returns:
            検出した顔の数。

        Raises:
            FileNotFoundError: 画像の読み込みに失敗した場合。
        """
        frame: np.ndarray | None = cv2.imread(str(input_path))
        if frame is None:
            raise FileNotFoundError(f"画像を読み込めません: {input_path}")

        frame_h, frame_w = frame.shape[:2]
        boxes = self.detector.detect(frame)
        face_count = 0

        for box in boxes:
            wl = self.whitelist_manager
            if wl is not None and wl.is_whitelisted(
                frame, box.x, box.y, box.width, box.height,
            ):
                logger.debug(
                    "ホワイトリスト対象のためスキップ: %s", box,
                )
                continue

            px, py, pw, ph = apply_padding(
                box.x, box.y, box.width, box.height,
                self.padding, frame_h, frame_w,
            )
            frame = self.mosaic_style.apply(frame, px, py, pw, ph, self.strength)
            face_count += 1

        cv2.imwrite(str(output_path), frame)
        logger.info(
            "%s -> %s (%d 件のモザイク適用)",
            input_path.name, output_path.name, face_count,
        )

        # 元画像の EXIF を出力にコピー
        try:
            src_img = Image.open(input_path)
            exif_data = src_img.info.get("exif")
            if exif_data is not None:
                dst_img = Image.open(output_path)
                dst_img.save(output_path, exif=exif_data)
        except Exception:
            logger.debug("EXIF コピーをスキップしました: %s", input_path.name)

        return face_count
