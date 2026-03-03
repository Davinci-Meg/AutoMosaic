"""動画のモザイク処理"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm

from automosaic.detector.base import FaceDetector
from automosaic.mosaic.styles import MosaicStyle, apply_padding

logger = logging.getLogger("automosaic")


class VideoProcessor:
    """動画ファイルに対して顔検出・モザイク処理を行う。"""

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
        """動画を読み込み、各フレームの顔にモザイクを適用して保存する。

        Args:
            input_path: 入力動画のパス。
            output_path: 出力動画のパス。

        Returns:
            処理したフレームの総数。

        Raises:
            FileNotFoundError: 動画のオープンに失敗した場合。
        """
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"動画を開けません: {input_path}")

        fps: float = cap.get(cv2.CAP_PROP_FPS)
        width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 一時ファイルに映像のみ書き出す
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(tmp_fd)
        tmp_video = Path(tmp_path)

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp_video), fourcc, fps, (width, height))

        processed_frames = 0
        with tqdm(total=total_frames, desc=input_path.name, unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_h, frame_w = frame.shape[:2]
                boxes = self.detector.detect(frame)

                for box in boxes:
                    wl = self.whitelist_manager
                    if wl is not None and wl.is_whitelisted(
                        frame, box.x, box.y, box.width, box.height,
                    ):
                        continue

                    px, py, pw, ph = apply_padding(
                        box.x, box.y, box.width, box.height,
                        self.padding, frame_h, frame_w,
                    )
                    frame = self.mosaic_style.apply(
                        frame, px, py, pw, ph, self.strength,
                    )

                writer.write(frame)
                processed_frames += 1
                pbar.update(1)

        cap.release()
        writer.release()

        # ffmpeg で音声トラックを元動画からコピー
        self._merge_audio(input_path, tmp_video, output_path)

        # 一時ファイル削除
        tmp_video.unlink(missing_ok=True)

        logger.info(
            "%s -> %s (%d フレーム処理)",
            input_path.name, output_path.name, processed_frames,
        )
        return processed_frames

    def _merge_audio(
        self, input_path: Path, tmp_video: Path, output_path: Path
    ) -> None:
        """ffmpeg で映像と元動画の音声を結合する。"""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            logger.warning("ffmpeg が見つかりません。音声なしの映像のみ出力します。")
            shutil.move(str(tmp_video), str(output_path))
            return

        cmd = [
            ffmpeg,
            "-i", str(tmp_video),
            "-i", str(input_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            "-y",
            str(output_path),
        ]
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except subprocess.CalledProcessError:
            logger.warning("ffmpeg による音声結合に失敗しました。映像のみ出力します。")
            shutil.move(str(tmp_video), str(output_path))
