"""AutoMosaic CLI エントリーポイント"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from automosaic import __version__
from automosaic.detector.base import FaceDetector
from automosaic.mosaic.styles import get_mosaic_style
from automosaic.utils.file_io import (
    collect_files,
    ensure_output_dir,
    is_image,
    is_video,
    resolve_output_path,
)
from automosaic.utils.logger import setup_logger


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="./output/",
    help="出力先ディレクトリ",
)
@click.option(
    "--style", "-s",
    type=click.Choice(["pixelate", "blur"]),
    default="pixelate",
    help="モザイクスタイル",
)
@click.option(
    "--strength", "-S",
    type=click.IntRange(1, 50),
    default=15,
    help="モザイクの強度",
)
@click.option(
    "--detector", "-d",
    type=click.Choice(["mediapipe", "yolo"]),
    default="mediapipe",
    help="顔検出エンジン",
)
@click.option(
    "--confidence", "-c",
    type=click.FloatRange(0.0, 1.0),
    default=0.5,
    help="検出の信頼度閾値",
)
@click.option(
    "--padding", "-p",
    type=click.FloatRange(0.0, 1.0),
    default=0.2,
    help="モザイク領域のパディング率",
)
@click.option(
    "--whitelist", "-w",
    type=click.Path(),
    default=None,
    help="ホワイトリスト画像ディレクトリ",
)
@click.option(
    "--recursive", "-r",
    is_flag=True,
    default=False,
    help="サブディレクトリを再帰的に検索",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="既存ファイルを上書き",
)
@click.option(
    "--gpu",
    is_flag=True,
    default=False,
    help="GPU を使用する (YOLO のみ)",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="詳細なログ出力",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="処理対象ファイルの一覧のみ表示",
)
@click.version_option(version=__version__, prog_name="automosaic")
def main(
    input_path: str,
    output: str,
    style: str,
    strength: int,
    detector: str,
    confidence: float,
    padding: float,
    whitelist: str | None,
    recursive: bool,
    overwrite: bool,
    gpu: bool,
    verbose: bool,
    dry_run: bool,
) -> None:
    """画像・動画の顔を自動検出してモザイク処理を行う。

    INPUT_PATH: 処理対象のファイルまたはディレクトリのパス
    """
    logger = setup_logger(verbose)

    face_detector = _create_detector(detector, confidence, gpu, logger)
    mosaic_style = get_mosaic_style(style)

    whitelist_manager = None
    if whitelist is not None:
        from automosaic.whitelist.manager import WhitelistManager

        whitelist_manager = WhitelistManager(Path(whitelist))

    src = Path(input_path)
    files = collect_files(src, recursive)

    if not files:
        logger.error(
            "処理対象のファイルが見つかりません: %s", src,
        )
        sys.exit(1)

    if dry_run:
        click.echo(f"処理対象ファイル ({len(files)} 件):")
        for f in files:
            click.echo(f"  {f}")
        return

    output_dir = Path(output)
    ensure_output_dir(output_dir)

    from automosaic.processor.image import ImageProcessor
    from automosaic.processor.video import VideoProcessor

    image_proc = ImageProcessor(
        detector=face_detector,
        mosaic_style=mosaic_style,
        strength=strength,
        padding=padding,
        whitelist_manager=whitelist_manager,
    )
    video_proc = VideoProcessor(
        detector=face_detector,
        mosaic_style=mosaic_style,
        strength=strength,
        padding=padding,
        whitelist_manager=whitelist_manager,
    )

    success_count = 0
    error_count = 0

    for file_path in files:
        out = resolve_output_path(file_path, output_dir)

        if not overwrite and out.exists():
            logger.info("スキップ (既存): %s", file_path.name)
            continue

        try:
            if is_image(file_path):
                image_proc.process(file_path, out)
            elif is_video(file_path):
                video_proc.process(file_path, out)
            success_count += 1
        except Exception:
            logger.exception(
                "処理中にエラーが発生しました: %s",
                file_path.name,
            )
            error_count += 1

    logger.info(
        "完了: %d 件成功, %d 件エラー (全 %d 件)",
        success_count, error_count, len(files),
    )


def _create_detector(
    name: str,
    confidence: float,
    gpu: bool,
    logger: logging.Logger,
) -> FaceDetector:
    """検出エンジン名からインスタンスを生成する。"""
    if name == "yolo":
        from automosaic.detector.yolo import YOLOFaceDetector

        if gpu:
            try:
                import torch

                if not torch.cuda.is_available():
                    logger.warning(
                        "CUDA が利用できません。CPU で実行します。",
                    )
                    gpu = False
            except ImportError:
                logger.warning(
                    "PyTorch が見つかりません。CPU で実行します。",
                )
                gpu = False

        return YOLOFaceDetector(confidence=confidence, use_gpu=gpu)

    from automosaic.detector.mediapipe import MediaPipeFaceDetector
    return MediaPipeFaceDetector(confidence=confidence)


if __name__ == "__main__":
    main()
