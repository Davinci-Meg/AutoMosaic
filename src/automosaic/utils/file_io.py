"""ファイル入出力ユーティリティ"""

from pathlib import Path

IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS: set[str] = {".mp4", ".mov", ".avi", ".mkv"}


def is_image(path: Path) -> bool:
    """拡張子で画像ファイルか判定する。"""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_video(path: Path) -> bool:
    """拡張子で動画ファイルか判定する。"""
    return path.suffix.lower() in VIDEO_EXTENSIONS


def is_supported(path: Path) -> bool:
    """対応するファイル形式か判定する。"""
    return is_image(path) or is_video(path)


def resolve_output_path(
    input_path: Path, output_dir: Path, suffix: str = ""
) -> Path:
    """出力先ファイルパスを解決する。

    Args:
        input_path: 入力ファイルのパス。
        output_dir: 出力先ディレクトリ。
        suffix: ファイル名に付加する接尾辞（例: "_mosaic"）。空なら元のファイル名。
    """
    stem = input_path.stem
    ext = input_path.suffix
    filename = f"{stem}{suffix}{ext}"
    return output_dir / filename


def ensure_output_dir(output_dir: Path) -> None:
    """出力先ディレクトリが存在しなければ作成する。"""
    output_dir.mkdir(parents=True, exist_ok=True)


def collect_files(input_path: Path, recursive: bool = False) -> list[Path]:
    """ディレクトリ内の対応ファイル一覧を取得する。

    input_path がファイルなら、対応形式であればそのファイルのみ返す。
    ディレクトリなら、中の対応ファイルを一覧で返す。

    Args:
        input_path: 入力パス（ファイルまたはディレクトリ）。
        recursive: True なら再帰的に検索する。
    """
    if input_path.is_file():
        return [input_path] if is_supported(input_path) else []

    if not input_path.is_dir():
        return []

    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in input_path.glob(pattern)
        if p.is_file() and is_supported(p)
    )
