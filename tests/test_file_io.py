"""file_io モジュールのテスト"""

from pathlib import Path

from automosaic.utils.file_io import (
    collect_files,
    is_image,
    is_supported,
    is_video,
    resolve_output_path,
)


class TestIsImage:
    def test_is_image(self) -> None:
        """画像拡張子が正しく判定されること。"""
        assert is_image(Path("photo.jpg")) is True
        assert is_image(Path("photo.jpeg")) is True
        assert is_image(Path("photo.png")) is True
        assert is_image(Path("photo.bmp")) is True
        assert is_image(Path("photo.webp")) is True
        assert is_image(Path("video.mp4")) is False
        assert is_image(Path("file.txt")) is False


class TestIsVideo:
    def test_is_video(self) -> None:
        """動画拡張子が正しく判定されること。"""
        assert is_video(Path("clip.mp4")) is True
        assert is_video(Path("clip.mov")) is True
        assert is_video(Path("clip.avi")) is True
        assert is_video(Path("clip.mkv")) is True
        assert is_video(Path("photo.jpg")) is False
        assert is_video(Path("file.txt")) is False


class TestIsSupported:
    def test_is_supported(self) -> None:
        """対応する拡張子が正しく判定されること。"""
        assert is_supported(Path("photo.jpg")) is True
        assert is_supported(Path("clip.mp4")) is True
        assert is_supported(Path("file.txt")) is False
        assert is_supported(Path("data.csv")) is False


class TestCollectFiles:
    def test_collect_files_directory(self, tmp_path: Path) -> None:
        """ディレクトリ内のファイル収集。"""
        (tmp_path / "a.jpg").touch()
        (tmp_path / "b.png").touch()
        (tmp_path / "c.txt").touch()
        files: list[Path] = collect_files(tmp_path)
        names: list[str] = [f.name for f in files]
        assert "a.jpg" in names
        assert "b.png" in names
        assert "c.txt" not in names

    def test_collect_files_recursive(self, tmp_path: Path) -> None:
        """再帰検索でサブディレクトリのファイルも収集されること。"""
        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "a.jpg").touch()
        (sub / "b.png").touch()
        (sub / "c.txt").touch()

        non_recursive: list[Path] = collect_files(tmp_path, recursive=False)
        recursive: list[Path] = collect_files(tmp_path, recursive=True)

        non_recursive_names: list[str] = [f.name for f in non_recursive]
        recursive_names: list[str] = [f.name for f in recursive]

        assert "a.jpg" in non_recursive_names
        assert "b.png" not in non_recursive_names
        assert "a.jpg" in recursive_names
        assert "b.png" in recursive_names
        assert "c.txt" not in recursive_names


class TestResolveOutputPath:
    def test_resolve_output_path(self, tmp_path: Path) -> None:
        """パス解決が正しいこと。"""
        result: Path = resolve_output_path(
            Path("photos/input.jpg"), tmp_path, suffix="_mosaic"
        )
        assert result == tmp_path / "input_mosaic.jpg"

        result_no_suffix: Path = resolve_output_path(
            Path("photos/input.jpg"), tmp_path
        )
        assert result_no_suffix == tmp_path / "input.jpg"
