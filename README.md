# AutoMosaic

[![CI](https://github.com/automosaic/automosaic/actions/workflows/ci.yml/badge.svg)](https://github.com/automosaic/automosaic/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/automosaic)](https://pypi.org/project/automosaic/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

画像・動画の顔を自動検出し、モザイク処理を施すコマンドラインツールです。プライバシー保護やSNS投稿前の匿名化処理を、専門的な画像編集ソフトを使わずにワンコマンドで実行できます。

## 特徴

- **高精度な顔検出** -- ディープラーニングベース（MediaPipe / YOLO 選択可能）
- **画像・動画の両方に対応** -- JPG, PNG, BMP, WebP, MP4, MOV, AVI, MKV
- **モザイクスタイル選択** -- ピクセレート / ガウスぼかし、強度調整可能
- **バッチ処理** -- ディレクトリ単位での一括処理（再帰オプション付き）
- **ホワイトリスト機能** -- 特定の顔をモザイク対象から除外
- **GPU アクセラレーション** -- CUDA 対応で高速処理

## インストール

```bash
pip install automosaic
```

### オプション依存パッケージ

```bash
# YOLO 高精度検出エンジン
pip install "automosaic[yolo]"

# ホワイトリスト機能（顔照合）
pip install "automosaic[whitelist]"

# すべてのオプション + 開発ツール
pip install "automosaic[all]"
```

## 使い方

### 基本（画像1枚にモザイク処理）

```bash
automosaic photo.jpg
```

### 動画にぼかしモザイクを適用

```bash
automosaic video.mp4 --style blur --strength 25
```

### ディレクトリ内を一括処理

```bash
automosaic ./photos/ -o ./output/ --recursive
```

### ホワイトリスト（自分の顔を除外）

```bash
automosaic group_photo.jpg --whitelist ./my_face/
```

### 高精度モード（YOLO + GPU）

```bash
automosaic event.mp4 --detector yolo --gpu
```

## オプション一覧

| オプション | 短縮形 | 型 | デフォルト | 説明 |
|-----------|--------|-----|-----------|------|
| `<INPUT>` | - | PATH | (必須) | 入力ファイルまたはディレクトリ |
| `--output` | `-o` | PATH | `./output/` | 出力先パス |
| `--style` | `-s` | STR | `pixelate` | モザイクスタイル (`pixelate` / `blur`) |
| `--strength` | `-S` | INT | `15` | モザイク強度 (1-50) |
| `--detector` | `-d` | STR | `mediapipe` | 検出エンジン (`mediapipe` / `yolo`) |
| `--confidence` | `-c` | FLOAT | `0.5` | 検出信頼度の閾値 (0.0-1.0) |
| `--padding` | `-p` | FLOAT | `0.2` | 検出領域の余白率 (0.0-1.0) |
| `--whitelist` | `-w` | PATH | `None` | 除外する顔の参照画像ディレクトリ |
| `--recursive` | `-r` | FLAG | `False` | ディレクトリを再帰的に処理 |
| `--overwrite` | - | FLAG | `False` | 既存の出力ファイルを上書き |
| `--gpu` | - | FLAG | `False` | GPU アクセラレーションを有効化 |
| `--verbose` | `-v` | FLAG | `False` | 詳細ログの出力 |
| `--dry-run` | - | FLAG | `False` | 処理をシミュレーション（実行しない） |
| `--version` | `-V` | FLAG | - | バージョン情報の表示 |

## ライセンス

[MIT License](LICENSE)
