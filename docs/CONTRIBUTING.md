# コントリビューションガイド

AutoMosaic への貢献を歓迎します。このドキュメントでは、開発環境のセットアップからプルリクエストの送信までの手順を説明します。

## 開発環境セットアップ

### 前提条件

- Python 3.10 以上
- Git

### 手順

1. リポジトリをフォーク・クローン

```bash
git clone https://github.com/<your-username>/automosaic.git
cd automosaic
```

2. 開発用依存パッケージも含めてインストール

```bash
pip install -e ".[all]"
```

これにより以下がインストールされます:

- 本体の依存パッケージ（opencv-python, mediapipe, click 等）
- YOLO 検出エンジン（ultralytics）
- ホワイトリスト機能（face_recognition）
- 開発ツール（pytest, ruff, mypy）

---

## コードスタイル

### Ruff（リンター・フォーマッター）

プロジェクトでは [Ruff](https://docs.astral.sh/ruff/) を使用しています。

```bash
# リントチェック
ruff check src/ tests/

# 自動修正
ruff check src/ tests/ --fix
```

設定は `pyproject.toml` に定義されています:

- ターゲットバージョン: Python 3.10
- 行の長さ: 88文字
- 有効なルール: E, F, W, I, N, UP, B, A, SIM

### mypy（型チェック）

```bash
mypy src/automosaic/ --ignore-missing-imports
```

すべての関数に型アノテーションを付けてください（`disallow_untyped_defs = true`）。

---

## テスト

### テスト実行

```bash
# 全テスト実行（カバレッジ付き）
pytest

# 特定のテストファイルを実行
pytest tests/test_mosaic.py

# 特定のテスト関数を実行
pytest tests/test_mosaic.py::test_pixelate_mosaic

# 詳細出力
pytest -v
```

### テストの書き方

- テストファイルは `tests/` ディレクトリに配置
- ファイル名は `test_<module>.py` の形式
- テスト関数名は `test_<動作の説明>` の形式
- フィクスチャは `tests/conftest.py` に定義
- テスト用のサンプルファイルは `tests/fixtures/` に配置

### カバレッジ目標

コアモジュール（detector, mosaic, processor）のカバレッジ **80% 以上**を維持してください。

---

## プルリクエストプロセス

### 1. ブランチを作成

```bash
git checkout -b feature/<機能名>
# または
git checkout -b fix/<バグの説明>
```

### 2. 変更を実装

- 既存のコードスタイルに従う
- 新機能にはテストを追加する
- 型アノテーションを付ける

### 3. チェックを通す

PR を送る前に、以下がすべてパスすることを確認してください:

```bash
ruff check src/ tests/
mypy src/automosaic/ --ignore-missing-imports
pytest
```

### 4. コミット

コミットメッセージは日本語で、変更内容を簡潔に記述してください。

```bash
git commit -m "ナンバープレート検出機能を追加"
```

### 5. プルリクエストを送信

- PR のタイトルは変更内容を簡潔に記述
- 本文には変更の目的と概要を記載
- 関連する Issue がある場合はリンクを記載

### CI チェック

PR を作成すると、GitHub Actions により以下が自動実行されます:

- Python 3.10 / 3.11 / 3.12 でのテスト
- Ruff によるリントチェック
- mypy による型チェック
- pytest によるテスト実行とカバレッジレポート

すべてのチェックがパスしないとマージできません。
