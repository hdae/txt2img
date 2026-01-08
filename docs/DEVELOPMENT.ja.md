# 開発ガイド

## プロジェクト構成

- **`server/`**: Python FastAPI バックエンド。
  - **`src/`**: アプリケーション・ソースコード。
  - **`presets/`**: モデル設定JSONファイル。
- **`client/`**: React/Vite フロントエンド。
- **`outputs/`**: 生成画像の出力先（git対象外）。
- **`models/`**: ダウンロードされたモデルキャッシュ（git対象外）。

## Taskfile コマンド

このプロジェクトではコマンド管理に [Task](https://taskfile.dev/)
を使用しています。

```bash
# ライフサイクル
task up             # 全コンテナ起動 (Server + Client)
task up.server      # サーバーのみ起動
task down           # 全コンテナ停止
task restart        # 再起動

# 開発
task shell          # サーバーコンテナ内でシェルを開く
task logs           # ログを表示
task logs.server    # サーバーログを表示

# データ管理
task reset          # サーバーデータ領域をリセット（キャッシュは保持）
task clean-outputs  # outputs/ 内の生成画像を削除

# デプロイ
task deploy.build   # デプロイ用Dockerイメージをビルド
task deploy.run     # デプロイ用イメージをローカルで実行
```

## Python 開発

これらのコマンドはコンテナ内（`task shell`）で実行してください:

```bash
# 依存関係の同期
uv sync --all-extras

# Lintチェック
uv run ruff check src/

# テスト実行
uv run pytest tests/ -v
```
