# txt2img

Diffusers + FastAPI を使用したテキスト画像生成サービス。

[🇺🇸 English](README.md) | [🤖 for Agents](AGENTS.md)

## ドキュメント

- **[API ガイド](docs/API_GUIDE.ja.md)**: エンドポイントの使い方やSSEの詳細。
- **[モデルと設定](docs/MODELS.ja.md)**: 対応モデル、パラメータ、VRAM設定。
- **[開発ガイド](docs/DEVELOPMENT.ja.md)**: プロジェクト構成と開発コマンド。

## クイックスタート (ローカル)

1. **環境構築**:
   ```bash
   cp .env.example .env
   vim .env  # CONFIG を設定 (例: config/chroma/flash)
   ```

2. **起動**:
   ```bash
   task up
   # または: docker compose up
   ```

3. **アクセス**:
   - Web UI: http://localhost:5173
   - API Docs: http://localhost:8000/docs

## クラウド / RunPod

RunPodなどのクラウドGPU環境では、ビルド済みのDockerイメージを使用してください。

- **イメージ**: `hdae/txt2img:latest`
- **環境変数**:
  - `CONFIG`: モデルプリセットパス (例: `sdxl/illustrious`).
  - `CIVITAI_API_KEY`: モデルのダウンロードに必要。
  - `HF_TOKEN`: HuggingFaceの認証が必要なモデルに必要。
  - `VRAM_PROFILE`: `full` (24GB+推奨), `balanced`, または `lowvram`.

### RunPod テンプレート実行例

```bash
docker run --rm -p 8000:8000 --gpus all \
  -e CONFIG=sdxl/illustrious \
  -e CIVITAI_API_KEY=your_key \
  hdae/txt2img:latest
```
