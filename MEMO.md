# txt2img 機能メモ

## コア機能

### モデル管理

- **1モデル固定起動**: `CONFIG`環境変数でプリセット指定
- **自動ダウンロード**: Civitai AIR URN、HuggingFace、URLからモデルを自動取得
- **LoRAサポート**: SDXL向け、トリガーワード自動取得・適用

### VRAM最適化

- **full**: GPU常駐（高速、VRAM大）
- **balanced**: CPU offload + VAE tiling
- **lowvram**: Group offload（16GB VRAM対応）

### 画像生成

- **非同期生成**: POST即座にjob_id返却
- **SSE進捗**: `/api/sse/{job_id}`でリアルタイム進捗
  - キュー位置表示（あと何件で自分の番か）
  - ステップ進捗
- **出力形式**: PNG（ロスレス）/ WebP（軽量）選択可
- **サムネイル**: 自動生成（1/4サイズ WebP）
- **メタデータ**: PNG info / EXIF形式で生成パラメータ埋め込み

### API

- **Swagger UI**: `/docs`で仕様確認
- **パラメータスキーマ**: `/api/info`でJSON Schema取得
- **ギャラリー**: `/api/images`で生成画像一覧
- **ギャラリーSSE**: `/api/sse/gallery`で新着画像をリアルタイム受信

## 対応モデル

| タイプ       | 特徴                           | LoRA |
| ------------ | ------------------------------ | ---- |
| SDXL         | アニメ向け、タグ形式プロンプト | ✅   |
| Chroma       | 軽量Flux派生、自然言語         | ❌   |
| Flux Dev     | 高品質、50ステップ             | ❌   |
| Flux Schnell | 高速、4ステップ                | ❌   |
| Z-Image      | テキスト描画に強い、8ステップ  | ❌   |

## プロンプト

- **SDXL**: タグ形式 (`masterpiece, 1girl, ...`)
- **その他**: 自然言語 (`A photo of...`)
- **A1111構文**: `(emphasis)`, `[de-emphasis]`, `(word:1.5)`
- **BREAK**: 長文分割キーワード

## 固定パラメータ

APIから`steps`, `cfg_scale`, `sampler`は削除済み。
各モデルで最適値を内部で使用：

| モデル       | Steps | CFG | Sampler |
| ------------ | ----- | --- | ------- |
| SDXL         | 20    | 7.0 | euler_a |
| Chroma       | 4     | 4.0 | -       |
| Flux Dev     | 50    | 3.5 | -       |
| Flux Schnell | 4     | 0.0 | -       |
| Z-Image      | 8     | 0.0 | -       |

## デプロイ

```bash
task up              # 起動
task down            # 停止
task shell           # コンテナ内シェル
task logs            # ログ確認
```

## 開発

```bash
# Dockerコンテナ内で
uv sync --all-extras
uv run ruff check src/
uv run pytest tests/ -v
```
