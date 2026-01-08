# API ガイド

## 概要

テキストから画像への生成（Text-to-Image）、ジョブ管理、画像取得のためのAPIエンドポイントを提供します。
FastAPIで構築されており、デフォルトではポート8000で動作します。

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **JSON Schema**: 現在読み込まれているモデルの正確なパラメータスキーマは
  `GET /api/info` で取得できます。

## エンドポイント概要

### ジョブ管理

- `POST /api/generate`: 生成ジョブを登録します。`job_id` を返します。
- `GET /api/jobs/{job_id}`: ジョブのステータスを取得します。
- `GET /api/sse/{job_id}`: リアルタイム進捗をストリーミングします (Server-Sent
  Events)。

### 画像ギャラリー

- `GET /api/images`: 生成された画像の一覧を取得します（ページネーション対応）。
- `GET /api/images/{filename}`: フルサイズの画像を取得します。
- `GET /api/thumbs/{filename}`: サムネイルを取得します。
- `GET /api/sse/gallery`: 新しく生成された画像をリアルタイムで受信します。

## リアルタイム進捗 (SSE)

APIはServer-Sent Events (SSE) によるリアルタイム更新をサポートしています。

### ジョブ進捗 (`/api/sse/{job_id}`)

このエンドポイントに接続すると、JSON形式の更新を受け取れます:

```json
{
    "status": "processing",
    "step": 5,
    "total_steps": 20,
    "preview": "data:image/jpeg;base64,..." // JPEGエンコードされたプレビュー
}
```

ジョブが待機中の場合は、キューの位置（待ち順）も報告されます。

## リクエスト例

### SDXL / Illustrious

アニメ調のイラストや詳細なプロンプトに最適です。

```jsonc
{
    "prompt": "masterpiece, best quality, 1girl, solo, long blonde hair, blue eyes, white dress, flower garden, soft lighting",
    "negative_prompt": "worst quality, low quality, blurry, bad anatomy, bad hands, watermark",
    "width": 1024,
    "height": 1024,
    "seed": 42, // ランダムの場合は null
    "loras": [] // SDXLのみ: LoRA対応
}
```

### LoRA使用時 (SDXLのみ)

```jsonc
{
    "prompt": "masterpiece, best quality, 1girl, portrait, looking at viewer",
    "negative_prompt": "worst quality, low quality",
    "loras": [
        {
            "id": "civitai_1963644", // /api/info から取得
            "weight": 0.8,
            "trigger_weight": 0.5
        }
    ]
}
```

### Chroma

軽量なFlux派生モデル。自然言語プロンプトが機能します。**LoRAは非対応です。**

```jsonc
{
    "prompt": "a fluffy orange cat sleeping on a sunny windowsill, soft natural light",
    "width": 1024,
    "height": 1024
}
```

### Flux Dev / Schnell

高品質、または高速生成。自然言語プロンプトを使用してください。

```jsonc
{
    "prompt": "A photorealistic landscape of snow-capped mountains",
    "width": 1024,
    "height": 1024
}
```

### Z-Image Turbo

高速な8ステップモデルで、テキスト描画に優れています。

```jsonc
{
    "prompt": "A vintage coffee shop interior with warm lighting",
    "width": 1024,
    "height": 1024
}
```
