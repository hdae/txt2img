# API Guide

## Overview

The API provides endpoints for text-to-image generation, job management, and
image retrieval. It is built with FastAPI and runs on port 8000 by default.

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **JSON Schema**: Get the precise parameter schema for the currently loaded
  model at `GET /api/info`.

## Endpoints Summary

### Job Management

- `POST /api/generate`: Submit a generation job. Returns `job_id`.
- `GET /api/jobs/{job_id}`: Get job status.
- `GET /api/sse/{job_id}`: Stream real-time progress (Server-Sent Events).

### Image Gallery

- `GET /api/images`: List generated images (paginated).
- `GET /api/images/{filename}`: Get full-size image.
- `GET /api/thumbs/{filename}`: Get thumbnail.
- `GET /api/sse/gallery`: Stream new images as they are generated.

## Real-time Progress (SSE)

The API supports Server-Sent Events (SSE) for real-time updates.

### Job Progress (`/api/sse/{job_id}`)

Connect to this endpoint to receive JSON updates:

```json
{
    "status": "processing",
    "step": 5,
    "total_steps": 20,
    "preview": "data:image/jpeg;base64,..." // JPEG encoded preview
}
```

It also reports queue position if the job is waiting.

## Request Examples

### SDXL / Illustrious

Best for anime-style illustrations with detailed prompts.

```jsonc
{
    "prompt": "masterpiece, best quality, 1girl, solo, long blonde hair, blue eyes, white dress, flower garden, soft lighting",
    "negative_prompt": "worst quality, low quality, blurry, bad anatomy, bad hands, watermark",
    "width": 1024,
    "height": 1024,
    "seed": 42, // null for random
    "loras": [] // SDXL only: LoRA support
}
```

### With LoRA (SDXL only)

```jsonc
{
    "prompt": "masterpiece, best quality, 1girl, portrait, looking at viewer",
    "negative_prompt": "worst quality, low quality",
    "loras": [
        {
            "id": "civitai_1963644", // Get from /api/info
            "weight": 0.8,
            "trigger_weight": 0.5
        }
    ]
}
```

### Chroma

Lightweight Flux variant. Works well with natural language prompts. **LoRA not
supported.**

```jsonc
{
    "prompt": "a fluffy orange cat sleeping on a sunny windowsill, soft natural light",
    "width": 1024,
    "height": 1024
}
```

### Flux Dev / Schnell

High quality or high speed. Use natural language prompts.

```jsonc
{
    "prompt": "A photorealistic landscape of snow-capped mountains",
    "width": 1024,
    "height": 1024
}
```

### Z-Image Turbo

Fast 8-step model with excellent text rendering.

```jsonc
{
    "prompt": "A vintage coffee shop interior with warm lighting",
    "width": 1024,
    "height": 1024
}
```
