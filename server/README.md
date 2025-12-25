# txt2img Server

Text-to-Image generation service using Diffusers + FastAPI.

## Setup

```bash
# Install dependencies
uv sync

# Run development server (with CONFIG)
CONFIG=sdxl uv run uvicorn txt2img.main:app --reload
```

## Configuration

### Environment Variables

| Variable          | Description                      | Default                  |
| ----------------- | -------------------------------- | ------------------------ |
| `CONFIG`          | JSON config, URL, or preset name | Required                 |
| `CIVITAI_API_KEY` | Civitai API key                  | -                        |
| `HF_TOKEN`        | HuggingFace token                | -                        |
| `OUTPUT_DIR`      | Output directory                 | `./outputs`              |
| `MODEL_CACHE_DIR` | Model cache directory            | `/workspace/models`      |
| `PRESETS_DIR`     | Presets directory                | `/workspace/app/presets` |

### JSON Config

Create a JSON config file or pass inline:

```json
{
    "type": "sdxl",
    "model": "urn:air:sdxl:checkpoint:civitai:827184@2514310",
    "vae": null,
    "loras": [],
    "quantization": "none",
    "training_resolution": 1024,
    "default_steps": 20,
    "default_cfg": 7.0,
    "default_sampler": "euler",
    "output_format": "webp",
    "prompt_parser": "lpw"
}
```

## API Endpoints

- `POST /api/generate` - Create generation job
- `GET /api/sse/{job_id}` - SSE progress stream
- `GET /api/images` - Image gallery
- `GET /api/images/{id}.{ext}` - Get image (with extension)
- `GET /api/thumbs/{id}.webp` - Get thumbnail
- `GET /api/info` - Server info (model, resolution, prompt_parser)
