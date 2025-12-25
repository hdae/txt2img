# txt2img Server

Text-to-Image generation service using Diffusers + FastAPI.

## Setup

```bash
# Install dependencies
uv sync

# Run development server
MODEL=urn:air:sdxl:checkpoint:civitai:827184@2514310 uv run uvicorn txt2img.main:app --reload
```

## Environment Variables

| Variable              | Description                      | Default     |
| --------------------- | -------------------------------- | ----------- |
| `MODEL`               | Model AIR URN or HF repo         | Required    |
| `LORA`                | LoRA URNs/URLs (comma-separated) | -           |
| `CIVITAI_API_KEY`     | Civitai API key                  | -           |
| `HF_TOKEN`            | HuggingFace token                | -           |
| `OUTPUT_DIR`          | Output directory                 | `./outputs` |
| `OUTPUT_FORMAT`       | `png` or `webp`                  | `webp`      |
| `TRAINING_RESOLUTION` | Training resolution              | `1024`      |
| `PROMPT_PARSER`       | `legacy` or `compel`             | `legacy`    |
| `ENABLE_COMPILE`      | Enable torch.compile             | `true`      |

## API Endpoints

- `POST /api/generate` - Create generation job
- `GET /api/sse/{job_id}` - SSE progress stream
- `GET /api/images` - Image gallery
- `GET /api/images/{id}` - Get image
- `GET /api/info` - Server info
