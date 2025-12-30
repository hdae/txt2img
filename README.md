# txt2img

Text-to-Image generation service using Diffusers + FastAPI.

## Quick Start

```bash
# 1. Copy environment file
cp .env.example .env

# 2. Edit .env with your settings
vim .env

# 3. Start containers
task up
# or: docker compose up
```

## Requirements

- Docker + Docker Compose
- NVIDIA GPU with CUDA support
- [Task](https://taskfile.dev/) (optional)
- `hdae/ai-base` Docker image

## Configuration

Edit `.env` file:

| Variable          | Description                | Required        |
| ----------------- | -------------------------- | --------------- |
| `CONFIG`          | Preset path or JSON config | ✓               |
| `CIVITAI_API_KEY` | Civitai API key            | For some models |
| `HF_TOKEN`        | HuggingFace token          | For some models |

See [.env.example](.env.example) for all options.

## Supported Models

| Type           | Description                                     | Example Preset     |
| -------------- | ----------------------------------------------- | ------------------ |
| `sdxl`         | SDXL and derivatives (Illustrious, etc.)        | `sdxl/illustrious` |
| `flux`         | Flux.1 [dev] - guidance-distilled (~50 steps)   | `flux/dev`         |
| `flux_schnell` | Flux.1 [schnell] - timestep-distilled (4 steps) | `flux/schnell`     |
| `chroma`       | Chroma - 8.9B lightweight Flux variant          | `chroma/flash`     |
| `zimage`       | Z-Image Turbo - 6B fast model (8 steps)         | `zimage/turbo`     |

## Preset Configuration

Presets are JSON files in `server/presets/`. See
[config.schema.json](server/presets/config.schema.json) for full schema.

### SDXL Example

```json
{
    "$schema": "../config.schema.json",
    "type": "sdxl",
    "model": "urn:air:sdxl:checkpoint:civitai:827184@2514310",
    "vae": null,
    "loras": [
        { "ref": "urn:air:sdxl:lora:civitai:1377820@1963644" }
    ],
    "training_resolution": 1024,
    "default_steps": 20,
    "default_cfg": 7.0,
    "default_sampler": "euler_a",
    "output_format": "webp"
}
```

### Model Reference Formats

- **Civitai AIR**: `urn:air:sdxl:checkpoint:civitai:827184@2514310`
- **HuggingFace**: `black-forest-labs/FLUX.1-dev`
- **URL**: `https://example.com/model.safetensors`
- **Local**: `file:///path/to/model.safetensors`

## LoRA Configuration

LoRAs are specified as an array of objects in preset files:

```json
{
    "loras": [
        { "ref": "urn:air:sdxl:lora:civitai:1377820@1963644" },
        {
            "ref": "https://example.com/lora.safetensors",
            "triggers": ["mytrigger"],
            "weight": 1.0,
            "trigger_weight": 0.5
        }
    ]
}
```

| Property         | Description                                           | Default |
| ---------------- | ----------------------------------------------------- | ------- |
| `ref`            | Civitai AIR URN or URL (required)                     | -       |
| `triggers`       | Trigger words (auto-detected from Civitai if omitted) | `null`  |
| `weight`         | LoRA model weight                                     | `1.0`   |
| `trigger_weight` | Trigger embedding weight                              | `0.5`   |

### API Usage

```json
{
    "loras": [
        { "id": "civitai_1963644", "weight": 1.0, "trigger_weight": 0.5 }
    ]
}
```

## Task Commands

```bash
task up           # Start all containers
task up.server    # Start only server
task down         # Stop containers
task shell        # Open shell in server
task logs         # View logs
task reset        # Reset server data
```

## API Endpoints

| Endpoint                 | Method | Description                     |
| ------------------------ | ------ | ------------------------------- |
| `/health`                | GET    | Health check                    |
| `/api/info`              | GET    | Server info (model, LoRAs, etc) |
| `/api/generate`          | POST   | Create generation job           |
| `/api/sse/{job_id}`      | GET    | SSE progress updates            |
| `/api/images`            | GET    | List generated images           |
| `/api/images/{id}.{ext}` | GET    | Get full image                  |
| `/api/thumbs/{id}.webp`  | GET    | Get thumbnail                   |

## Project Structure

```
txt2img/
├── docker-compose.yml
├── Taskfile.yml
├── start.sh
├── .env.example
├── server/                   # Python/FastAPI backend
│   ├── pyproject.toml
│   ├── presets/              # Model presets
│   │   ├── config.schema.json
│   │   ├── sdxl/
│   │   ├── flux/
│   │   ├── chroma/
│   │   └── zimage/
│   └── src/txt2img/
│       ├── main.py           # FastAPI entrypoint
│       ├── config.py         # Configuration
│       ├── api/              # API layer
│       ├── core/             # Core modules
│       │   ├── job_queue.py
│       │   ├── lora_manager.py
│       │   ├── prompt_parser.py
│       │   └── image_processor.py
│       ├── pipelines/        # Model pipelines
│       │   ├── sdxl.py
│       │   ├── flux_dev.py
│       │   ├── flux_schnell.py
│       │   ├── chroma.py
│       │   └── zimage.py
│       └── providers/        # Model providers
│           ├── civitai.py
│           └── huggingface.py
└── outputs/                  # Generated images
```
