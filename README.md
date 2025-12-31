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

| Variable          | Description             | Default |
| ----------------- | ----------------------- | ------- |
| `CONFIG`          | Preset path (see below) | -       |
| `VRAM_PROFILE`    | VRAM optimization mode  | `full`  |
| `OUTPUT_FORMAT`   | Output image format     | `png`   |
| `CIVITAI_API_KEY` | Civitai API key         | -       |
| `HF_TOKEN`        | HuggingFace token       | -       |

### Recommended Presets

```bash
# .env examples
CONFIG=sdxl/illustrious     # SDXL Illustrious derivative
CONFIG=chroma/flash         # Fast 4-step generation
CONFIG=zimage/turbo         # High quality 8-step
```

See [.env.example](.env.example) for all options.

### VRAM Profiles

| Profile    | VRAM Target | Optimization                           |
| ---------- | ----------- | -------------------------------------- |
| `full`     | 24GB+       | No offloading (maximum speed)          |
| `balanced` | 12-16GB     | Model CPU offload + VAE tiling         |
| `lowvram`  | 8GB         | Group offload (streaming) + VAE tiling |

### Output Format

| Format | Description                 |
| ------ | --------------------------- |
| `png`  | Lossless, larger files      |
| `webp` | Lossy (quality=95), smaller |

## Supported Models

| Type           | Description                                     | Example Preset     |
| -------------- | ----------------------------------------------- | ------------------ |
| `sdxl`         | SDXL and derivatives (Illustrious, etc.)        | `sdxl/illustrious` |
| `flux`         | Flux.1 [dev] - guidance-distilled (~50 steps)   | `flux/dev`         |
| `flux_schnell` | Flux.1 [schnell] - timestep-distilled (4 steps) | `flux/schnell`     |
| `chroma`       | Chroma - 8.9B lightweight Flux variant          | `chroma/flash`     |
| `zimage`       | Z-Image Turbo - 6B fast model (8 steps)         | `zimage/turbo`     |

### Model Reference Formats

- **Civitai AIR**: `urn:air:sdxl:checkpoint:civitai:827184@2514310`
- **HuggingFace**: `black-forest-labs/FLUX.1-dev`
- **URL**: `https://example.com/model.safetensors`
- **Local**: `file:///path/to/model.safetensors`

## API Usage

- **Swagger UI**: http://localhost:8000/docs
- **Request Examples**: See [GUIDE.md](GUIDE.md)

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
  "default_sampler": "euler_a"
}
```

### LoRA Options

| Property         | Description                                           | Default |
| ---------------- | ----------------------------------------------------- | ------- |
| `ref`            | Civitai AIR URN or URL (required)                     | -       |
| `triggers`       | Trigger words (auto-detected from Civitai if omitted) | `null`  |
| `weight`         | LoRA model weight                                     | `1.0`   |
| `trigger_weight` | Trigger embedding weight                              | `0.5`   |

## Task Commands

```bash
task up           # Start all containers
task up.server    # Start only server
task down         # Stop containers
task shell        # Open shell in server
task logs         # View logs
task reset        # Reset server data
```
