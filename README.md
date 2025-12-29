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

| Variable              | Description                | Required        |
| --------------------- | -------------------------- | --------------- |
| `MODEL`               | Civitai AIR URN or HF repo | ✓               |
| `CIVITAI_API_KEY`     | Civitai API key            | For some models |
| `TRAINING_RESOLUTION` | `1024` or `1024x1024`      |                 |
| `PROMPT_PARSER`       | `lpw` or `compel`          |                 |

See [.env.example](.env.example) for all options.

## Task Commands

```bash
task up           # Start all containers
task up.server    # Start only server
task down         # Stop containers
task shell        # Open shell in server
task logs         # View logs
task reset        # Reset server data
```

## Endpoints

- `http://localhost:8000` - API server
- `http://localhost:3000` - Web client

## Project Structure

```
txt2img/
├── docker-compose.yml
├── Taskfile.yml
├── start.sh
├── .env.example
├── server/           # Python/FastAPI backend
│   ├── pyproject.toml
│   └── src/txt2img/
└── client/           # React/Vite frontend
    ├── package.json
    └── src/
```

## LoRA Configuration

In preset files, LoRAs are specified as an array of objects:

```json
{
    "loras": [
        { "ref": "urn:air:sdxl:lora:civitai:1377820@1963644" },
        {
            "ref": "https://example.com/lora.safetensors",
            "triggers": ["mytrigger", "custom_tag"]
        }
    ]
}
```

- `ref` (required): Civitai AIR URN or direct URL
- `triggers` (optional): Manual trigger words. If omitted, auto-detected from
  Civitai API

### API Usage

```json
{
    "loras": [
        { "id": "civitai_1963644", "weight": 1.0, "trigger_weight": 0.5 }
    ]
}
```

- `weight`: LoRA model weight (overall influence)
- `trigger_weight`: Trigger embedding weight (target influence), set to 0 to
  disable

## Known Issues

### LoRA + Quantization Causes Black Images

**Problem**: Using TorchAO quantization (`int8wo`, `fp8wo`) together with LoRAs
produces completely black images.

**Cause**: TorchAO's `quantize_()` replaces Linear layers with quantized
tensors, which conflicts with PEFT's LoRA adapter injection mechanism.

**Workaround**: The server automatically disables quantization when LoRAs are
loaded. If you need both:

- Use quantization without LoRAs, OR
- Use LoRAs without quantization

This is a known upstream compatibility issue between TorchAO and PEFT.
