# txt2img

Text-to-Image generation service using Diffusers + FastAPI.

[🇯🇵 日本語 (Japanese)](README.ja.md) | [🤖 for Agents](AGENTS.md)

## Documentation

- **[API Guide](docs/API_GUIDE.md)**: Endpoints usage and SSE details.
- **[Models & Config](docs/MODELS.md)**: Supported models and parameters.
- **[Development](docs/DEVELOPMENT.md)**: Project structure and dev commands.

## Quick Start (Local)

1. **Setup Environment**:
   ```bash
   cp .env.example .env
   vim .env  # Set CONFIG (e.g., config/chroma/flash)
   ```

2. **Run**:
   ```bash
   task up
   # or: docker compose up
   ```

3. **Access**:
   - Web UI: http://localhost:5173
   - API Docs: http://localhost:8000/docs

## Cloud / RunPod

For cloud deployments (e.g., RunPod), use the pre-built Docker image.

- **Image**: `hdae/txt2img:latest`
- **Environment Variables**:
  - `CONFIG`: Model preset path (e.g., `sdxl/illustrious`).
  - `CIVITAI_API_KEY`: Required for downloading models.
  - `HF_TOKEN`: Required for HuggingFace gated models.
  - `VRAM_PROFILE`: `full` (recommended for 24GB+), `balanced`, or `lowvram`.

### RunPod Template Example

```bash
docker run --rm -p 8000:8000 --gpus all \
  -e CONFIG=sdxl/illustrious \
  -e CIVITAI_API_KEY=your_key \
  hdae/txt2img:latest
```
