# Syntax=docker/dockerfile:1

# Base image with Python 3.13 and uv pre-installed
FROM hdae/ai-base:latest

WORKDIR /workspace/app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=3.12

# Copy dependency files
# We copy pyproject.toml and uv.lock first to leverage Docker cache
COPY server/pyproject.toml server/uv.lock README.md ./


# Install dependencies
# --frozen: Sync using lockfile and error if out of sync
# --no-install-project: Install only dependencies, not the project itself yet
RUN uv sync --frozen --no-install-project --no-dev

# Copy application source code
COPY server/src ./src
COPY server/presets ./presets

# Install the project itself
RUN uv sync --frozen --no-dev

# Copy client build for static serving
COPY client/dist /workspace/client

# Create necessary directories
RUN mkdir -p /workspace/models /workspace/outputs /workspace/.hf-cache

# Expose API port
EXPOSE 8000

# Set entrypoint using `uv run` to ensure correct environment
ENTRYPOINT ["uv", "run", "uvicorn", "txt2img.main:app", "--host", "0.0.0.0", "--port", "8000"]
