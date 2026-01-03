# Syntax=docker/dockerfile:1

# Stage 1: Generate requirements.txt using uv
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app
COPY server/pyproject.toml server/uv.lock ./
# Generate requirements.txt excluding development dependencies
RUN uv export --frozen --no-dev --no-emit-project --output-file requirements.txt

# Stage 2: Final image using official PyTorch base
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

WORKDIR /workspace/app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy requirements from builder stage
COPY --from=builder /app/requirements.txt .

# Install dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and install the project
COPY server/src ./src
COPY server/pyproject.toml README.md ./
RUN pip install --no-deps -e .

COPY server/presets ./presets

# Copy client build for static serving
COPY client/dist /workspace/client

# Create necessary directories
RUN mkdir -p /workspace/models /workspace/outputs /workspace/.hf-cache

# Expose API port
EXPOSE 8000

# Set entrypoint (using uvicorn directly as uv is not in this image/needed)
ENTRYPOINT ["uvicorn", "txt2img.main:app", "--host", "0.0.0.0", "--port", "8000"]
