#!/bin/bash
set -euo pipefail

# ========================================
# txt2img Server Start Script
# ========================================
# Entrypoint has already:
# - Adjusted UID/GID
# - Installed Python via uv
# - Created and activated virtual environment
#
# Environment variables from entrypoint:
# - APP_DIR=/workspace
# - VIRTUAL_ENV=/workspace/.venv
# - PYTHON_VERSION
# ========================================

APP_DIR="${APP_DIR:-/workspace}"
SERVER_DIR="$APP_DIR/app"

echo "====================================="
echo "txt2img Server Startup"
echo "====================================="

# ========================================
# Install Dependencies
# ========================================
if [ -f "$SERVER_DIR/pyproject.toml" ]; then
    echo "Installing server dependencies..."
    cd "$SERVER_DIR"
    uv sync
    echo "✓ Dependencies installed"
else
    echo "ERROR: pyproject.toml not found in $SERVER_DIR"
    exit 1
fi

# ========================================
# Create Output Directory
# ========================================
mkdir -p "$APP_DIR/outputs"
export OUTPUT_DIR="$APP_DIR/outputs"

# ========================================
# Start Server
# ========================================
echo ""
echo "Starting txt2img server..."
echo "Model: ${MODEL:-not set}"
echo "Training Resolution: ${TRAINING_RESOLUTION:-1024}"
echo "Prompt Parser: ${PROMPT_PARSER:-lpw}"
echo ""

cd "$SERVER_DIR"
exec uv run uvicorn txt2img.main:app \
    --host 0.0.0.0 \
    --port 8000
