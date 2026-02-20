#!/bin/bash
set -euo pipefail

# ========================================
# txt2img startup script for hdae/ai-base
# ========================================
# entrypoint.sh has already:
# - adjusted UID/GID
# - installed Python via uv
# - created/activated /workspace/.venv

APP_DIR="${APP_DIR:-/workspace}"
PROJECT_DIR="${APP_DIR}/app"
SERVER_DIR="${PROJECT_DIR}/server"

clone_or_update() {
    local repo_url="$1"
    local target_dir="$2"
    local ref="${3:-main}"

    if [ -d "$target_dir/.git" ]; then
        echo "Updating $target_dir to $ref..."
        cd "$target_dir"
        git fetch origin --quiet
        git checkout "$ref" --quiet
        git reset --hard "origin/$ref" 2>/dev/null || git reset --hard "$ref"
        cd - >/dev/null
    else
        echo "Cloning $repo_url to $target_dir..."
        git clone --quiet "$repo_url" "$target_dir"
        cd "$target_dir"
        git checkout "$ref" --quiet
        cd - >/dev/null
    fi
}

# Optional bootstrap mode: clone source when /workspace/app is not mounted.
if [ ! -d "$PROJECT_DIR/.git" ] && [ -n "${APP_GIT_URL:-}" ]; then
    clone_or_update "${APP_GIT_URL}" "$PROJECT_DIR" "${APP_GIT_REF:-main}"
fi

if [ ! -d "$SERVER_DIR" ]; then
    echo "ERROR: server directory not found at $SERVER_DIR"
    echo "Mount this repo to /workspace/app or set APP_GIT_URL."
    exit 1
fi

export PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_DIR}}"

echo "Syncing Python dependencies..."
if [ -f "${SERVER_DIR}/uv.lock" ]; then
    uv sync --project "$SERVER_DIR" --frozen
else
    uv sync --project "$SERVER_DIR"
fi

if [ $# -gt 0 ]; then
    echo "Executing custom command: $*"
    exec "$@"
fi

echo "Starting txt2img API server..."
exec uv run --project "$SERVER_DIR" python -m txt2img.main
