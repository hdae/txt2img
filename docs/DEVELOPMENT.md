# Development Guide

## Project Structure

- **`server/`**: Python FastAPI backend.
  - **`src/`**: Application source code.
  - **`presets/`**: Model configuration JSONs.
- **`client/`**: React/Vite frontend.
- **`outputs/`**: Generated images (git-ignored).
- **`models/`**: Downloaded model cache (git-ignored).

## Taskfile Commands

This project uses [Task](https://taskfile.dev/) for command management.

```bash
# Lifecycle
task up             # Start full stack (Server + Client + Redis/etc)
task up.server      # Start only server
task down           # Stop all containers
task restart        # Restart containers

# Development
task shell          # Open shell inside server container
task logs           # View logs
task logs.server    # View server logs

# Data Management
task reset          # Reset server data volume (preserves caches)
task clean-outputs  # Delete all generated images in outputs/

# Deployment
task deploy.build   # Build Docker image for deployment
task deploy.run     # Run deployment image locally
```

## Python Development

Run these commands inside the container (`task shell`):

```bash
# Sync dependencies
uv sync --all-extras

# Linting
uv run ruff check src/

# Testing
uv run pytest tests/ -v
```
