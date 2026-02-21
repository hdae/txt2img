FROM hdae/ai-base:latest

ENV APP_DIR=/workspace \
    PROJECT_ROOT=/workspace/app \
    UV_CACHE_DIR=/workspace/.uv-cache \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /workspace/app

# Copy only runtime assets. client/dist is expected to be pre-built.
COPY --chown=app:app server/pyproject.toml server/uv.lock ./server/
COPY --chown=app:app server/src ./server/src
COPY --chown=app:app server/presets ./server/presets
COPY --chown=app:app server/vendor ./server/vendor
COPY --chown=app:app client/dist ./client/dist

RUN uv sync --project /workspace/app/server --frozen

EXPOSE 8000

CMD ["uv", "run", "--project", "/workspace/app/server", "--no-sync", "python", "-m", "txt2img.main"]
