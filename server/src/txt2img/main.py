"""FastAPI application main entry point."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from txt2img.api.router import router
from txt2img.config import get_settings
from txt2img.core.job_queue import job_queue
from txt2img.pipelines import get_pipeline

# ANSI color codes for terminal
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


# Custom formatter with emoji prefixes and time-only format
class EmojiFormatter(logging.Formatter):
    """Log formatter with emoji prefixes based on module and content."""

    # Default emojis by log level
    LEVEL_EMOJI = {
        logging.DEBUG: "🔍",
        logging.INFO: "ℹ️ ",
        logging.WARNING: "⚠️ ",
        logging.ERROR: "❌",
        logging.CRITICAL: "🔥",
    }

    # Module-specific emojis (overrides level emoji for INFO)
    MODULE_EMOJI = {
        "txt2img.providers.civitai": "📥",  # Download
        "txt2img.providers.huggingface": "📥",  # Download
        "txt2img.core.lora_manager": "🎨",  # LoRA
        "txt2img.pipelines.sdxl": "🖼️ ",  # SDXL Generation
        "txt2img.pipelines.flux_dev": "🖼️ ",  # Flux Generation
        "txt2img.pipelines.flux_schnell": "🖼️ ",  # Flux Generation
        "txt2img.pipelines.chroma": "🖼️ ",  # Chroma Generation
        "txt2img.pipelines.zimage": "🖼️ ",  # Z-Image Generation
        "txt2img.core.job_queue": "📋",  # Jobs
        "txt2img.core.image_processor": "💾",  # Save
        "txt2img.api.router": "🌐",  # API
        "httpx": "🌍",  # HTTP requests
    }

    def format(self, record):
        # Use level emoji for warnings/errors, module emoji for info
        if record.levelno >= logging.WARNING:
            emoji = self.LEVEL_EMOJI.get(record.levelno, "")
        else:
            emoji = self.MODULE_EMOJI.get(record.name, self.LEVEL_EMOJI.get(record.levelno, ""))
        record.emoji = emoji
        return super().format(record)


# Setup logging with time-only format and emojis
handler = logging.StreamHandler()
handler.setFormatter(EmojiFormatter("%(asctime)s %(emoji)s %(message)s", datefmt="%H:%M:%S"))
logging.root.handlers = [handler]
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(__name__)


async def job_worker():
    """Background worker that processes jobs from the queue."""
    logger.info("Job worker started")

    while True:
        try:
            # Get next job from queue
            job = await job_queue.get_next_job()
            logger.info(f"Processing job {job.id}")

            # Mark as running
            await job_queue.mark_job_running(job.id)

            try:
                # Progress callback for SSE updates
                # Bind job_id explicitly to avoid closure issue
                job_id = job.id

                async def progress_callback(
                    step: int, preview: str | None, *, bound_job_id: str = job_id
                ):
                    await job_queue.update_job_progress(bound_job_id, step, preview)

                # Generate image
                saved = await get_pipeline().generate(
                    job.params,
                    progress_callback=lambda s, p: asyncio.create_task(progress_callback(s, p)),
                )

                # Mark as completed
                from txt2img.core.job_queue import JobResult

                result = JobResult(
                    image_id=saved.id,
                    image_url=f"/api/images/{saved.id}",
                    thumbnail_url=f"/api/images/{saved.id}?thumbnail=true",
                    metadata=saved.metadata.to_dict(),
                )
                await job_queue.mark_job_completed(job.id, result)
                logger.info(f"Job {job.id} completed: {saved.id}")

            except Exception as e:
                logger.exception(f"Job {job.id} failed")
                await job_queue.mark_job_failed(job.id, str(e))

        except asyncio.CancelledError:
            logger.info("Job worker cancelled")
            break
        except Exception:
            logger.exception("Job worker error")
            await asyncio.sleep(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    from txt2img.config import load_model_config
    from txt2img.core.lora_manager import lora_manager

    settings = get_settings()

    # Create output directories
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)

    # Load model configuration
    logger.info("Loading model configuration...")
    load_model_config()

    # Load model
    logger.info("Loading model...")
    await get_pipeline().load_model()
    logger.info("Model loaded successfully")

    # Initialize LoRA manager
    logger.info("Initializing LoRAs...")
    await lora_manager.initialize()
    logger.info(f"LoRAs initialized: {len(lora_manager.loras)} registered")

    # Start job worker
    worker_task = asyncio.create_task(job_worker())

    yield

    # Cleanup
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass


# Create FastAPI app
app = FastAPI(
    title="txt2img Service",
    description="Text-to-Image generation service using Diffusers",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Log validation errors with request body for debugging."""
    body = await request.body()
    body_str = body.decode("utf-8", errors="replace")[:500]  # Truncate long bodies
    errors = exc.errors()

    # Log with red color for visibility
    logger.error(f"{RED}[VALIDATION ERROR] {request.method} {request.url.path}{RESET}")
    logger.error(f"{RED}  Body: {body_str}{RESET}")
    for err in errors:
        loc = " -> ".join(str(loc_part) for loc_part in err.get("loc", []))
        logger.error(f"{RED}  • {loc}: {err.get('msg')}{RESET}")

    return JSONResponse(
        status_code=422,
        content={"detail": errors},
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


def main():
    """Run the application."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "txt2img.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
