"""FastAPI application main entry point."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from txt2img.api.router import router
from txt2img.config import get_settings
from txt2img.core.job_queue import job_queue
from txt2img.core.pipeline import pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
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
                saved = await pipeline.generate(
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

    settings = get_settings()

    # Create output directories
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)

    # Load model configuration
    logger.info("Loading model configuration...")
    load_model_config()

    # Load model
    logger.info("Loading model...")
    await pipeline.load_model()
    logger.info("Model loaded successfully")

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
