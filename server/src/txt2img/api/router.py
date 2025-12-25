"""API router."""

import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from starlette.requests import Request

from txt2img.api.schemas import (
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    ImageInfo,
    ImageListResponse,
    ServerInfo,
)
from txt2img.api.sse import sse_endpoint
from txt2img.config import get_settings
from txt2img.core.image_processor import list_images
from txt2img.core.job_queue import GenerationParams, job_queue
from txt2img.core.pipeline import pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


@router.post(
    "/generate",
    response_model=GenerateResponse,
    responses={500: {"model": ErrorResponse}},
)
async def generate_image(request: GenerateRequest) -> GenerateResponse:
    """Create image generation job.

    Returns job_id and SSE URL for progress monitoring.
    """
    params = GenerationParams(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        steps=request.steps,
        cfg_scale=request.cfg_scale,
        seed=request.seed,
        sampler=request.sampler,
    )

    job = job_queue.create_job(params)

    return GenerateResponse(
        job_id=job.id,
        sse_url=f"/api/sse/{job.id}",
    )


@router.get("/sse/{job_id}")
async def get_job_sse(request: Request, job_id: str):
    """SSE endpoint for job progress updates."""
    return await sse_endpoint(request, job_id)


@router.get("/images", response_model=ImageListResponse)
async def get_images(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> ImageListResponse:
    """Get list of generated images (gallery)."""
    images = list_images(limit=limit, offset=offset)

    return ImageListResponse(
        images=[ImageInfo(**img) for img in images],
        total=len(images),  # TODO: Get actual total count
        offset=offset,
        limit=limit,
    )


@router.get("/images/{image_id}.{ext}")
async def get_image_with_ext(image_id: str, ext: str) -> FileResponse:
    """Get full-size image file with extension.

    Args:
        image_id: Image ID
        ext: File extension (png, webp)
    """
    settings = get_settings()

    if ext not in ("png", "webp"):
        raise HTTPException(status_code=400, detail="Invalid extension")

    file_path = settings.output_dir / "images" / f"{image_id}.{ext}"
    media_type = f"image/{ext}"

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=file_path.name,
    )


@router.get("/thumbs/{image_id}.webp")
async def get_thumbnail(image_id: str) -> FileResponse:
    """Get thumbnail image (always WebP).

    Args:
        image_id: Image ID
    """
    settings = get_settings()

    file_path = settings.output_dir / "thumbs" / f"{image_id}.webp"

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    return FileResponse(
        path=file_path,
        media_type="image/webp",
        filename=file_path.name,
    )


@router.get("/info", response_model=ServerInfo)
async def get_server_info() -> ServerInfo:
    """Get server information including model name, training resolution, and prompt parser."""
    from txt2img.config import get_model_config

    config = get_model_config()

    return ServerInfo(
        model_name=pipeline.model_name or "Not loaded",
        training_resolution=str(config.training_resolution),
        prompt_parser=config.prompt_parser.value,
    )
