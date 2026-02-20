"""API router."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from starlette.requests import Request

from txt2img.api.schemas import (
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    ImageInfo,
    ImageListResponse,
    LoraInfo,
    ServerInfo,
)
from txt2img.api.sse import gallery_sse_endpoint, sse_endpoint
from txt2img.config import get_output_format, get_settings
from txt2img.core.image_processor import list_images
from txt2img.core.job_queue import GenerationParams, job_queue
from txt2img.core.lora_manager import lora_manager
from txt2img.pipelines import get_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


def _resolve_cfg_scale(request_cfg_scale: float | None, schema: dict[str, Any]) -> float:
    """Resolve cfg_scale using pipeline schema defaults and range."""
    defaults = schema.get("defaults", {})
    properties = schema.get("properties", {})
    cfg_schema = properties.get("cfg_scale")

    default_cfg = defaults.get("cfg_scale", 7.0)
    if isinstance(cfg_schema, dict):
        default_cfg = cfg_schema.get("default", default_cfg)

    value = request_cfg_scale if request_cfg_scale is not None else default_cfg
    try:
        resolved = float(value)
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=422, detail="Invalid cfg_scale value") from e

    if isinstance(cfg_schema, dict):
        minimum = cfg_schema.get("minimum")
        maximum = cfg_schema.get("maximum")
        if isinstance(minimum, (int, float)) and resolved < float(minimum):
            raise HTTPException(
                status_code=422,
                detail=f"cfg_scale must be >= {minimum}",
            )
        if isinstance(maximum, (int, float)) and resolved > float(maximum):
            raise HTTPException(
                status_code=422,
                detail=f"cfg_scale must be <= {maximum}",
            )

    return resolved


def _resolve_sampler(request_sampler: str | None, schema: dict[str, Any]) -> str:
    """Resolve sampler using pipeline schema defaults and enum."""
    defaults = schema.get("defaults", {})
    properties = schema.get("properties", {})
    sampler_schema = properties.get("sampler")

    default_sampler = defaults.get("sampler", "euler_a")
    if isinstance(sampler_schema, dict):
        default_sampler = sampler_schema.get("default", default_sampler)

    resolved = request_sampler if request_sampler is not None else default_sampler
    if not isinstance(resolved, str) or not resolved:
        raise HTTPException(status_code=422, detail="Invalid sampler value")

    if isinstance(sampler_schema, dict):
        enum_values = sampler_schema.get("enum")
        if isinstance(enum_values, list) and enum_values and resolved not in enum_values:
            raise HTTPException(
                status_code=422,
                detail=f"sampler must be one of: {', '.join(enum_values)}",
            )

    return resolved


@router.post(
    "/generate",
    response_model=GenerateResponse,
    responses={500: {"model": ErrorResponse}},
)
async def generate_image(request: GenerateRequest) -> GenerateResponse:
    """Create image generation job.

    Returns job_id and SSE URL for progress monitoring.
    """
    # Debug logging
    logger.info(f"Generate request: prompt={request.prompt[:50] if request.prompt else 'empty'}")
    logger.info(f"Generate request loras: {request.loras}")

    # Convert loras to list of dicts
    loras_list = None
    if request.loras:
        loras_list = [
            {"id": lora.id, "weight": lora.weight, "trigger_weight": lora.trigger_weight}
            for lora in request.loras
        ]

    schema = get_pipeline().get_parameter_schema()
    resolved_cfg_scale = _resolve_cfg_scale(request.cfg_scale, schema)
    resolved_sampler = _resolve_sampler(request.sampler, schema)

    params = GenerationParams(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        cfg_scale=resolved_cfg_scale,
        sampler=resolved_sampler,
        seed=request.seed,
        loras=loras_list,
    )

    job = job_queue.create_job(params)

    return GenerateResponse(
        job_id=job.id,
        sse_url=f"/api/jobs/{job.id}/sse",
    )


@router.get("/gallery/sse")
async def get_gallery_sse(request: Request):
    """SSE endpoint for gallery updates (new images)."""
    return await gallery_sse_endpoint(request)


@router.get("/jobs/{job_id}/sse")
async def get_job_sse(request: Request, job_id: str):
    """SSE endpoint for job progress updates."""
    return await sse_endpoint(request, job_id)


@router.get("/images", response_model=ImageListResponse)
async def get_images(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> ImageListResponse:
    """Get list of generated images (gallery)."""
    images, total = list_images(limit=limit, offset=offset)

    return ImageListResponse(
        images=[ImageInfo(**img) for img in images],
        total=total,
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
    """Get server information including model name, training resolution, prompt parser, and available LoRAs."""
    from txt2img.config import get_model_config

    config = get_model_config()

    # Get available LoRAs
    available_loras = [
        LoraInfo(
            id=lora.id,
            name=lora.name,
            trigger_words=lora.trigger_words,
            weight=lora.weight,
            trigger_weight=lora.trigger_weight,
        )
        for lora in lora_manager.get_available_loras()
    ]

    return ServerInfo(
        model_name=get_pipeline().model_name or "Not loaded",
        training_resolution=str(config.training_resolution),
        output_format=get_output_format().value,
        available_loras=available_loras,
        parameter_schema=get_pipeline().get_parameter_schema(),
    )
