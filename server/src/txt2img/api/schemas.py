"""API schemas using Pydantic."""

from typing import Any

from pydantic import BaseModel, Field


class LoraRequest(BaseModel):
    """LoRA request for generation."""

    id: str = Field(description="LoRA ID")
    weight: float = Field(
        default=1.0, ge=0.0, le=2.0, description="LoRA weight (overall influence)"
    )
    trigger_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description="Trigger embedding weight (target influence)",
    )


class GenerateRequest(BaseModel):
    """Request body for image generation.

    Note: steps, cfg_scale, sampler are fixed per pipeline and not configurable.
    """

    prompt: str = Field(description="Positive prompt")
    negative_prompt: str = Field(default="", description="Negative prompt (SDXL only)")
    width: int = Field(default=1024, ge=256, le=2048, description="Image width")
    height: int = Field(default=1024, ge=256, le=2048, description="Image height")
    seed: int | None = Field(default=None, description="Random seed (null for random)")
    loras: list[LoraRequest] | None = Field(default=None, description="LoRAs to apply (SDXL only)")


class GenerateResponse(BaseModel):
    """Response from generate endpoint."""

    job_id: str = Field(description="Job ID")
    sse_url: str = Field(description="SSE endpoint URL for progress updates")


class ImageInfo(BaseModel):
    """Image information for gallery."""

    id: str
    thumbnail_url: str
    full_url: str
    metadata: dict


class ImageListResponse(BaseModel):
    """Response from image list endpoint."""

    images: list[ImageInfo]
    total: int
    offset: int
    limit: int


class LoraInfo(BaseModel):
    """LoRA information for /info endpoint."""

    id: str
    name: str
    trigger_words: list[str] = Field(default_factory=list)
    weight: float = Field(default=1.0, description="Recommended LoRA weight")
    trigger_weight: float = Field(default=0.5, description="Recommended trigger embedding weight")


class ServerInfo(BaseModel):
    """Server information response."""

    model_name: str
    training_resolution: str
    output_format: str = Field(default="png", description="Output image format (png or webp)")
    available_loras: list[LoraInfo] = Field(default_factory=list)
    parameter_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for generation parameters"
    )


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: str | None = None
