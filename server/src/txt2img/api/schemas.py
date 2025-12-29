"""API schemas using Pydantic."""

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
    """Request body for image generation."""

    prompt: str = Field(description="Positive prompt")
    negative_prompt: str = Field(default="", description="Negative prompt")
    width: int = Field(default=1024, ge=256, le=2048, description="Image width")
    height: int = Field(default=1024, ge=256, le=2048, description="Image height")
    steps: int = Field(default=20, ge=1, le=100, description="Number of inference steps")
    cfg_scale: float = Field(default=7.0, ge=1.0, le=30.0, description="CFG scale")
    seed: int | None = Field(default=None, description="Random seed (null for random)")
    sampler: str = Field(default="euler", description="Sampler name")
    loras: list[LoraRequest] | None = Field(default=None, description="LoRAs to apply with weights")


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
    available_loras: list[LoraInfo] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: str | None = None
