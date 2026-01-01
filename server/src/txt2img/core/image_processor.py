"""Image processor for output generation and metadata handling."""

import io
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from txt2img.config import OutputFormat, get_settings

logger = logging.getLogger(__name__)


@dataclass
class ImageMetadata:
    """Metadata for generated image."""

    prompt: str
    negative_prompt: str
    seed: int
    steps: int
    cfg_scale: float
    width: int
    height: int
    model_name: str
    sampler: str | None = None  # SDXL only
    loras: list[str] | None = None
    created_at: str | None = None

    def to_pnginfo_string(self) -> str:
        """Convert to StableDiffusionWebUI compatible pnginfo string."""
        parts = [self.prompt]

        if self.negative_prompt:
            parts.append(f"Negative prompt: {self.negative_prompt}")

        params = [
            f"Steps: {self.steps}",
            f"CFG scale: {self.cfg_scale}",
            f"Seed: {self.seed}",
            f"Size: {self.width}x{self.height}",
            f"Model: {self.model_name}",
        ]

        if self.sampler:
            params.insert(1, f"Sampler: {self.sampler}")

        if self.loras:
            lora_str = ", ".join(self.loras)
            params.append(f"Lora: {lora_str}")

        parts.append(", ".join(params))
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            "width": self.width,
            "height": self.height,
            "sampler": self.sampler,
            "model_name": self.model_name,
            "loras": self.loras,
            "created_at": self.created_at,
        }


@dataclass
class SavedImage:
    """Information about a saved image."""

    id: str
    full_path: Path
    thumbnail_path: Path
    format: OutputFormat
    metadata: ImageMetadata


def generate_image_id() -> str:
    """Generate unique image ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique = uuid.uuid4().hex[:8]
    return f"{timestamp}_{unique}"


def save_image(
    image: Image.Image,
    metadata: ImageMetadata,
    image_id: str | None = None,
) -> SavedImage:
    """Save image with metadata in configured format.

    Args:
        image: PIL Image to save
        metadata: Image metadata
        image_id: Optional image ID (generated if not provided)

    Returns:
        SavedImage with paths and info
    """
    from txt2img.config import get_output_format

    settings = get_settings()
    output_format = get_output_format()

    if image_id is None:
        image_id = generate_image_id()

    # Ensure output directory exists
    output_dir = settings.output_dir / "images"
    thumb_dir = settings.output_dir / "thumbs"
    output_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir.mkdir(parents=True, exist_ok=True)

    # Add creation timestamp to metadata
    metadata.created_at = datetime.now().isoformat()
    pnginfo_str = metadata.to_pnginfo_string()

    # Save full size image
    if output_format == OutputFormat.PNG:
        full_path = output_dir / f"{image_id}.png"
        pnginfo = PngInfo()
        pnginfo.add_text("parameters", pnginfo_str)
        image.save(full_path, format="PNG", pnginfo=pnginfo)
    else:
        full_path = output_dir / f"{image_id}.webp"
        # WebP stores metadata in EXIF-like format
        exif_data = image.getexif()
        # Store in UserComment (tag 37510)
        exif_data[37510] = pnginfo_str.encode("utf-8")
        image.save(full_path, format="WEBP", quality=95, exif=exif_data)

    # Save thumbnail (always WebP, 1/4 size)
    thumb_size = (image.width // 4, image.height // 4)
    thumbnail = image.copy()
    thumbnail.thumbnail(thumb_size, Image.Resampling.LANCZOS)
    thumb_path = thumb_dir / f"{image_id}.webp"
    thumbnail.save(thumb_path, format="WEBP", quality=80)

    # Save metadata JSON
    meta_dir = settings.output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / f"{image_id}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata.to_dict(), f, ensure_ascii=False, indent=2)

    logger.info(f"Saved image: {full_path}")

    return SavedImage(
        id=image_id,
        full_path=full_path,
        thumbnail_path=thumb_path,
        format=output_format,
        metadata=metadata,
    )


def load_image_metadata(image_id: str) -> ImageMetadata | None:
    """Load metadata for an image.

    Args:
        image_id: Image ID

    Returns:
        ImageMetadata or None if not found
    """
    settings = get_settings()
    meta_path = settings.output_dir / "metadata" / f"{image_id}.json"

    if not meta_path.exists():
        return None

    with open(meta_path, encoding="utf-8") as f:
        data = json.load(f)

    return ImageMetadata(**data)


def list_images(limit: int = 50, offset: int = 0) -> list[dict]:
    """List generated images.

    Args:
        limit: Maximum number of images to return
        offset: Offset for pagination

    Returns:
        List of image info dicts
    """
    settings = get_settings()
    meta_dir = settings.output_dir / "metadata"

    if not meta_dir.exists():
        return []

    # Get all metadata files sorted by modification time (newest first)
    meta_files = sorted(
        meta_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    # Apply pagination
    meta_files = meta_files[offset : offset + limit]

    results = []
    for meta_path in meta_files:
        image_id = meta_path.stem
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)

        results.append(
            {
                "id": image_id,
                "thumbnail_url": f"/api/images/{image_id}?thumbnail=true",
                "full_url": f"/api/images/{image_id}",
                "metadata": metadata,
            }
        )

    return results


def image_to_base64(image: Image.Image, format: str = "webp", quality: int = 80) -> str:
    """Convert PIL Image to base64 string.

    Args:
        image: PIL Image
        format: Output format (webp, png, jpeg)
        quality: Quality for lossy formats

    Returns:
        Base64 encoded string
    """
    import base64

    buffer = io.BytesIO()

    if format.lower() == "png":
        image.save(buffer, format="PNG")
    elif format.lower() == "jpeg":
        image.save(buffer, format="JPEG", quality=quality)
    else:
        image.save(buffer, format="WEBP", quality=quality)

    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")
