"""HuggingFace model downloader."""

import logging
from collections.abc import Callable
from pathlib import Path

import httpx

from txt2img.config import get_settings
from txt2img.utils.air_parser import HuggingFaceResource, URLResource

logger = logging.getLogger(__name__)


async def download_from_url(
    resource: URLResource,
    target_dir: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """Download file from direct URL.

    Args:
        resource: URL resource to download
        target_dir: Target directory (defaults to cache_dir)
        progress_callback: Optional callback for download progress

    Returns:
        Path to downloaded file
    """
    settings = get_settings()

    if target_dir is None:
        target_dir = settings.model_cache_dir / "downloads"

    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / resource.filename

    # Skip if already downloaded
    if file_path.exists():
        logger.info(f"File already exists: {file_path}")
        return file_path

    logger.info(f"Downloading {resource.filename} from URL...")

    async with httpx.AsyncClient(follow_redirects=True) as client:
        async with client.stream(
            "GET",
            resource.url,
            timeout=httpx.Timeout(30.0, read=None),
        ) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))

            with open(file_path, "wb") as f:
                downloaded = 0
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total_size)

    logger.info(f"Downloaded: {file_path}")
    return file_path


def get_hf_model_path(resource: HuggingFaceResource) -> str:
    """Get HuggingFace model path for diffusers loading.

    For HuggingFace resources, we return the repo_id directly
    as diffusers can load from hub.

    Args:
        resource: HuggingFace resource

    Returns:
        Repo ID string for diffusers
    """
    return resource.repo_id


async def download_hf_lora(
    resource: HuggingFaceResource,
    filename: str = "pytorch_lora_weights.safetensors",
    target_dir: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """Download LoRA from HuggingFace Hub.

    Args:
        resource: HuggingFace resource
        filename: LoRA filename in the repo
        target_dir: Target directory
        progress_callback: Optional callback for download progress

    Returns:
        Path to downloaded file
    """
    settings = get_settings()

    if target_dir is None:
        target_dir = settings.model_cache_dir / "huggingface" / "lora"

    target_dir.mkdir(parents=True, exist_ok=True)

    # Construct HuggingFace URL
    url = f"https://huggingface.co/{resource.repo_id}/resolve/main/{filename}"

    safe_repo_name = resource.repo_id.replace("/", "_")
    file_path = target_dir / f"{safe_repo_name}_{filename}"

    if file_path.exists():
        logger.info(f"LoRA already exists: {file_path}")
        return file_path

    logger.info(f"Downloading LoRA from HuggingFace: {resource.repo_id}/{filename}")

    headers = {}
    if settings.hf_token:
        headers["Authorization"] = f"Bearer {settings.hf_token}"

    async with httpx.AsyncClient(follow_redirects=True) as client:
        async with client.stream(
            "GET",
            url,
            headers=headers,
            timeout=httpx.Timeout(30.0, read=None),
        ) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))

            with open(file_path, "wb") as f:
                downloaded = 0
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total_size)

    logger.info(f"Downloaded: {file_path}")
    return file_path
