"""Civitai model downloader."""

import logging
from collections.abc import Callable
from pathlib import Path

import httpx

from txt2img.config import get_settings
from txt2img.models.air_parser import AIRResource

logger = logging.getLogger(__name__)

CIVITAI_API_BASE = "https://civitai.com/api/v1"
CIVITAI_DOWNLOAD_BASE = "https://civitai.com/api/download/models"


async def get_model_info(model_id: str) -> dict:
    """Get model information from Civitai API.

    Args:
        model_id: Civitai model ID

    Returns:
        Model info dict from API
    """
    settings = get_settings()
    headers = {}
    if settings.civitai_api_key:
        headers["Authorization"] = f"Bearer {settings.civitai_api_key}"

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{CIVITAI_API_BASE}/models/{model_id}",
            headers=headers,
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()


async def get_model_version_info(version_id: str) -> dict:
    """Get model version information from Civitai API.

    Args:
        version_id: Civitai model version ID

    Returns:
        Model version info dict from API
    """
    settings = get_settings()
    headers = {}
    if settings.civitai_api_key:
        headers["Authorization"] = f"Bearer {settings.civitai_api_key}"

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{CIVITAI_API_BASE}/model-versions/{version_id}",
            headers=headers,
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()


async def download_model(
    resource: AIRResource,
    target_dir: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """Download model from Civitai.

    Args:
        resource: AIR resource to download
        target_dir: Target directory (defaults to cache_dir)
        progress_callback: Optional callback for download progress (bytes_downloaded, total_bytes)

    Returns:
        Path to downloaded file

    Raises:
        ValueError: If resource is not a Civitai resource
        httpx.HTTPError: If download fails
    """
    settings = get_settings()

    if target_dir is None:
        target_dir = settings.model_cache_dir / "civitai" / resource.type.value

    target_dir.mkdir(parents=True, exist_ok=True)

    version_id = resource.civitai_model_version_id
    if not version_id:
        raise ValueError("Civitai resource must have a version ID for download")

    # Get version info to determine filename
    version_info = await get_model_version_info(version_id)
    files = version_info.get("files", [])

    if not files:
        raise ValueError(f"No files found for model version {version_id}")

    # Prefer safetensors format
    target_file = None
    for file in files:
        if file.get("name", "").endswith(".safetensors"):
            target_file = file
            break
    if target_file is None:
        target_file = files[0]

    filename = target_file["name"]
    download_url = target_file.get("downloadUrl") or f"{CIVITAI_DOWNLOAD_BASE}/{version_id}"
    file_path = target_dir / filename

    # Skip if already downloaded
    if file_path.exists():
        expected_size = target_file.get("sizeKB", 0) * 1024
        if expected_size and file_path.stat().st_size >= expected_size * 0.99:
            logger.info(f"Model already exists: {file_path}")
            return file_path

    logger.info(f"Downloading {filename} from Civitai...")

    headers = {}
    if settings.civitai_api_key:
        headers["Authorization"] = f"Bearer {settings.civitai_api_key}"

    async with httpx.AsyncClient(follow_redirects=True) as client:
        async with client.stream(
            "GET",
            download_url,
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
