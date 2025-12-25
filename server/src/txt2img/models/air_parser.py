"""Civitai AIR URN parser.

AIR format: urn:air:{ecosystem}:{type}:{source}:{id}@{version?}.{format?}
Examples:
    - urn:air:sdxl:checkpoint:civitai:827184@2514310
    - urn:air:sdxl:lora:civitai:328553@368189
"""

import re
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse


class ModelEcosystem(str, Enum):
    """Supported model ecosystems."""

    SD1 = "sd1"
    SD2 = "sd2"
    SDXL = "sdxl"
    FLUX = "flux"


class ModelType(str, Enum):
    """Supported model types."""

    CHECKPOINT = "checkpoint"
    LORA = "lora"
    EMBEDDING = "embedding"


class ModelSource(str, Enum):
    """Model sources."""

    CIVITAI = "civitai"
    HUGGINGFACE = "huggingface"


@dataclass(frozen=True)
class AIRResource:
    """Parsed AIR URN resource."""

    ecosystem: ModelEcosystem
    type: ModelType
    source: ModelSource
    id: str
    version: str | None = None
    format: str | None = None

    @property
    def civitai_model_version_id(self) -> str | None:
        """Get Civitai model version ID if available."""
        if self.source == ModelSource.CIVITAI and self.version:
            return self.version
        return None

    @property
    def civitai_model_id(self) -> str | None:
        """Get Civitai model ID if available."""
        if self.source == ModelSource.CIVITAI:
            return self.id
        return None


@dataclass(frozen=True)
class URLResource:
    """Direct URL resource."""

    url: str
    filename: str


@dataclass(frozen=True)
class HuggingFaceResource:
    """HuggingFace Hub resource."""

    repo_id: str
    filename: str | None = None


ModelRef = AIRResource | URLResource | HuggingFaceResource


# AIR URN pattern
AIR_PATTERN = re.compile(
    r"^(?:urn:)?(?:air:)?"
    r"(?P<ecosystem>sd1|sd2|sdxl|flux):"
    r"(?P<type>checkpoint|lora|embedding):"
    r"(?P<source>civitai|huggingface):"
    r"(?P<id>[^@.]+)"
    r"(?:@(?P<version>[^.]+))?"
    r"(?:\.(?P<format>\w+))?$",
    re.IGNORECASE,
)


def parse_air_urn(urn: str) -> AIRResource:
    """Parse AIR URN string into AIRResource.

    Args:
        urn: AIR URN string

    Returns:
        Parsed AIRResource

    Raises:
        ValueError: If URN format is invalid
    """
    match = AIR_PATTERN.match(urn.strip())
    if not match:
        raise ValueError(f"Invalid AIR URN format: {urn}")

    return AIRResource(
        ecosystem=ModelEcosystem(match.group("ecosystem").lower()),
        type=ModelType(match.group("type").lower()),
        source=ModelSource(match.group("source").lower()),
        id=match.group("id"),
        version=match.group("version"),
        format=match.group("format"),
    )


def is_air_urn(ref: str) -> bool:
    """Check if string is an AIR URN."""
    return AIR_PATTERN.match(ref.strip()) is not None


def is_url(ref: str) -> bool:
    """Check if string is a URL."""
    try:
        result = urlparse(ref)
        return result.scheme in ("http", "https") and bool(result.netloc)
    except Exception:
        return False


def is_huggingface_repo(ref: str) -> bool:
    """Check if string looks like a HuggingFace repo ID (org/repo format)."""
    if is_url(ref) or is_air_urn(ref):
        return False
    # HF repos are typically in format: org/repo or user/repo
    return bool(re.match(r"^[\w.-]+/[\w.-]+$", ref))


def parse_model_ref(ref: str) -> ModelRef:
    """Parse model reference string into appropriate type.

    Supports:
        - AIR URN: urn:air:sdxl:checkpoint:civitai:12345@67890
        - URL: https://example.com/model.safetensors
        - HuggingFace repo: stabilityai/stable-diffusion-xl-base-1.0

    Args:
        ref: Model reference string

    Returns:
        Parsed ModelRef (AIRResource, URLResource, or HuggingFaceResource)

    Raises:
        ValueError: If reference format is not recognized
    """
    ref = ref.strip()

    if is_air_urn(ref):
        return parse_air_urn(ref)

    if is_url(ref):
        # Extract filename from URL
        parsed = urlparse(ref)
        path_parts = parsed.path.split("/")
        filename = path_parts[-1] if path_parts else "model.safetensors"
        return URLResource(url=ref, filename=filename)

    if is_huggingface_repo(ref):
        return HuggingFaceResource(repo_id=ref)

    raise ValueError(f"Unrecognized model reference format: {ref}")
