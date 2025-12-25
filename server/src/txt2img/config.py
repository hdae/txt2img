"""Configuration module using pydantic-settings and JSON config."""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class OutputFormat(str, Enum):
    """Output image format."""

    PNG = "png"
    WEBP = "webp"


class PromptParser(str, Enum):
    """Prompt parsing mode."""

    LPW = "lpw"  # A1111/WebUI compatible (Long Prompt Weighting)
    COMPEL = "compel"  # Compel library native syntax


class QuantizationType(str, Enum):
    """Quantization type for model optimization."""

    NONE = "none"
    INT8_WEIGHT_ONLY = "int8wo"
    INT4_WEIGHT_ONLY = "int4wo"
    FP8_WEIGHT_ONLY = "fp8wo"


class ModelType(str, Enum):
    """Model architecture type (follows AIR naming)."""

    SDXL = "sdxl"  # Stable Diffusion XL and derivatives (Illustrious, etc.)
    SD3 = "sd3"  # Stable Diffusion 3.x
    FLUX = "flux"  # Flux.1 / Flux 2


@dataclass
class ModelConfig:
    """Model configuration loaded from JSON."""

    # Required
    model: str  # AIR URN, HuggingFace repo, or URL

    # Model type
    type: ModelType = ModelType.SDXL

    # VAE (null = use model's embedded VAE)
    vae: str | None = None

    # LoRAs
    loras: list[str] = field(default_factory=list)

    # Performance
    quantization: QuantizationType = QuantizationType.NONE
    vae_tiling: bool = False

    # Generation defaults
    training_resolution: int = 1024
    default_steps: int = 20
    default_cfg: float = 7.0
    default_sampler: str = "euler"

    # Output
    output_format: OutputFormat = OutputFormat.WEBP

    # Prompt parsing
    prompt_parser: PromptParser = PromptParser.LPW

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        """Create ModelConfig from dictionary."""
        # Handle enum conversions
        if "type" in data and isinstance(data["type"], str):
            data["type"] = ModelType(data["type"])
        if "quantization" in data and isinstance(data["quantization"], str):
            data["quantization"] = QuantizationType(data["quantization"])
        if "output_format" in data and isinstance(data["output_format"], str):
            data["output_format"] = OutputFormat(data["output_format"])
        if "prompt_parser" in data and isinstance(data["prompt_parser"], str):
            data["prompt_parser"] = PromptParser(data["prompt_parser"])

        # Filter to only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}

        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "model": self.model,
            "vae": self.vae,
            "loras": self.loras,
            "quantization": self.quantization.value,
            "vae_tiling": self.vae_tiling,
            "training_resolution": self.training_resolution,
            "default_steps": self.default_steps,
            "default_cfg": self.default_cfg,
            "default_sampler": self.default_sampler,
            "output_format": self.output_format.value,
            "prompt_parser": self.prompt_parser.value,
        }


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    These are infrastructure-level settings, not model configuration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Config source (JSON string, URL, or file path)
    config: str | None = Field(
        default=None, description="Model config as JSON string, URL, or file:// path"
    )

    # API keys
    civitai_api_key: str | None = Field(default=None, description="Civitai API key")
    hf_token: str | None = Field(default=None, description="HuggingFace token")

    # Directories
    output_dir: Path = Field(default=Path("./outputs"), description="Output directory")
    model_cache_dir: Path = Field(
        default=Path("/workspace/models"), description="Model cache directory"
    )
    presets_dir: Path = Field(
        default=Path("/workspace/app/presets"), description="Presets directory"
    )

    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global model config (mutable, loaded at startup)
_model_config: ModelConfig | None = None


def load_model_config(config_source: str | None = None) -> ModelConfig:
    """Load model configuration from various sources.

    Args:
        config_source: JSON string, URL (https://), or file path (file://)
                      If None, uses CONFIG environment variable

    Returns:
        ModelConfig instance

    Raises:
        ValueError: If no valid config found
    """
    global _model_config

    settings = get_settings()
    source = config_source or settings.config

    if not source:
        raise ValueError(
            "No config provided. Set CONFIG environment variable or pass config_source"
        )

    config_data: dict[str, Any] | None = None

    # Try to parse as JSON string first
    if source.strip().startswith("{"):
        try:
            config_data = json.loads(source)
            logger.info("Loaded config from JSON string")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON config: {e}") from e

    # Try as URL (https://)
    elif source.startswith("https://") or source.startswith("http://"):
        try:
            response = httpx.get(source, timeout=30.0)
            response.raise_for_status()
            config_data = response.json()
            logger.info(f"Loaded config from URL: {source}")
        except Exception as e:
            raise ValueError(f"Failed to load config from URL: {e}") from e

    # Try as file path (file:// or direct path)
    elif source.startswith("file://") or source.startswith("/") or Path(source).exists():
        file_path = source.replace("file://", "") if source.startswith("file://") else source
        path = Path(file_path)

        # Check presets directory if relative path
        if not path.is_absolute() and not path.exists():
            preset_path = settings.presets_dir / path
            if preset_path.exists():
                path = preset_path

        if not path.exists():
            raise ValueError(f"Config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            config_data = json.load(f)
        logger.info(f"Loaded config from file: {path}")

    else:
        # Try as preset name (without .json extension)
        preset_path = settings.presets_dir / f"{source}.json"
        if preset_path.exists():
            with open(preset_path, encoding="utf-8") as f:
                config_data = json.load(f)
            logger.info(f"Loaded config from preset: {preset_path}")
        else:
            raise ValueError(f"Unknown config format: {source}")

    if config_data is None:
        raise ValueError("Failed to load config data")

    _model_config = ModelConfig.from_dict(config_data)
    return _model_config


def get_model_config() -> ModelConfig:
    """Get the loaded model configuration.

    Raises:
        RuntimeError: If config not loaded yet
    """
    if _model_config is None:
        raise RuntimeError("Model config not loaded. Call load_model_config() first.")
    return _model_config
