"""Configuration module using pydantic-settings."""

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OutputFormat(str, Enum):
    """Output image format."""

    PNG = "png"
    WEBP = "webp"


class PromptParser(str, Enum):
    """Prompt parsing mode."""

    LEGACY = "legacy"  # StableDiffusionWebUI compatible
    COMPEL = "compel"  # Compel library


class QuantizationType(str, Enum):
    """Quantization type for model optimization."""

    NONE = "none"  # No quantization
    INT8_WEIGHT_ONLY = "int8wo"  # TorchAO int8 weight-only
    INT4_WEIGHT_ONLY = "int4wo"  # TorchAO int4 weight-only
    FP8_WEIGHT_ONLY = "fp8wo"  # TorchAO fp8 weight-only (requires H100/4090)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Model settings
    model: str = Field(description="Model AIR URN or HuggingFace repository")
    lora: str | None = Field(default=None, description="LoRA URNs/URLs, comma-separated")
    preset_url: str | None = Field(default=None, description="Preset YAML/JSON URL")

    # API keys
    civitai_api_key: str | None = Field(default=None, description="Civitai API key")
    hf_token: str | None = Field(default=None, description="HuggingFace token")

    # Directories
    output_dir: Path = Field(default=Path("./outputs"), description="Output directory")
    model_cache_dir: Path = Field(
        default=Path("/workspace/models"), description="Model cache directory"
    )

    # Output settings
    output_format: OutputFormat = Field(
        default=OutputFormat.WEBP, description="Output image format"
    )
    training_resolution: str = Field(
        default="1024", description="Training resolution (e.g., '1024' or '1024x1024')"
    )

    # Prompt parsing
    prompt_parser: PromptParser = Field(
        default=PromptParser.LEGACY, description="Prompt parsing mode"
    )

    # Performance settings
    quantization: QuantizationType = Field(
        default=QuantizationType.NONE, description="Quantization type (none, int8wo, int4wo, fp8wo)"
    )
    vae_tiling: bool = Field(default=False, description="Force enable VAE tiling for memory saving")

    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")

    def get_base_resolution(self) -> int:
        """Parse training_resolution and return base size in pixels."""
        if "x" in self.training_resolution:
            w, h = self.training_resolution.split("x")
            return int((int(w) * int(h)) ** 0.5)
        return int(self.training_resolution)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
