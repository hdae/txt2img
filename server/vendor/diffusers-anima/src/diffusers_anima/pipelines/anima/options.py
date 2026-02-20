from __future__ import annotations

from dataclasses import dataclass

from .constants import (
    DEFAULT_QWEN_TOKENIZER_REPO,
    DEFAULT_T5_TOKENIZER_REPO,
    DEFAULT_TEXT_ENCODER_CONFIG_REPO,
    DEFAULT_TEXT_ENCODER_WEIGHTS,
    DEFAULT_VAE_REPO,
)


@dataclass(frozen=True)
class AnimaComponents:
    model_path: str
    text_encoder_weights: str = DEFAULT_TEXT_ENCODER_WEIGHTS
    text_encoder_config_repo: str = DEFAULT_TEXT_ENCODER_CONFIG_REPO
    qwen_tokenizer_repo: str = DEFAULT_QWEN_TOKENIZER_REPO
    t5_tokenizer_repo: str = DEFAULT_T5_TOKENIZER_REPO
    vae_repo: str = DEFAULT_VAE_REPO


@dataclass(frozen=True)
class AnimaLoaderOptions:
    local_files_only: bool
    cache_dir: str | None = None
    force_download: bool = False
    token: str | bool | None = None
    revision: str | None = None
    proxies: dict[str, str] | None = None


@dataclass(frozen=True)
class AnimaRuntimeOptions:
    device: str = "auto"
    dtype: str = "auto"
    text_encoder_dtype: str = "auto"
    enable_model_cpu_offload: bool = False
    enable_vae_slicing: bool = False
    enable_vae_tiling: bool = False
    enable_vae_xformers: bool = False
