"""Flux.1 [dev] Pipeline - guidance-distilled variant."""

import logging
import random
from collections.abc import Callable
from typing import Any

import torch
from diffusers import FluxPipeline
from PIL import Image

from txt2img.config import VramProfile, get_model_config, get_settings
from txt2img.core.base_pipeline import BasePipeline
from txt2img.core.image_processor import ImageMetadata, SavedImage, save_image
from txt2img.core.job_queue import GenerationParams
from txt2img.providers.civitai import download_model as civitai_download
from txt2img.providers.huggingface import download_from_url
from txt2img.utils.air_parser import (
    AIRResource,
    HuggingFaceResource,
    ModelSource,
    URLResource,
    parse_model_ref,
)

logger = logging.getLogger(__name__)

# Default model for Flux.1 [dev]
DEFAULT_FLUX_DEV_MODEL = "black-forest-labs/FLUX.1-dev"

# Fixed generation parameters for Flux.1 [dev]
FLUX_DEV_FIXED_STEPS = 50
FLUX_DEV_FIXED_CFG_SCALE = 3.5


class FluxDevPipeline(BasePipeline):
    """Flux.1 [dev] Pipeline - guidance-distilled variant.

    Flux.1 [dev] features:
    - Guidance-distilled (uses guidance_scale around 3.5)
    - Recommended ~50 inference steps
    - Non-commercial license
    """

    def __init__(self) -> None:
        self.pipe: FluxPipeline | None = None
        self._model_name: str = ""
        self.loaded_loras: list[str] = []

    @property
    def model_name(self) -> str:
        """Get the loaded model name."""
        return self._model_name

    def get_parameter_schema(self) -> dict[str, Any]:
        """Get JSON Schema for Flux.1 [dev] generation parameters."""
        return {
            "model_type": "flux_dev",
            "prompt_style": "natural",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Natural language prompt",
                },
                "width": {
                    "type": "integer",
                    "default": 1024,
                    "minimum": 256,
                    "maximum": 2048,
                },
                "height": {
                    "type": "integer",
                    "default": 1024,
                    "minimum": 256,
                    "maximum": 2048,
                },
                "seed": {
                    "type": ["integer", "null"],
                    "default": None,
                    "description": "Random seed (null for random)",
                },
            },
            "required": ["prompt"],
            "fixed": {
                "steps": FLUX_DEV_FIXED_STEPS,
                "cfg_scale": FLUX_DEV_FIXED_CFG_SCALE,
            },
        }

    async def load_model(self) -> None:
        """Load Flux.1 [dev] model from various sources."""
        config = get_model_config()
        settings = get_settings()

        # Use default model if not specified
        model_ref_str = config.model if config.model else DEFAULT_FLUX_DEV_MODEL

        logger.info(f"Loading Flux.1 [dev] model: {model_ref_str}")

        model_ref = parse_model_ref(model_ref_str)

        # Flux uses bfloat16 by default
        if isinstance(model_ref, AIRResource):
            if model_ref.source == ModelSource.CIVITAI:
                model_path = await civitai_download(model_ref)
                self._model_name = model_path.stem
                self.pipe = FluxPipeline.from_single_file(
                    str(model_path),
                    torch_dtype=torch.bfloat16,
                )
            else:
                raise ValueError(f"Unsupported AIR source: {model_ref.source}")
        elif isinstance(model_ref, HuggingFaceResource):
            self._model_name = model_ref.repo_id.split("/")[-1]
            self.pipe = FluxPipeline.from_pretrained(
                model_ref.repo_id,
                torch_dtype=torch.bfloat16,
                token=settings.hf_token,
            )
        elif isinstance(model_ref, URLResource):
            downloaded_path = await download_from_url(model_ref)
            self._model_name = downloaded_path.stem
            self.pipe = FluxPipeline.from_single_file(
                str(downloaded_path),
                torch_dtype=torch.bfloat16,
            )
        else:
            raise ValueError(f"Unsupported model reference type: {type(model_ref)}")

        # Apply VRAM profile
        from txt2img.config import get_vram_profile

        self._apply_vram_profile(get_vram_profile())

        logger.info(f"Flux.1 [dev] model loaded: {self._model_name}")

    def _apply_vram_profile(self, profile: VramProfile) -> None:
        """Apply VRAM optimization based on profile."""
        if not self.pipe:
            return

        logger.info(f"Applying VRAM profile: {profile.value}")

        if profile == VramProfile.FULL:
            self.pipe = self.pipe.to("cuda")
            logger.info("VRAM profile: FULL - no offloading")

        elif profile == VramProfile.BALANCED:
            self.pipe.enable_model_cpu_offload()
            self.pipe.vae.enable_tiling()
            logger.info("VRAM profile: BALANCED - model offload + VAE tiling")

        elif profile == VramProfile.LOWVRAM:
            self.pipe.transformer.enable_group_offload(
                onload_device=torch.device("cuda"),
                offload_device=torch.device("cpu"),
                offload_type="leaf_level",
                use_stream=True,
            )
            for name, component in self.pipe.components.items():
                if name != "transformer" and isinstance(component, torch.nn.Module):
                    component.cuda()
            self.pipe.vae.enable_tiling()
            logger.info("VRAM profile: LOWVRAM - group offload + VAE tiling")

    async def generate(
        self,
        params: GenerationParams,
        progress_callback: Callable[[int, str | None], Any] | None = None,
    ) -> SavedImage:
        """Generate image using Flux.1 [dev]."""
        if not self.pipe:
            raise RuntimeError("Pipeline not loaded")

        logger.info(
            f"Starting Flux.1 [dev] generation: "
            f"prompt={params.prompt[:50]}, steps={FLUX_DEV_FIXED_STEPS}"
        )

        # Determine seed
        seed = params.seed if params.seed is not None else random.randint(0, 2**32 - 1)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        logger.info(f"Seed: {seed}")

        # Run pipeline
        import asyncio

        def _run_pipeline():
            with torch.inference_mode():
                return self.pipe(
                    prompt=params.prompt,
                    height=params.height,
                    width=params.width,
                    num_inference_steps=FLUX_DEV_FIXED_STEPS,
                    guidance_scale=FLUX_DEV_FIXED_CFG_SCALE,
                    generator=generator,
                )

        logger.info("Calling Flux [dev] pipeline...")
        result = await asyncio.to_thread(_run_pipeline)
        logger.info("Flux [dev] pipeline completed")

        image: Image.Image = result.images[0]

        # Save image with metadata
        metadata = ImageMetadata(
            prompt=params.prompt,
            negative_prompt="",
            seed=seed,
            steps=FLUX_DEV_FIXED_STEPS,
            cfg_scale=FLUX_DEV_FIXED_CFG_SCALE,
            width=params.width,
            height=params.height,
            model_name=self._model_name,
        )

        saved = save_image(image, metadata)

        # Clear CUDA cache
        torch.cuda.empty_cache()

        return saved
