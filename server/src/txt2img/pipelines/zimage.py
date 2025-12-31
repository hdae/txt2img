"""Z-Image Turbo Pipeline - 6B fast text-to-image model."""

import logging
import random
from collections.abc import Callable
from typing import Any

import torch
from diffusers import ZImagePipeline
from PIL import Image

from txt2img.config import VramProfile, get_model_config, get_settings
from txt2img.core.base_pipeline import BasePipeline
from txt2img.core.image_processor import ImageMetadata, SavedImage, save_image
from txt2img.core.job_queue import GenerationParams

logger = logging.getLogger(__name__)

# Default model for Z-Image Turbo
DEFAULT_ZIMAGE_MODEL = "Tongyi-MAI/Z-Image-Turbo"


class ZImagePipelineImpl(BasePipeline):
    """Z-Image Turbo Pipeline - 6B parameter fast text-to-image model.

    Z-Image Turbo features:
    - 6B parameters
    - 8 inference steps for high-quality output
    - Sub-second generation speed
    - Strong text rendering (English and Chinese)
    - guidance_scale should be 0.0 for best results
    """

    def __init__(self) -> None:
        self.pipe: ZImagePipeline | None = None
        self._model_name: str = ""
        self.loaded_loras: list[str] = []

    @property
    def model_name(self) -> str:
        """Get the loaded model name."""
        return self._model_name

    async def load_model(self) -> None:
        """Load Z-Image Turbo model."""
        config = get_model_config()
        settings = get_settings()

        # Use default model if not specified
        model_ref = config.model if config.model else DEFAULT_ZIMAGE_MODEL

        logger.info(f"Loading Z-Image Turbo model: {model_ref}")

        # Z-Image uses bfloat16
        self.pipe = ZImagePipeline.from_pretrained(
            model_ref,
            torch_dtype=torch.bfloat16,
            token=settings.hf_token,
        )

        self._model_name = model_ref.split("/")[-1] if "/" in model_ref else model_ref

        # Apply VRAM profile
        from txt2img.config import get_vram_profile

        self._apply_vram_profile(get_vram_profile())

        logger.info(f"Z-Image Turbo model loaded: {self._model_name}")

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
        """Generate image using Z-Image Turbo.

        Z-Image Turbo specifics:
        - guidance_scale should be 0.0 for best results
        - Optimal at 8 inference steps
        """
        if not self.pipe:
            raise RuntimeError("Pipeline not loaded")

        # Z-Image Turbo defaults
        steps = params.steps if params.steps else 8
        guidance_scale = 0.0  # Z-Image Turbo works best with 0.0

        logger.info(
            f"Starting Z-Image Turbo generation: "
            f"prompt={params.prompt[:50]}, steps={steps}"
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
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )

        logger.info("Calling Z-Image Turbo pipeline...")
        result = await asyncio.to_thread(_run_pipeline)
        logger.info("Z-Image Turbo pipeline completed")

        image: Image.Image = result.images[0]

        # Save image with metadata
        metadata = ImageMetadata(
            prompt=params.prompt,
            negative_prompt=params.negative_prompt,
            seed=seed,
            steps=steps,
            cfg_scale=guidance_scale,
            width=params.width,
            height=params.height,
            sampler="zimage_turbo",
            model_name=self._model_name,
            loras=None,
        )

        saved = save_image(image, metadata)

        # Clear CUDA cache
        torch.cuda.empty_cache()

        return saved
