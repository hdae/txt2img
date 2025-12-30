"""Chroma Pipeline - lightweight 8.9B Flux variant."""

import logging
import random
from collections.abc import Callable
from typing import Any

import torch
from diffusers import ChromaPipeline
from PIL import Image

from txt2img.config import get_model_config, get_settings
from txt2img.core.base_pipeline import BasePipeline
from txt2img.core.image_processor import ImageMetadata, SavedImage, save_image
from txt2img.core.job_queue import GenerationParams

logger = logging.getLogger(__name__)

# Default model for Chroma
DEFAULT_CHROMA_MODEL = "lodestones/Chroma1-Flash"


class ChromaPipelineImpl(BasePipeline):
    """Chroma Pipeline - lightweight 8.9B parameter Flux variant.

    Chroma is optimized for:
    - Single T5 text encoder (no CLIP)
    - Strong prompt adherence
    - Face quality improvements
    """

    def __init__(self) -> None:
        self.pipe: ChromaPipeline | None = None
        self._model_name: str = ""
        self.loaded_loras: list[str] = []

    @property
    def model_name(self) -> str:
        """Get the loaded model name."""
        return self._model_name

    async def load_model(self) -> None:
        """Load Chroma model."""
        config = get_model_config()
        settings = get_settings()

        # Use default model if not specified
        model_ref = config.model if config.model else DEFAULT_CHROMA_MODEL

        logger.info(f"Loading Chroma model: {model_ref}")

        # Chroma uses bfloat16 by default
        self.pipe = ChromaPipeline.from_pretrained(
            model_ref,
            torch_dtype=torch.bfloat16,
            token=settings.hf_token,
        )

        self._model_name = model_ref.split("/")[-1] if "/" in model_ref else model_ref

        # Move to GPU
        self.pipe = self.pipe.to("cuda")

        logger.info(f"Chroma model loaded: {self._model_name}")

    async def generate(
        self,
        params: GenerationParams,
        progress_callback: Callable[[int, str | None], Any] | None = None,
    ) -> SavedImage:
        """Generate image using Chroma.

        Chroma specifics:
        - Uses single T5 encoder
        - Recommended 4-20 steps depending on variant
        - guidance_scale around 3.5-7.5
        """
        if not self.pipe:
            raise RuntimeError("Pipeline not loaded")

        # Chroma defaults
        steps = params.steps if params.steps else 4
        guidance_scale = params.cfg_scale if params.cfg_scale else 4.0

        logger.info(
            f"Starting Chroma generation: "
            f"prompt={params.prompt[:50]}, steps={steps}, guidance={guidance_scale}"
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
                    width=params.height,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )

        logger.info("Calling Chroma pipeline...")
        result = await asyncio.to_thread(_run_pipeline)
        logger.info("Chroma pipeline completed")

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
            sampler="chroma",
            model_name=self._model_name,
            loras=None,
        )

        saved = save_image(image, metadata)

        # Clear CUDA cache
        torch.cuda.empty_cache()

        return saved
