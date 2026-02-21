"""Anima Pipeline implementation."""

import logging
import random
from collections.abc import Callable
from typing import Any

import torch
from diffusers_anima import AnimaPipeline
from PIL import Image

from txt2img.config import VramProfile, get_model_config, get_settings
from txt2img.core.base_pipeline import BasePipeline
from txt2img.core.image_processor import ImageMetadata, SavedImage, save_image
from txt2img.core.job_queue import GenerationParams

logger = logging.getLogger(__name__)

# Fixed default model if no model string is given in config
DEFAULT_ANIMA_MODEL = "circlestone-labs/Anima::split_files/diffusion_models/anima-preview.safetensors"

# Default params based on Anima's recommendations
ANIMA_DEFAULT_STEPS = 20
ANIMA_DEFAULT_CFG_SCALE = 4.0

CFG_BATCH_MODE_SPLIT = "split"
CFG_BATCH_MODE_CONCAT = "concat"


class AnimaPipelineImpl(BasePipeline):
    """Anima Pipeline implementation using diffusers-anima."""

    def __init__(self) -> None:
        self.pipe: AnimaPipeline | None = None
        self._model_name: str = ""
        self.loaded_loras: list[str] = []
        self._cfg_batch_mode: str = CFG_BATCH_MODE_SPLIT

    @property
    def model_name(self) -> str:
        """Get the loaded model name."""
        return self._model_name

    def get_parameter_schema(self) -> dict[str, Any]:
        """Get JSON Schema for Anima generation parameters."""
        config = get_model_config()
        return {
            "model_type": "anima",
            "prompt_style": "tags",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Tag-based prompt",
                },
                "negative_prompt": {
                    "type": "string",
                    "description": "Tag-based negative prompt",
                    "default": "",
                },
                "width": {
                    "type": "integer",
                    "default": config.training_resolution,
                    "minimum": 256,
                    "maximum": 2048,
                },
                "height": {
                    "type": "integer",
                    "default": config.training_resolution,
                    "minimum": 256,
                    "maximum": 2048,
                },
                "seed": {
                    "type": ["integer", "null"],
                    "default": None,
                    "description": "Random seed (null for random)",
                },
                "cfg_scale": {
                    "type": "number",
                    "default": config.default_cfg or ANIMA_DEFAULT_CFG_SCALE,
                    "minimum": 0.0,
                    "maximum": 30.0,
                },
            },
            "required": ["prompt"],
            "fixed": {
                "steps": config.default_steps or ANIMA_DEFAULT_STEPS,
            },
        }

    async def load_model(self) -> None:
        """Load Anima model."""
        config = get_model_config()
        settings = get_settings()

        self._configure_sdpa_backends()

        model_ref = config.model if config.model else DEFAULT_ANIMA_MODEL

        logger.info(f"Loading Anima model: {model_ref}")

        # Determine loading function based on if it's a single file or a repo ID
        if model_ref.endswith(".safetensors") or model_ref.endswith(".ckpt"):
            pipe_kwargs = {
                "pretrained_model_link_or_path": model_ref,
                "torch_dtype": torch.bfloat16,
            }
            if settings.hf_token:
                pipe_kwargs["token"] = settings.hf_token
            self.pipe = AnimaPipeline.from_single_file(**pipe_kwargs)
        else:
            pipe_kwargs = {
                "pretrained_model_name_or_path": model_ref,
                "torch_dtype": torch.bfloat16,
            }
            if settings.hf_token:
                pipe_kwargs["token"] = settings.hf_token
            self.pipe = AnimaPipeline.from_pretrained(**pipe_kwargs)

        self._model_name = model_ref.split("/")[-1] if "/" in model_ref else model_ref

        # Apply VRAM profile
        from txt2img.config import get_vram_profile

        profile = get_vram_profile()
        self._apply_vram_profile(profile)
        self._apply_cfg_batch_mode_override(config.cfg_batch_mode, source="preset")
        self._apply_cfg_batch_mode_override(settings.anima_cfg_batch_mode, source="env")
        logger.info(f"Anima CFG batch mode: {self._cfg_batch_mode}")

        logger.info(f"Anima model loaded: {self._model_name}")

    def _configure_sdpa_backends(self) -> None:
        """Enable SDPA backends for CUDA attention kernels."""
        if not torch.cuda.is_available():
            return

        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        logger.info(
            "SDPA backends enabled: flash=%s, mem_efficient=%s, math=%s",
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
        )

    def _apply_vram_profile(self, profile: VramProfile) -> None:
        """Apply VRAM optimization based on profile."""
        if not self.pipe:
            return

        logger.info(f"Applying VRAM profile: {profile.value}")

        if profile == VramProfile.FULL:
            self.pipe = self.pipe.to("cuda")
            self._cfg_batch_mode = CFG_BATCH_MODE_CONCAT
            logger.info("VRAM profile: FULL - no offloading")

        elif profile == VramProfile.BALANCED:
            self.pipe.enable_model_cpu_offload()
            if hasattr(self.pipe, "vae") and hasattr(self.pipe.vae, "enable_tiling"):
                self.pipe.vae.enable_tiling()
            self._cfg_batch_mode = CFG_BATCH_MODE_SPLIT
            logger.info("VRAM profile: BALANCED - model offload + VAE tiling")

        elif profile == VramProfile.LOWVRAM:
            if hasattr(self.pipe, "transformer"):
                self.pipe.transformer.enable_group_offload(
                    onload_device=torch.device("cuda"),
                    offload_device=torch.device("cpu"),
                    offload_type="leaf_level",
                    use_stream=True,
                )
                for name, component in self.pipe.components.items():
                    if name != "transformer" and isinstance(component, torch.nn.Module):
                        component.cuda()
            else:
                self.pipe.enable_sequential_cpu_offload()

            if hasattr(self.pipe, "vae") and hasattr(self.pipe.vae, "enable_tiling"):
                self.pipe.vae.enable_tiling()
            self._cfg_batch_mode = CFG_BATCH_MODE_SPLIT
            logger.info("VRAM profile: LOWVRAM - group/sequential offload + VAE tiling")

    def _apply_cfg_batch_mode_override(
        self,
        mode_override: str | None,
        *,
        source: str,
    ) -> None:
        """Apply optional override for Anima CFG batch mode."""
        if not mode_override:
            return

        normalized = mode_override.strip().lower()
        if normalized in {CFG_BATCH_MODE_SPLIT, CFG_BATCH_MODE_CONCAT}:
            self._cfg_batch_mode = normalized
            logger.info(f"Applied cfg_batch_mode override from {source}: {normalized}")
            return

        logger.warning(
            "Invalid cfg_batch_mode override from %s ('%s'), using %s",
            source,
            mode_override,
            self._cfg_batch_mode,
        )

    async def generate(
        self,
        params: GenerationParams,
        progress_callback: Callable[[int, str | None], Any] | None = None,
    ) -> SavedImage:
        """Generate image using Anima."""
        if not self.pipe:
            raise RuntimeError("Pipeline not loaded")

        config = get_model_config()
        steps = config.default_steps or ANIMA_DEFAULT_STEPS
        cfg_scale = params.cfg_scale

        logger.info(
            f"Starting Anima generation: "
            f"prompt={params.prompt[:50]}, steps={steps}, cfg={cfg_scale}, "
            f"cfg_batch_mode={self._cfg_batch_mode}"
        )

        seed = params.seed if params.seed is not None else random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=config.rng_source).manual_seed(seed)
        logger.info(f"Seed: {seed}")

        import asyncio

        loop = asyncio.get_running_loop()

        step_callback = None
        if progress_callback:
            def _callback_fn(pipe, step, timestep, callback_kwargs):
                asyncio.run_coroutine_threadsafe(
                    progress_callback(step + 1, None),
                    loop
                )
                return callback_kwargs

            step_callback = _callback_fn

        def _run_pipeline():
            with torch.inference_mode():
                kwargs = {
                    "prompt": params.prompt,
                    "negative_prompt": params.negative_prompt or "",
                    "height": params.height,
                    "width": params.width,
                    "num_inference_steps": steps,
                    "guidance_scale": cfg_scale,
                    "generator": generator,
                    "cfg_batch_mode": self._cfg_batch_mode,
                }
                if step_callback:
                    kwargs["callback_on_step_end"] = step_callback

                return self.pipe(**kwargs)

        logger.info("Calling Anima pipeline...")
        result = await asyncio.to_thread(_run_pipeline)
        logger.info("Anima pipeline completed")

        image: Image.Image = result.images[0]

        metadata = ImageMetadata(
            prompt=params.prompt,
            negative_prompt=params.negative_prompt or "",
            seed=seed,
            steps=steps,
            cfg_scale=cfg_scale,
            width=params.width,
            height=params.height,
            model_name=self._model_name,
        )

        saved = save_image(image, metadata)
        torch.cuda.empty_cache()

        return saved
