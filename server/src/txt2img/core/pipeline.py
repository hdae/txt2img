"""SDXL Pipeline with model configuration support."""

import logging
import random
from collections.abc import Callable
from typing import Any

import torch
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
)
from PIL import Image

from txt2img.config import (
    ModelType,
    QuantizationType,
    get_model_config,
    get_settings,
)
from txt2img.core.image_processor import ImageMetadata, SavedImage, save_image
from txt2img.core.job_queue import GenerationParams
from txt2img.models.air_parser import (
    AIRResource,
    HuggingFaceResource,
    ModelSource,
    URLResource,
    parse_model_ref,
)
from txt2img.models.civitai import download_model as civitai_download
from txt2img.models.huggingface import download_from_url, get_hf_model_path

logger = logging.getLogger(__name__)

# Scheduler mapping
SCHEDULERS = {
    "euler": EulerDiscreteScheduler,
    "euler_a": EulerAncestralDiscreteScheduler,
    "dpm++_2m": DPMSolverMultistepScheduler,
}


class SDXLPipeline:
    """SDXL Pipeline manager with model configuration support."""

    def __init__(self) -> None:
        self.pipe: StableDiffusionXLPipeline | None = None
        self.model_name: str = ""
        self.loaded_loras: list[str] = []
        self._quantized = False

    async def load_model(self) -> None:
        """Load model from ModelConfig."""
        config = get_model_config()

        logger.info(f"Loading model: {config.model} (type: {config.type.value})")

        # Load base model based on type
        if config.type in (ModelType.SDXL,):
            await self._load_sdxl_model(config.model)
        elif config.type == ModelType.SD3:
            raise NotImplementedError("SD3 pipeline not yet implemented")
        elif config.type == ModelType.FLUX:
            raise NotImplementedError("Flux pipeline not yet implemented")
        else:
            raise ValueError(f"Unsupported model type: {config.type}")

        # Move to GPU
        self.pipe = self.pipe.to("cuda")

        # Load VAE if specified
        await self._load_vae(config.vae)

        # Enable memory optimizations
        self.pipe.vae.enable_slicing()

        if config.vae_tiling:
            self.pipe.vae.enable_tiling()
            logger.info("VAE tiling enabled")

        # Apply quantization
        await self._apply_quantization(config.quantization)

        logger.info(f"Model loaded: {self.model_name}")

        # Load LoRAs if configured
        if config.loras:
            await self.load_loras(config.loras)

    async def _load_sdxl_model(self, model_ref_str: str) -> None:
        """Load SDXL model from various sources."""
        settings = get_settings()
        model_ref = parse_model_ref(model_ref_str)

        if isinstance(model_ref, AIRResource):
            if model_ref.source == ModelSource.CIVITAI:
                model_path = await civitai_download(model_ref)
                self.model_name = model_path.stem
                self.pipe = StableDiffusionXLPipeline.from_single_file(
                    str(model_path),
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                )
            else:
                raise ValueError(f"Unsupported AIR source: {model_ref.source}")
        elif isinstance(model_ref, HuggingFaceResource):
            model_path = get_hf_model_path(model_ref)
            self.model_name = model_ref.repo_id.split("/")[-1]
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                token=settings.hf_token,
            )
        elif isinstance(model_ref, URLResource):
            downloaded_path = await download_from_url(model_ref)
            self.model_name = downloaded_path.stem
            self.pipe = StableDiffusionXLPipeline.from_single_file(
                str(downloaded_path),
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
        else:
            raise ValueError(f"Unsupported model reference type: {type(model_ref)}")

    async def _load_vae(self, vae_ref: str | None) -> None:
        """Load VAE from various sources or use model's embedded VAE."""
        if not self.pipe:
            raise RuntimeError("Pipeline not loaded")

        settings = get_settings()

        if vae_ref is None:
            logger.info("Using model's embedded VAE")
            return

        logger.info(f"Loading VAE: {vae_ref}")

        vae_model_ref = parse_model_ref(vae_ref)

        if isinstance(vae_model_ref, AIRResource):
            if vae_model_ref.source == ModelSource.CIVITAI:
                vae_path = await civitai_download(vae_model_ref)
                vae = AutoencoderKL.from_single_file(
                    str(vae_path),
                    torch_dtype=torch.float16,
                )
            else:
                raise ValueError(f"Unsupported VAE AIR source: {vae_model_ref.source}")
        elif isinstance(vae_model_ref, HuggingFaceResource):
            vae = AutoencoderKL.from_pretrained(
                vae_model_ref.repo_id,
                torch_dtype=torch.float16,
                token=settings.hf_token,
            )
        elif isinstance(vae_model_ref, URLResource):
            vae_path = await download_from_url(vae_model_ref)
            vae = AutoencoderKL.from_single_file(
                str(vae_path),
                torch_dtype=torch.float16,
            )
        else:
            raise ValueError(f"Unsupported VAE reference type: {type(vae_model_ref)}")

        self.pipe.vae = vae.to("cuda")
        logger.info("VAE loaded and replaced")

    async def _apply_quantization(self, quantization: QuantizationType) -> None:
        """Apply TorchAO quantization to model."""
        if not self.pipe or quantization == QuantizationType.NONE or self._quantized:
            return

        logger.info(f"Applying TorchAO quantization: {quantization.value}")

        try:
            from torchao.quantization import (
                Int4WeightOnlyConfig,
                Int8WeightOnlyConfig,
                quantize_,
            )

            if quantization == QuantizationType.INT8_WEIGHT_ONLY:
                quantize_(self.pipe.unet, Int8WeightOnlyConfig())
                quantize_(self.pipe.vae, Int8WeightOnlyConfig())
                logger.info("Applied int8 weight-only quantization to UNet and VAE")
            elif quantization == QuantizationType.INT4_WEIGHT_ONLY:
                quantize_(self.pipe.unet, Int4WeightOnlyConfig())
                logger.info("Applied int4 weight-only quantization to UNet")
            elif quantization == QuantizationType.FP8_WEIGHT_ONLY:
                from torchao.quantization import Float8WeightOnlyConfig

                quantize_(self.pipe.unet, Float8WeightOnlyConfig())
                logger.info("Applied fp8 weight-only quantization to UNet")

            self._quantized = True
        except Exception as e:
            logger.warning(f"TorchAO quantization failed: {e}, continuing without quantization")

    async def load_loras(self, lora_refs: list[str]) -> None:
        """Load LoRA weights.

        Args:
            lora_refs: List of LoRA references (AIR URN or URL)
        """
        if not self.pipe:
            raise RuntimeError("Pipeline not loaded")

        for ref in lora_refs:
            ref = ref.strip()
            if not ref:
                continue

            logger.info(f"Loading LoRA: {ref}")

            lora_ref = parse_model_ref(ref)

            if isinstance(lora_ref, AIRResource):
                if lora_ref.source == ModelSource.CIVITAI:
                    lora_path = await civitai_download(lora_ref)
                else:
                    raise ValueError(f"Unsupported LoRA source: {lora_ref.source}")
            elif isinstance(lora_ref, URLResource):
                lora_path = await download_from_url(lora_ref)
            else:
                raise ValueError(f"Unsupported LoRA reference type: {type(lora_ref)}")

            # Load LoRA weights
            adapter_name = lora_path.stem
            self.pipe.load_lora_weights(
                str(lora_path.parent),
                weight_name=lora_path.name,
                adapter_name=adapter_name,
            )
            self.loaded_loras.append(adapter_name)
            logger.info(f"LoRA loaded: {adapter_name}")

    async def generate(
        self,
        params: GenerationParams,
        progress_callback: Callable[[int, str | None], Any] | None = None,
    ) -> SavedImage:
        """Generate image from parameters.

        Args:
            params: Generation parameters
            progress_callback: Optional callback (step_num, preview_base64)

        Returns:
            SavedImage with result
        """
        if not self.pipe:
            raise RuntimeError("Pipeline not loaded")

        logger.info(f"Starting generation: prompt={params.prompt[:50]}, steps={params.steps}")

        # Set scheduler
        scheduler_class = SCHEDULERS.get(params.sampler, EulerDiscreteScheduler)
        self.pipe.scheduler = scheduler_class.from_config(self.pipe.scheduler.config)
        logger.info(f"Scheduler set: {params.sampler}")

        # Determine seed
        seed = params.seed if params.seed is not None else random.randint(0, 2**32 - 1)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        logger.info(f"Seed: {seed}")

        # Simple generation without callback for debugging
        logger.info("Calling pipeline...")
        result = self.pipe(
            prompt=params.prompt,
            negative_prompt=params.negative_prompt,
            width=params.width,
            height=params.height,
            num_inference_steps=params.steps,
            guidance_scale=params.cfg_scale,
            generator=generator,
        )
        logger.info("Pipeline completed")

        image: Image.Image = result.images[0]

        # Save image with metadata
        metadata = ImageMetadata(
            prompt=params.prompt,
            negative_prompt=params.negative_prompt,
            seed=seed,
            steps=params.steps,
            cfg_scale=params.cfg_scale,
            width=params.width,
            height=params.height,
            sampler=params.sampler,
            model_name=self.model_name,
            loras=self.loaded_loras if self.loaded_loras else None,
        )

        saved = save_image(image, metadata)

        return saved


# Global pipeline instance
pipeline = SDXLPipeline()
