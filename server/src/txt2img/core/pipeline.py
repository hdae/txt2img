"""SDXL Pipeline with torch.compile support."""

import logging
import random
from collections.abc import Callable
from typing import Any

import torch
from diffusers import (
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
)
from PIL import Image

from txt2img.config import get_settings
from txt2img.core.image_processor import ImageMetadata, SavedImage, save_image
from txt2img.core.job_queue import GenerationParams
from txt2img.core.prompt_parser import ParsedPrompt
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
    """SDXL Pipeline manager with torch.compile support."""

    def __init__(self) -> None:
        self.pipe: StableDiffusionXLPipeline | None = None
        self.model_name: str = ""
        self.loaded_loras: list[str] = []
        self._compiled = False

    async def load_model(self) -> None:
        """Load model from configuration."""
        settings = get_settings()

        logger.info(f"Loading model: {settings.model}")

        model_ref = parse_model_ref(settings.model)

        # Determine model path based on reference type
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

        # Move to GPU
        self.pipe = self.pipe.to("cuda")

        # Replace VAE with FP16-fixed version
        from diffusers import AutoencoderKL

        logger.info("Loading madebyollin/sdxl-vae-fp16-fix...")
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
            token=settings.hf_token,
        ).to("cuda")
        self.pipe.vae = vae

        # Enable memory optimizations (new API)
        self.pipe.vae.enable_slicing()

        if settings.vae_tiling:
            self.pipe.vae.enable_tiling()
            logger.info("VAE tiling enabled")

        # Apply TorchAO quantization if enabled
        from txt2img.config import QuantizationType

        if settings.quantization != QuantizationType.NONE and not self._compiled:
            logger.info(f"Applying TorchAO quantization: {settings.quantization.value}")
            try:
                from torchao.quantization import (
                    Int4WeightOnlyConfig,
                    Int8WeightOnlyConfig,
                    quantize_,
                )

                if settings.quantization == QuantizationType.INT8_WEIGHT_ONLY:
                    quantize_(self.pipe.unet, Int8WeightOnlyConfig())
                    quantize_(self.pipe.vae, Int8WeightOnlyConfig())
                    logger.info("Applied int8 weight-only quantization to UNet and VAE")
                elif settings.quantization == QuantizationType.INT4_WEIGHT_ONLY:
                    quantize_(self.pipe.unet, Int4WeightOnlyConfig())
                    logger.info("Applied int4 weight-only quantization to UNet")
                elif settings.quantization == QuantizationType.FP8_WEIGHT_ONLY:
                    from torchao.quantization import Float8WeightOnlyConfig

                    quantize_(self.pipe.unet, Float8WeightOnlyConfig())
                    logger.info("Applied fp8 weight-only quantization to UNet")

                self._compiled = True  # Reusing flag to prevent re-quantization
            except Exception as e:
                logger.warning(f"TorchAO quantization failed: {e}, continuing without quantization")

        logger.info(f"Model loaded: {self.model_name}")

        # Load LoRAs if configured
        if settings.lora:
            await self.load_loras(settings.lora.split(","))

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

    def _encode_prompt_legacy(
        self,
        parsed: ParsedPrompt,
        negative_parsed: ParsedPrompt,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode prompt using legacy weighted encoding.

        Args:
            parsed: Parsed positive prompt
            negative_parsed: Parsed negative prompt

        Returns:
            Tuple of (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
        """
        if not self.pipe:
            raise RuntimeError("Pipeline not loaded")

        device = self.pipe.device
        dtype = self.pipe.text_encoder.dtype

        def encode_weighted_chunk(chunks: list) -> tuple[torch.Tensor, torch.Tensor]:
            """Encode a list of weighted chunks and combine them."""
            all_embeds_1 = []
            all_embeds_2 = []
            all_weights = []

            for chunk in chunks:
                # Encode each part separately
                for part in chunk:
                    # Get embeddings from both text encoders
                    tokens_1 = self.pipe.tokenizer(
                        part.text,
                        padding="max_length",
                        max_length=self.pipe.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids.to(device)

                    tokens_2 = self.pipe.tokenizer_2(
                        part.text,
                        padding="max_length",
                        max_length=self.pipe.tokenizer_2.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids.to(device)

                    with torch.no_grad():
                        embeds_1 = self.pipe.text_encoder(tokens_1, output_hidden_states=True)
                        embeds_2 = self.pipe.text_encoder_2(tokens_2, output_hidden_states=True)

                    all_embeds_1.append(embeds_1.hidden_states[-2])
                    all_embeds_2.append(embeds_2.hidden_states[-2])
                    all_weights.append(part.weight)

            # Weighted average of embeddings
            if all_embeds_1:
                weights_tensor = torch.tensor(all_weights, device=device, dtype=dtype)
                weights_tensor = weights_tensor / weights_tensor.sum()

                prompt_embeds_1 = sum(
                    e * w for e, w in zip(all_embeds_1, weights_tensor, strict=False)
                )
                prompt_embeds_2 = sum(
                    e * w for e, w in zip(all_embeds_2, weights_tensor, strict=False)
                )

                # Get pooled output from last encoder
                pooled = all_embeds_2[-1] if all_embeds_2 else None
            else:
                prompt_embeds_1 = torch.zeros((1, 77, 768), device=device, dtype=dtype)
                prompt_embeds_2 = torch.zeros((1, 77, 1280), device=device, dtype=dtype)
                pooled = None

            combined = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
            return combined, pooled

        # Encode positive and negative prompts
        prompt_embeds, pooled_prompt = encode_weighted_chunk(parsed.chunks)
        negative_embeds, neg_pooled = encode_weighted_chunk(negative_parsed.chunks)

        # Handle pooled outputs
        if pooled_prompt is None:
            pooled_prompt = torch.zeros((1, 1280), device=device, dtype=dtype)
        else:
            # Get the actual pooled output
            pooled_prompt = self.pipe.text_encoder_2(
                self.pipe.tokenizer_2(
                    parsed.chunks[0][0].text if parsed.chunks else "",
                    padding="max_length",
                    max_length=self.pipe.tokenizer_2.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(device)
            ).text_embeds

        if neg_pooled is None:
            neg_pooled = torch.zeros((1, 1280), device=device, dtype=dtype)
        else:
            neg_pooled = self.pipe.text_encoder_2(
                self.pipe.tokenizer_2(
                    negative_parsed.chunks[0][0].text if negative_parsed.chunks else "",
                    padding="max_length",
                    max_length=self.pipe.tokenizer_2.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(device)
            ).text_embeds

        return prompt_embeds, negative_embeds, pooled_prompt, neg_pooled

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
