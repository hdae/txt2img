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
        self.loaded_loras: list[str] = []  # All LoRAs loaded in memory
        self.last_used_loras: set[str] = set()  # LoRAs used in last generation
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

        # Disable VAE upcast to float32 (saves VRAM, prevents spikes)
        if hasattr(self.pipe.vae.config, "force_upcast"):
            self.pipe.vae.config.force_upcast = False
            logger.info("VAE force_upcast disabled")

        if config.vae_tiling:
            self.pipe.vae.enable_tiling()
            logger.info("VAE tiling enabled")

        # LoRAs are loaded on-demand during generation (not at startup)
        # This saves VRAM when LoRAs are not immediately needed

        # Apply quantization (skip if LoRAs might be used later)
        # WARNING: TorchAO quantization + PEFT LoRA causes black images
        if config.loras and config.quantization != QuantizationType.NONE:
            logger.warning(
                "Quantization disabled: LoRA is configured and quantization causes black images. "
                "See README for details."
            )
        elif config.quantization != QuantizationType.NONE:
            await self._apply_quantization(config.quantization)

        logger.info(f"Model loaded: {self.model_name}")

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
        """Apply TorchAO quantization to UNet only (not VAE)."""
        if not self.pipe or quantization == QuantizationType.NONE or self._quantized:
            return

        logger.info(f"Applying TorchAO quantization: {quantization.value}")

        try:
            from torchao.quantization import quantize_

            if quantization == QuantizationType.INT8_WEIGHT_ONLY:
                # Use version=2 to avoid deprecation warning
                # Only quantize UNet, not VAE (VAE quantization causes black images)
                from torchao.quantization import Int8WeightOnlyConfig

                quantize_(self.pipe.unet, Int8WeightOnlyConfig(version=2))
                logger.info("Applied int8 weight-only quantization to UNet")
            elif quantization == QuantizationType.FP8_WEIGHT_ONLY:
                from torchao.quantization import Float8WeightOnlyConfig

                quantize_(self.pipe.unet, Float8WeightOnlyConfig())
                logger.info("Applied fp8 weight-only quantization to UNet")

            self._quantized = True
        except Exception as e:
            logger.warning(f"TorchAO quantization failed: {e}, continuing without quantization")

    async def load_loras(self, lora_configs: list) -> None:
        """Load LoRA weights.

        Args:
            lora_configs: List of LoraConfig objects
        """
        if not self.pipe:
            raise RuntimeError("Pipeline not loaded")

        for lora_config in lora_configs:
            ref = lora_config.ref.strip()
            if not ref:
                continue

            logger.info(f"Loading LoRA: {ref}")

            try:
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

                # Suppress PEFT warnings during loading
                import warnings

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Already found a `peft_config`")
                    warnings.filterwarnings(
                        "ignore", message="Adapter .* was active which is now deleted"
                    )
                    self.pipe.load_lora_weights(
                        str(lora_path.parent),
                        weight_name=lora_path.name,
                        adapter_name=adapter_name,
                    )
                self.loaded_loras.append(adapter_name)
                logger.info(f"LoRA loaded: {adapter_name}")

            except RuntimeError:
                # Handle incompatible LoRA (wrong architecture, size mismatch, etc.)
                logger.warning(f"Failed to load LoRA {ref}: incompatible with model")
                # Try to clean up partial adapter state
                try:
                    if hasattr(self.pipe, "delete_adapters"):
                        adapter_name = lora_path.stem if "lora_path" in dir() else None
                        if adapter_name:
                            import warnings

                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                self.pipe.delete_adapters(adapter_name)
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"Failed to load LoRA {ref}: {e}")

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

        # Handle LoRA application
        from txt2img.core.lora_manager import lora_manager

        applied_loras = []
        trigger_info = []  # List of (trigger_words, trigger_weight)

        if params.loras:
            # Collect trigger info and apply LoRAs
            for lora_req in params.loras:
                lora_id = lora_req.get("id") if isinstance(lora_req, dict) else lora_req
                weight = lora_req.get("weight", 1.0) if isinstance(lora_req, dict) else 1.0
                trigger_weight = (
                    lora_req.get("trigger_weight", 0.5) if isinstance(lora_req, dict) else 0.5
                )

                lora_info = lora_manager.get_lora(lora_id)
                if lora_info and lora_info.path:
                    # Load LoRA if not already loaded
                    adapter_name = lora_info.path.stem
                    if adapter_name not in self.loaded_loras:
                        self.pipe.load_lora_weights(
                            str(lora_info.path.parent),
                            weight_name=lora_info.path.name,
                            adapter_name=adapter_name,
                        )
                        self.loaded_loras.append(adapter_name)
                        logger.info(f"LoRA loaded: {adapter_name}")

                    applied_loras.append({"name": adapter_name, "weight": weight})

                    # Collect trigger words for separate embedding
                    if lora_info.trigger_words and trigger_weight > 0:
                        trigger_info.append((", ".join(lora_info.trigger_words), trigger_weight))

            # Set LoRA scales for requested LoRAs only
            if applied_loras:
                adapter_names = [lora["name"] for lora in applied_loras]
                adapter_weights = [lora["weight"] for lora in applied_loras]
                self.pipe.enable_lora()  # Re-enable in case it was disabled
                self.pipe.set_adapters(adapter_names, adapter_weights)
                logger.info(f"Applied LoRAs: {applied_loras}")
        else:
            # No LoRAs requested - disable all adapters
            if self.loaded_loras:
                self.pipe.disable_lora()
                logger.info("Disabled all LoRA adapters (none requested)")

        logger.info(f"Starting generation: prompt={params.prompt[:50]}, steps={params.steps}")

        # Set scheduler
        scheduler_class = SCHEDULERS.get(params.sampler, EulerDiscreteScheduler)
        self.pipe.scheduler = scheduler_class.from_config(self.pipe.scheduler.config)
        logger.info(f"Scheduler set: {params.sampler}")

        # Determine seed
        seed = params.seed if params.seed is not None else random.randint(0, 2**32 - 1)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        logger.info(f"Seed: {seed}")

        # Compute embeddings with trigger word support and LPW->Compel conversion
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self._compute_embeddings_with_triggers(
            params.prompt,
            params.negative_prompt,
            trigger_info,
        )

        # Run pipeline in thread pool to avoid blocking event loop
        import asyncio

        def _run_pipeline():
            with torch.inference_mode():
                return self.pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    width=params.width,
                    height=params.height,
                    num_inference_steps=params.steps,
                    guidance_scale=params.cfg_scale,
                    generator=generator,
                )

        logger.info("Calling pipeline...")
        result = await asyncio.to_thread(_run_pipeline)
        logger.info("Pipeline completed")

        image: Image.Image = result.images[0]

        # Save image with metadata (include trigger info in saved prompt)
        saved_prompt = params.prompt
        if trigger_info:
            triggers_str = ", ".join([t[0] for t in trigger_info])
            saved_prompt = f"[Triggers: {triggers_str}] {params.prompt}"

        lora_names = [lora["name"] for lora in applied_loras] if applied_loras else None
        metadata = ImageMetadata(
            prompt=saved_prompt,
            negative_prompt=params.negative_prompt,
            seed=seed,
            steps=params.steps,
            cfg_scale=params.cfg_scale,
            width=params.width,
            height=params.height,
            sampler=params.sampler,
            model_name=self.model_name,
            loras=lora_names,
        )

        saved = save_image(image, metadata)

        # Track current LoRA usage and unload unused ones
        current_loras = {lora["name"] for lora in applied_loras} if applied_loras else set()

        if current_loras != self.last_used_loras:
            # Unload LoRAs that were used last time but not this time
            loras_to_unload = self.last_used_loras - current_loras
            for adapter_name in loras_to_unload:
                if adapter_name in self.loaded_loras:
                    try:
                        self.pipe.delete_adapters(adapter_name)
                        self.loaded_loras.remove(adapter_name)
                        logger.info(f"Unloaded LoRA: {adapter_name}")
                    except Exception as e:
                        logger.warning(f"Failed to unload LoRA {adapter_name}: {e}")

            self.last_used_loras = current_loras

        # Clear CUDA cache to release temporary VRAM
        torch.cuda.empty_cache()

        return saved

    def _compute_embeddings_with_triggers(
        self,
        prompt: str,
        negative_prompt: str,
        trigger_info: list[tuple[str, float]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute embeddings with BREAK segment support and LPW->Compel conversion.

        Supports both LPW (A1111) and Compel syntax in a unified way:
        - LPW syntax (word:1.5) is converted to Compel syntax (word)1.5
        - BREAK keyword splits prompt into segments that are concatenated
        - Triggers are prepended as a separate segment

        Args:
            prompt: User prompt (LPW or Compel syntax)
            negative_prompt: Negative prompt
            trigger_info: List of (trigger_words, weight) tuples

        Returns:
            Tuple of (prompt_embeds, negative_prompt_embeds,
                      pooled_prompt_embeds, negative_pooled_prompt_embeds)
        """
        import re
        import warnings

        from txt2img.core.prompt_parser import convert_a1111_to_compel

        # Suppress Compel deprecation warnings
        warnings.filterwarnings("ignore", message=".*passing multiple tokenizers.*")
        warnings.filterwarnings("ignore", message=".*Token indices sequence length.*")

        from compel import Compel, ReturnedEmbeddingsType

        compel = Compel(
            tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
            text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            truncate_long_prompts=False,
        )

        # Split prompt by BREAK keyword
        segments = re.split(r"\bBREAK\b", prompt, flags=re.IGNORECASE)
        segments = [s.strip() for s in segments if s.strip()]

        if not segments:
            segments = [""]

        # Convert each segment from LPW to Compel syntax and compute embeddings
        segment_embeds = []
        pooled = None

        with torch.inference_mode():
            for segment in segments:
                # Convert LPW syntax to Compel syntax
                compel_segment = convert_a1111_to_compel(segment)
                emb, pool = compel(compel_segment)
                segment_embeds.append(emb)
                if pooled is None:
                    pooled = pool

        # Concatenate all segment embeddings
        if len(segment_embeds) > 1:
            prompt_embeds = torch.cat(segment_embeds, dim=1)
            logger.info(
                f"Concatenated {len(segment_embeds)} BREAK segments, total tokens: {prompt_embeds.shape[1]}"
            )
        else:
            prompt_embeds = segment_embeds[0]

        pooled_prompt_embeds = pooled

        # Prepend trigger embeddings if any
        if trigger_info:
            trigger_parts = []
            with torch.inference_mode():
                for trigger_words, weight in trigger_info:
                    if weight > 0:
                        # Convert trigger words too (in case they have weights)
                        compel_trigger = convert_a1111_to_compel(trigger_words)
                        trigger_emb, _ = compel(compel_trigger)
                        trigger_parts.append(trigger_emb * weight)

            if trigger_parts:
                if len(trigger_parts) > 1:
                    total_trigger = torch.stack(trigger_parts).mean(dim=0)
                else:
                    total_trigger = trigger_parts[0]

                prompt_embeds = torch.cat([total_trigger, prompt_embeds], dim=1)
                logger.info(
                    f"Prepended trigger embeddings: {len(trigger_info)} triggers, "
                    f"total tokens: {prompt_embeds.shape[1]}"
                )

        # Process negative prompt (with BREAK support)
        neg_segments = re.split(r"\bBREAK\b", negative_prompt or "", flags=re.IGNORECASE)
        neg_segments = [s.strip() for s in neg_segments if s.strip()]

        if neg_segments:
            neg_embeds = []
            neg_pooled = None
            with torch.inference_mode():
                for seg in neg_segments:
                    compel_seg = convert_a1111_to_compel(seg)
                    emb, pool = compel(compel_seg)
                    neg_embeds.append(emb)
                    if neg_pooled is None:
                        neg_pooled = pool
            negative_prompt_embeds = (
                torch.cat(neg_embeds, dim=1) if len(neg_embeds) > 1 else neg_embeds[0]
            )
            negative_pooled_prompt_embeds = neg_pooled
        else:
            with torch.inference_mode():
                negative_prompt_embeds, negative_pooled_prompt_embeds = compel("")

        # Pad negative to match prompt length
        if negative_prompt_embeds.shape[1] != prompt_embeds.shape[1]:
            if negative_prompt_embeds.shape[1] < prompt_embeds.shape[1]:
                padding = torch.zeros(
                    1,
                    prompt_embeds.shape[1] - negative_prompt_embeds.shape[1],
                    negative_prompt_embeds.shape[2],
                    device=negative_prompt_embeds.device,
                    dtype=negative_prompt_embeds.dtype,
                )
                negative_prompt_embeds = torch.cat([negative_prompt_embeds, padding], dim=1)
            else:
                negative_prompt_embeds = negative_prompt_embeds[:, : prompt_embeds.shape[1], :]
            logger.info(f"Padded negative embeddings: {negative_prompt_embeds.shape[1]} tokens")

        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )


# Global pipeline instance
pipeline = SDXLPipeline()
