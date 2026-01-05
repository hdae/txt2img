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
    VramProfile,
    get_model_config,
    get_settings,
)
from txt2img.core.base_pipeline import BasePipeline
from txt2img.core.image_processor import ImageMetadata, SavedImage, save_image
from txt2img.core.job_queue import GenerationParams
from txt2img.providers.civitai import download_model as civitai_download
from txt2img.providers.huggingface import download_from_url, get_hf_model_path
from txt2img.utils.air_parser import (
    AIRResource,
    HuggingFaceResource,
    ModelSource,
    URLResource,
    parse_model_ref,
)

logger = logging.getLogger(__name__)

# Scheduler mapping
SCHEDULERS = {
    "euler": EulerDiscreteScheduler,
    "euler_a": EulerAncestralDiscreteScheduler,
    "dpm++_2m": DPMSolverMultistepScheduler,
}

# Fixed generation parameters for SDXL
SDXL_FIXED_STEPS = 20



class SDXLPipeline(BasePipeline):
    """SDXL Pipeline manager with model configuration support."""

    def __init__(self) -> None:
        self.pipe: StableDiffusionXLPipeline | None = None
        self._model_name: str = ""
        self.loaded_loras: list[str] = []  # All LoRAs loaded in memory
        self.last_used_loras: set[str] = set()  # LoRAs used in last generation

    @property
    def model_name(self) -> str:
        """Get the loaded model name."""
        return self._model_name

    def get_parameter_schema(self) -> dict[str, Any]:
        """Get JSON Schema for SDXL generation parameters."""
        # Get defaults from ModelConfig
        config = get_model_config()

        return {
            "model_type": "sdxl",
            "prompt_style": "tags",
            # Easy access to defaults from config
            "defaults": {
                "cfg_scale": config.default_cfg,
                "sampler": config.default_sampler,
                "steps": config.default_steps,
            },
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Tag-based prompt (comma-separated)",
                },
                "negative_prompt": {
                    "type": "string",
                    "default": "",
                    "description": "Negative prompt (tags to avoid)",
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
                "cfg_scale": {
                    "type": "number",
                    "default": config.default_cfg,
                    "minimum": 0,
                    "maximum": 15,
                    "step": 0.5,
                    "description": "CFG Scale (prompt adherence)",
                },
                "sampler": {
                    "type": "string",
                    "default": config.default_sampler,
                    "enum": ["euler", "euler_a", "dpm++_2m"],
                    "description": "Sampling method",
                },
                "seed": {
                    "type": ["integer", "null"],
                    "default": None,
                    "description": "Random seed (null for random)",
                },
                "loras": {
                    "type": ["array", "null"],
                    "default": None,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "weight": {
                                "type": "number",
                                "default": 1.0,
                                "minimum": 0.0,
                                "maximum": 2.0,
                            },
                            "trigger_weight": {
                                "type": "number",
                                "default": 0.5,
                                "minimum": 0.0,
                                "maximum": 2.0,
                            },
                        },
                        "required": ["id"],
                    },
                    "description": "LoRAs to apply (get IDs from available_loras)",
                },
            },
            "required": ["prompt"],
            "fixed": {
                "steps": config.default_steps,
            },
        }

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

        # Load VAE if specified
        await self._load_vae(config.vae)

        # Enable memory optimizations
        self.pipe.vae.enable_slicing()

        # Note: force_upcast defaults to True for better quality (float32 VAE decode)
        # Only disable if using fp16-specific VAE like madebyollin/sdxl-vae-fp16-fix

        # Apply VRAM profile optimizations
        from txt2img.config import get_vram_profile

        self._apply_vram_profile(get_vram_profile())

        # LoRAs are loaded on-demand during generation (not at startup)
        # This saves VRAM when LoRAs are not immediately needed

        logger.info(f"Model loaded: {self.model_name}")

    async def _load_sdxl_model(self, model_ref_str: str) -> None:
        """Load SDXL model from various sources."""
        settings = get_settings()
        model_ref = parse_model_ref(model_ref_str)

        if isinstance(model_ref, AIRResource):
            if model_ref.source == ModelSource.CIVITAI:
                model_path = await civitai_download(model_ref)
                self._model_name = model_path.stem
                self.pipe = StableDiffusionXLPipeline.from_single_file(
                    str(model_path),
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                )
            else:
                raise ValueError(f"Unsupported AIR source: {model_ref.source}")
        elif isinstance(model_ref, HuggingFaceResource):
            model_path = get_hf_model_path(model_ref)
            self._model_name = model_ref.repo_id.split("/")[-1]
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                token=settings.hf_token,
            )
        elif isinstance(model_ref, URLResource):
            downloaded_path = await download_from_url(model_ref)
            self._model_name = downloaded_path.stem
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

        self.pipe.vae = vae
        logger.info("VAE loaded and replaced")

    def _apply_vram_profile(self, profile: VramProfile) -> None:
        """Apply VRAM optimization based on profile.

        - FULL: No offloading, maximum speed (24GB+ VRAM for Flux, 12GB+ for SDXL)
        - BALANCED: Model-level CPU offload + VAE tiling (12-16GB for Flux, 8GB for SDXL)
        - LOWVRAM: Group offload with streaming + VAE tiling (8GB for Flux, 6GB for SDXL)
        """
        if not self.pipe:
            return

        logger.info(f"Applying VRAM profile: {profile.value}")

        if profile == VramProfile.FULL:
            # No offloading - move everything to GPU
            self.pipe = self.pipe.to("cuda")
            logger.info("VRAM profile: FULL - no offloading")

        elif profile == VramProfile.BALANCED:
            # Model-level CPU offload + VAE tiling
            self.pipe.enable_model_cpu_offload()
            self.pipe.vae.enable_tiling()
            logger.info("VRAM profile: BALANCED - model offload + VAE tiling")

        elif profile == VramProfile.LOWVRAM:
            # Model CPU offload + VAE tiling + slicing (for SDXL, most memory efficient stable option)
            # Note: sequential_cpu_offload has compatibility issues with from_single_file loading
            self.pipe.enable_model_cpu_offload()
            self.pipe.vae.enable_tiling()
            self.pipe.vae.enable_slicing()
            logger.info("VRAM profile: LOWVRAM - model offload + VAE tiling/slicing")

    def _apply_loras(
        self,
        lora_requests: list[dict] | None,
    ) -> tuple[list[dict], list[tuple[str, float]]]:
        """Apply LoRAs for this generation.

        Must be called BEFORE inference_mode to avoid requires_grad errors.

        Args:
            lora_requests: List of {"id": "...", "weight": 1.0, "trigger_weight": 0.5}

        Returns:
            Tuple of (applied_loras, trigger_info)
            - applied_loras: List of {"name": "...", "weight": 1.0}
            - trigger_info: List of (trigger_words, trigger_weight)
        """
        from txt2img.core.lora_manager import lora_manager

        applied_loras = []
        trigger_info = []

        with torch.no_grad():  # Avoid InferenceMode error
            # No LoRAs requested - disable all
            if not lora_requests:
                if self.loaded_loras:
                    self.pipe.disable_lora()
                    logger.info("Disabled all LoRAs")
                return [], []

            # Load and apply requested LoRAs
            for lora_req in lora_requests:
                lora_id = lora_req.get("id") if isinstance(lora_req, dict) else lora_req
                weight = lora_req.get("weight", 1.0) if isinstance(lora_req, dict) else 1.0
                trigger_weight = (
                    lora_req.get("trigger_weight", 0.5) if isinstance(lora_req, dict) else 0.5
                )

                lora_info = lora_manager.get_lora(lora_id)
                if lora_info and lora_info.path:
                    adapter_name = lora_info.path.stem

                    # Load if not already loaded
                    if adapter_name not in self.loaded_loras:
                        self.pipe.load_lora_weights(
                            str(lora_info.path.parent),
                            weight_name=lora_info.path.name,
                            adapter_name=adapter_name,
                        )
                        self.loaded_loras.append(adapter_name)
                        logger.info(f"LoRA loaded: {adapter_name}")

                    applied_loras.append({"name": adapter_name, "weight": weight})

                    # Collect trigger words
                    if lora_info.trigger_words and trigger_weight > 0:
                        trigger_info.append((", ".join(lora_info.trigger_words), trigger_weight))

            # Set adapter weights
            if applied_loras:
                adapter_names = [lora["name"] for lora in applied_loras]
                adapter_weights = [lora["weight"] for lora in applied_loras]
                self.pipe.set_adapters(adapter_names, adapter_weights)
                logger.info(f"Applied LoRAs: {applied_loras}")
        return applied_loras, trigger_info

    def _cleanup_loras(self, applied_loras: list[dict]) -> None:
        """Unload all LoRAs after generation.

        This ensures a clean state for the next generation, avoiding
        InferenceMode errors when set_adapters() calls requires_grad_(True).

        Args:
            applied_loras: List of LoRAs used in this generation
        """
        # Always unload all LoRAs to avoid InferenceMode issues
        for lora in applied_loras:
            adapter_name = lora["name"]
            if adapter_name in self.loaded_loras:
                try:
                    self.pipe.delete_adapters(adapter_name)
                    self.loaded_loras.remove(adapter_name)
                    logger.info(f"Unloaded LoRA: {adapter_name}")
                except Exception as e:
                    logger.warning(f"Failed to unload LoRA {adapter_name}: {e}")

        self.last_used_loras = set()


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

        # Apply LoRAs (must be done before inference_mode)
        applied_loras, trigger_info = self._apply_loras(params.loras)

        logger.info(f"Starting generation: prompt={params.prompt[:50]}, steps={SDXL_FIXED_STEPS}")

        # Set scheduler from params with Karras sigmas and trailing timesteps for Forge compatibility
        scheduler_class = SCHEDULERS.get(params.sampler, EulerAncestralDiscreteScheduler)
        scheduler_config = {
            **self.pipe.scheduler.config,
            "use_karras_sigmas": True,
            "timestep_spacing": "trailing",
        }
        self.pipe.scheduler = scheduler_class.from_config(scheduler_config)
        logger.info(f"Scheduler set: {params.sampler} (karras=True, trailing)")

        # Determine seed
        # Forge default uses GPU for random generation
        seed = params.seed if params.seed is not None else random.randint(0, 2**32 - 1)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        logger.info(f"Seed: {seed}")

        # Compute embeddings with trigger word support and LPW->Compel conversion
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            full_prompt,  # Normalized prompt with triggers for metadata
        ) = self._compute_embeddings_with_triggers(
            params.prompt,
            params.negative_prompt,
            trigger_info,
        )

        # Run pipeline in thread pool to avoid blocking event loop
        import asyncio
        from diffusers.callbacks import PipelineCallback

        # Get event loop reference for thread-safe callback
        loop = asyncio.get_running_loop()

        # Create callback wrapper for progress updates
        step_callback = None
        if progress_callback:
            class ProgressCallback(PipelineCallback):
                tensor_inputs = []  # No tensors needed

                def callback_fn(self, pipeline, step, timestep, callback_kwargs):
                    # Call the async callback from sync context using thread-safe method
                    asyncio.run_coroutine_threadsafe(
                        progress_callback(step + 1, None),  # step is 0-indexed
                        loop
                    )
                    return callback_kwargs

            step_callback = ProgressCallback()

        def _run_pipeline():
            with torch.inference_mode():
                kwargs = {
                    "prompt_embeds": prompt_embeds,
                    "negative_prompt_embeds": negative_prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
                    "width": params.width,
                    "height": params.height,
                    "num_inference_steps": SDXL_FIXED_STEPS,
                    "guidance_scale": params.cfg_scale,
                    "generator": generator,
                }
                if step_callback:
                    kwargs["callback_on_step_end"] = step_callback
                return self.pipe(**kwargs)

        logger.info("Calling pipeline...")
        result = await asyncio.to_thread(_run_pipeline)
        logger.info("Pipeline completed")

        image: Image.Image = result.images[0]

        # Save metadata with the actual normalized prompt (includes triggers)
        lora_names = [lora["name"] for lora in applied_loras] if applied_loras else None
        metadata = ImageMetadata(
            prompt=full_prompt,
            negative_prompt=params.negative_prompt,
            seed=seed,
            steps=SDXL_FIXED_STEPS,
            cfg_scale=params.cfg_scale,
            width=params.width,
            height=params.height,
            sampler=params.sampler,
            model_name=self.model_name,
            loras=lora_names,
        )

        saved = save_image(image, metadata)

        # Cleanup unused LoRAs
        self._cleanup_loras(applied_loras)

        # Clear CUDA cache to release temporary VRAM
        torch.cuda.empty_cache()

        return saved

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
            full_prompt,
        )

    def _compute_embeddings_with_triggers(
        self,
        prompt: str,
        negative_prompt: str,
        trigger_info: list[tuple[str, float]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
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
                      pooled_prompt_embeds, negative_pooled_prompt_embeds,
                      full_prompt)  # full_prompt is the normalized prompt with triggers
        """
        from compel import CompelForSDXL

        # New refactored imports from utils
        from txt2img.utils.prompt_parser import (
            PromptParser,
            build_compel_prompt,
            prepend_triggers
        )

        # Ensure text encoders are on GPU for Compel (fixes cpu_offload compatibility)
        # Use no_grad to avoid "requires_grad=True on inference tensor" error
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            if hasattr(self.pipe, "text_encoder") and self.pipe.text_encoder is not None:
                self.pipe.text_encoder = self.pipe.text_encoder.to(device)
            if hasattr(self.pipe, "text_encoder_2") and self.pipe.text_encoder_2 is not None:
                self.pipe.text_encoder_2 = self.pipe.text_encoder_2.to(device)

        # CompelForSDXL handles CLIP Skip 2 internally for SDXL
        compel = CompelForSDXL(self.pipe)

        # Prepend trigger words to prompt if any (separated by BREAK)
        # Using generic helper
        full_prompt = prepend_triggers(prompt, trigger_info)

        # Build Compel prompt (handles BREAK splitting, concatenation, and updates AST)
        # Using generic helper
        compel_prompt = build_compel_prompt(full_prompt, mode=PromptParser.LPW, use_concatenation=True)

        if compel_prompt != prompt and compel_prompt != '""':
            logger.info(f"Compel prompt: {compel_prompt[:100]}...")

        # Process negative prompt
        neg_prompt = negative_prompt or ""
        compel_neg_prompt = build_compel_prompt(neg_prompt, mode=PromptParser.LPW, use_concatenation=True)

        # Generate conditioning using CompelForSDXL
        with torch.inference_mode():
            conditioning = compel(compel_prompt, negative_prompt=compel_neg_prompt)

        prompt_embeds = conditioning.embeds
        pooled_prompt_embeds = conditioning.pooled_embeds
        negative_prompt_embeds = conditioning.negative_embeds
        negative_pooled_prompt_embeds = conditioning.negative_pooled_embeds

        # Pad negative to match prompt length if needed
        # (This logic remains here as it's tensor operations, unless we move it to tensor_utils later)
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
            full_prompt,
        )
