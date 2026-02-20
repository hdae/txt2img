from contextlib import contextmanager
import math
import numbers
from pathlib import Path
import re
from typing import Any
import warnings

from diffusers import AutoencoderKLQwenImage, DiffusionPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.utils.hub_utils import _get_model_file
import numpy as np
from PIL import Image
from safetensors.torch import load_file
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3Model

from ...loaders.lora_pipeline import AnimaLoraLoaderMixin
from ...models.transformers.modeling_anima_transformer import (
    AnimaTransformerModel,
    _convert_anima_state_dict_to_diffusers,
)
from .constants import (
    ANIMA_SAMPLING_MULTIPLIER,
    ANIMA_VAE_CONFIG,
    DEFAULT_QWEN_TOKENIZER_REPO,
    DEFAULT_T5_TOKENIZER_REPO,
    DEFAULT_TEXT_ENCODER_CONFIG_REPO,
    DEFAULT_TEXT_ENCODER_WEIGHTS,
    DEFAULT_VAE_REPO,
    DTYPE_MAP,
    DTYPE_NAME_MAP,
    FORGE_BETA_ALPHA,
    FORGE_BETA_BETA,
    HF_URL_PREFIXES,
    LOCAL_QWEN_TOKENIZER_DIR,
    LOCAL_T5_TOKENIZER_DIR,
    QWEN3_06B_CONFIG,
)
from .options import AnimaComponents, AnimaLoaderOptions, AnimaRuntimeOptions
from .pipeline_output import AnimaPipelineOutput

PromptInput = str | list[str] | tuple[str, ...]
GeneratorInput = torch.Generator | list[torch.Generator] | tuple[torch.Generator, ...]


class AnimaPipeline(DiffusionPipeline, AnimaLoraLoaderMixin):
    transformer: AnimaTransformerModel
    vae: AutoencoderKLQwenImage
    scheduler: FlowMatchEulerDiscreteScheduler
    text_encoder: Any
    prompt_tokenizer: Any | None
    execution_device: str
    model_dtype: torch.dtype
    text_encoder_dtype: torch.dtype
    use_module_cpu_offload: bool
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = ["prompt_tokenizer"]
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        *,
        transformer: AnimaTransformerModel,
        vae: AutoencoderKLQwenImage,
        scheduler: FlowMatchEulerDiscreteScheduler,
        text_encoder: Any,
        prompt_tokenizer: Any | None = None,
        execution_device: str = "auto",
        model_dtype: torch.dtype = torch.float32,
        text_encoder_dtype: torch.dtype = torch.float32,
        use_module_cpu_offload: bool = False,
        model_path: str | None = None,
        text_encoder_weights: str = DEFAULT_TEXT_ENCODER_WEIGHTS,
        text_encoder_config_repo: str = DEFAULT_TEXT_ENCODER_CONFIG_REPO,
        qwen_tokenizer_repo: str = DEFAULT_QWEN_TOKENIZER_REPO,
        t5_tokenizer_repo: str = DEFAULT_T5_TOKENIZER_REPO,
        vae_repo: str = DEFAULT_VAE_REPO,
    ):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
        )
        self.register_to_config(
            prompt_tokenizer=(None, None),
            model_path=model_path,
            text_encoder_weights=text_encoder_weights,
            text_encoder_config_repo=text_encoder_config_repo,
            qwen_tokenizer_repo=qwen_tokenizer_repo,
            t5_tokenizer_repo=t5_tokenizer_repo,
            vae_repo=vae_repo,
        )
        self.prompt_tokenizer = prompt_tokenizer
        self.execution_device = execution_device
        self.model_dtype = model_dtype
        self.text_encoder_dtype = text_encoder_dtype
        self.use_module_cpu_offload = use_module_cpu_offload

    def check_inputs(
        self,
        *,
        prompt: PromptInput,
        negative_prompt: PromptInput | None,
        image: Image.Image | np.ndarray | torch.Tensor | list[Image.Image] | tuple[Image.Image, ...] | None,
        mask_image: Image.Image | np.ndarray | torch.Tensor | list[Image.Image] | tuple[Image.Image, ...] | None,
        strength: float,
        width: int,
        height: int,
        num_inference_steps: int,
        num_images_per_prompt: int,
        generator: GeneratorInput | None,
        sampler: str,
        sigma_schedule: str,
        noise_seed_mode: str,
        cfg_batch_mode: str,
        output_type: str,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
    ) -> None:
        prompts, _ = _resolve_prompt_batches(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
        )
        batch_size = len(prompts)
        if strength <= 0.0 or strength > 1.0:
            raise ValueError("`strength` must be in (0.0, 1.0].")
        if width < 16 or height < 16:
            raise ValueError("`width` and `height` must be >= 16.")
        if width % 16 != 0 or height % 16 != 0:
            raise ValueError("`width` and `height` must be divisible by 16.")
        if num_inference_steps < 1:
            raise ValueError("`num_inference_steps` must be >= 1.")
        if image is None and mask_image is not None:
            raise ValueError("`mask_image` requires `image`.")
        if image is None and not math.isclose(strength, 1.0):
            raise ValueError("`strength` can be changed only when `image` is provided.")
        if image is not None and not isinstance(image, (Image.Image, np.ndarray, torch.Tensor, list, tuple)):
            raise ValueError("`image` must be a PIL image, numpy array, torch tensor, or a list/tuple of PIL images.")
        if mask_image is not None and not isinstance(mask_image, (Image.Image, np.ndarray, torch.Tensor, list, tuple)):
            raise ValueError(
                "`mask_image` must be a PIL image, numpy array, torch tensor, or a list/tuple of PIL images."
            )
        if isinstance(image, (list, tuple)):
            if len(image) == 0:
                raise ValueError("`image` list/tuple must not be empty.")
            if not all(isinstance(item, Image.Image) for item in image):
                raise ValueError("`image` list/tuple must contain only PIL.Image.Image.")
        if isinstance(mask_image, (list, tuple)):
            if len(mask_image) == 0:
                raise ValueError("`mask_image` list/tuple must not be empty.")
            if not all(isinstance(item, Image.Image) for item in mask_image):
                raise ValueError("`mask_image` list/tuple must contain only PIL.Image.Image.")
        _normalize_generator(generator, batch_size=batch_size)
        if sampler not in {"flowmatch_euler", "euler", "euler_a_rf", "euler_ancestral_rf"}:
            raise ValueError("`sampler` must be one of: flowmatch_euler, euler, euler_a_rf, euler_ancestral_rf.")
        if sigma_schedule not in {"beta", "uniform", "simple", "normal"}:
            raise ValueError("`sigma_schedule` must be one of: beta, uniform, simple, normal.")
        if sampler == "flowmatch_euler" and sigma_schedule != "uniform":
            raise ValueError("`flowmatch_euler` requires `sigma_schedule='uniform'`.")
        if noise_seed_mode not in {"comfy", "device"}:
            raise ValueError("`noise_seed_mode` must be one of: comfy, device.")
        if cfg_batch_mode not in {"split", "concat"}:
            raise ValueError("`cfg_batch_mode` must be one of: split, concat.")
        if output_type not in {"pil", "np"}:
            raise ValueError("`output_type` must be one of: pil, np.")
        if callback_on_step_end_tensor_inputs is not None:
            invalid = [name for name in callback_on_step_end_tensor_inputs if name not in self._callback_tensor_inputs]
            if invalid:
                raise ValueError(
                    "`callback_on_step_end_tensor_inputs` must be a subset of "
                    f"{self._callback_tensor_inputs}, but got {invalid}."
                )

    @torch.no_grad()
    def __call__(
        self,
        *,
        prompt: PromptInput,
        negative_prompt: PromptInput | None = None,
        image: Image.Image | np.ndarray | torch.Tensor | list[Image.Image] | tuple[Image.Image, ...] | None = None,
        mask_image: Image.Image | np.ndarray | torch.Tensor | list[Image.Image] | tuple[Image.Image, ...] | None = None,
        strength: float = 1.0,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 32,
        num_images_per_prompt: int = 1,
        guidance_scale: float = 4.0,
        seed: int | None = 42,
        generator: GeneratorInput | None = None,
        sampler: str = "euler_a_rf",
        sigma_schedule: str = "beta",
        beta_alpha: float = FORGE_BETA_ALPHA,
        beta_beta: float = FORGE_BETA_BETA,
        eta: float = 1.0,
        s_noise: float = 1.0,
        noise_seed_mode: str = "comfy",
        cfg_batch_mode: str = "split",
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Any | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
        extra_call_kwargs: dict[str, Any] | None = None,
    ) -> AnimaPipelineOutput | tuple[list[Image.Image] | np.ndarray]:
        resolved_callback_tensor_inputs = callback_on_step_end_tensor_inputs
        if resolved_callback_tensor_inputs is None:
            resolved_callback_tensor_inputs = ["latents"]

        self.check_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask_image,
            strength=strength,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            sampler=sampler,
            sigma_schedule=sigma_schedule,
            noise_seed_mode=noise_seed_mode,
            cfg_batch_mode=cfg_batch_mode,
            output_type=output_type,
            callback_on_step_end_tensor_inputs=resolved_callback_tensor_inputs,
        )

        try:
            images = _generate_image(
                self,
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask_image,
                strength=strength,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                guidance_scale=guidance_scale,
                seed=seed,
                generator=generator,
                sampler=sampler,
                sigma_schedule=sigma_schedule,
                beta_alpha=beta_alpha,
                beta_beta=beta_beta,
                eta=eta,
                s_noise=s_noise,
                noise_seed_mode=noise_seed_mode,
                cfg_batch_mode=cfg_batch_mode,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=resolved_callback_tensor_inputs,
                extra_call_kwargs=extra_call_kwargs,
            )
        finally:
            self.maybe_free_model_hooks()

        if output_type == "pil":
            output_images: list[Image.Image] | np.ndarray = images
        else:
            output_images = np.stack([np.asarray(image, dtype=np.uint8) for image in images], axis=0)

        if not return_dict:
            return (output_images,)
        return AnimaPipelineOutput(images=output_images)

    def save_pretrained(self, save_directory: str | Path, **kwargs: Any) -> None:
        super().save_pretrained(save_directory, **kwargs)
        if self.prompt_tokenizer is None:
            return
        _save_prompt_tokenizers_to_local_dir(
            prompt_tokenizer=self.prompt_tokenizer,
            save_directory=Path(save_directory),
        )

    @classmethod
    def _from_pretrained_local_directory(
        cls,
        pretrained_model_name_or_path: str,
        *,
        pipeline_dir: Path,
        kwargs: dict[str, Any],
    ) -> "AnimaPipeline":
        runtime_options = _runtime_options_from_kwargs(kwargs, consume=False)
        load_options = _loader_options_from_kwargs(kwargs, consume=False)

        super_kwargs = dict(kwargs)
        _runtime_options_from_kwargs(super_kwargs, consume=True)

        loaded = super().from_pretrained(pretrained_model_name_or_path, **super_kwargs)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded).__name__}")

        if loaded.prompt_tokenizer is None:
            qwen_source, t5_source, uses_local_tokenizers = _resolve_prompt_tokenizer_sources_for_local_dir(
                pipeline_dir=pipeline_dir,
                pipeline_config=loaded.config,
            )
            tokenizer_options = load_options
            if uses_local_tokenizers:
                tokenizer_options = AnimaLoaderOptions(
                    local_files_only=True,
                    cache_dir=load_options.cache_dir,
                    force_download=False,
                    token=load_options.token,
                    revision=load_options.revision,
                    proxies=load_options.proxies,
                )
            loaded.prompt_tokenizer = _load_prompt_tokenizer(
                qwen_tokenizer_source=qwen_source,
                t5_tokenizer_source=t5_source,
                options=tokenizer_options,
            )

        _apply_runtime_options_to_loaded_pipeline(
            loaded,
            runtime_options=runtime_options,
        )
        return loaded

    @classmethod
    def _from_pretrained_single_file_or_repo(
        cls,
        pretrained_model_name_or_path: str,
        *,
        kwargs: dict[str, Any],
    ) -> "AnimaPipeline":
        components = _components_from_pretrained_kwargs(
            model_path=str(pretrained_model_name_or_path),
            kwargs=kwargs,
        )
        runtime_options = _runtime_options_from_kwargs(kwargs, consume=False)
        load_options = _loader_options_from_kwargs(kwargs, consume=False)
        unknown = _collect_unknown_single_file_from_pretrained_kwargs(kwargs)
        if unknown:
            raise ValueError(f"Unsupported arguments for AnimaPipeline.from_pretrained: {', '.join(unknown)}")

        return _build_anima_pipeline(
            components=components,
            device=runtime_options.device,
            dtype=runtime_options.dtype,
            text_encoder_dtype=runtime_options.text_encoder_dtype,
            local_files_only=load_options.local_files_only,
            cache_dir=load_options.cache_dir,
            force_download=load_options.force_download,
            token=load_options.token,
            revision=load_options.revision,
            proxies=load_options.proxies,
            enable_model_cpu_offload=runtime_options.enable_model_cpu_offload,
            enable_vae_slicing=runtime_options.enable_vae_slicing,
            enable_vae_tiling=runtime_options.enable_vae_tiling,
            enable_vae_xformers=runtime_options.enable_vae_xformers,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any) -> "AnimaPipeline":
        path = Path(str(pretrained_model_name_or_path))
        if path.is_dir() and (path / "model_index.json").exists():
            return cls._from_pretrained_local_directory(
                pretrained_model_name_or_path,
                pipeline_dir=path,
                kwargs=kwargs,
            )
        return cls._from_pretrained_single_file_or_repo(
            pretrained_model_name_or_path,
            kwargs=kwargs,
        )

    @classmethod
    def from_single_file(cls, pretrained_model_link_or_path: str, **kwargs: Any) -> "AnimaPipeline":
        ignored_keys = []
        for key in ("config", "original_config", "original_config_file", "disable_mmap"):
            if key in kwargs:
                ignored_keys.append(key)
                kwargs.pop(key)
        if ignored_keys:
            warnings.warn(
                "Ignoring unsupported from_single_file arguments for AnimaPipeline: "
                + ", ".join(sorted(ignored_keys)),
                stacklevel=2,
            )
        return cls.from_pretrained(pretrained_model_link_or_path, **kwargs)


def _build_simple_sigmas(base_sigmas: torch.Tensor, *, steps: int) -> torch.Tensor:
    """Select sigmas for the simple schedule."""
    if steps < 1:
        raise ValueError("steps must be >= 1")

    stride = len(base_sigmas) / float(steps)
    picked = [float(base_sigmas[-(1 + int(i * stride))].item()) for i in range(steps)]
    picked.append(0.0)
    return torch.tensor(picked, device=base_sigmas.device, dtype=torch.float32)


def _create_seed_generators(
    *,
    seed: int,
    model_device: str,
    mode: str,
) -> tuple[torch.Generator, torch.Generator, str]:
    """Create RNGs for initial noise and sampler steps."""
    if mode == "comfy":
        init_generator = torch.Generator(device="cpu")
        init_generator.manual_seed(seed)

        step_device = model_device if model_device in {"cuda", "cpu"} else "cpu"
        step_seed = seed + 1 if step_device == "cpu" else seed
        step_generator = torch.Generator(device=step_device)
        step_generator.manual_seed(step_seed)
        return init_generator, step_generator, "cpu"

    if mode == "device":
        generator_device = model_device if model_device in {"cuda", "cpu"} else "cpu"
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(seed)
        return generator, generator, generator_device

    raise ValueError(f"Unsupported seed mode: {mode}")


def _normalize_generator(
    generator: GeneratorInput | None,
    *,
    batch_size: int,
) -> torch.Generator | list[torch.Generator] | None:
    if generator is None:
        return None

    if isinstance(generator, (list, tuple)):
        if batch_size < 1:
            raise ValueError(f"`batch_size` must be >= 1, got {batch_size}.")
        if len(generator) != batch_size:
            raise ValueError(
                f"`generator` list length must match batch size ({batch_size}), got {len(generator)}."
            )
        if len(generator) == 0:
            return None
        normalized: list[torch.Generator] = []
        first_device: str | None = None
        for item in generator:
            if not isinstance(item, torch.Generator):
                raise ValueError("`generator` list items must be torch.Generator instances.")
            device_type = item.device.type if hasattr(item, "device") else "cpu"
            if first_device is None:
                first_device = device_type
            elif device_type != first_device:
                raise ValueError(
                    "`generator` list items must be on the same device type. "
                    f"Got mixed devices: {first_device}, {device_type}."
                )
            normalized.append(item)
        return normalized

    if not isinstance(generator, torch.Generator):
        raise ValueError(
            "`generator` must be a torch.Generator or a list/tuple of torch.Generator instances."
        )
    return generator


def _normalize_prompt_list(prompt: PromptInput, *, input_name: str) -> list[str]:
    if isinstance(prompt, str):
        prompts = [prompt]
    elif isinstance(prompt, (list, tuple)):
        prompts = list(prompt)
    else:
        raise ValueError(f"`{input_name}` must be a string or a list/tuple of strings.")

    if len(prompts) == 0:
        raise ValueError(f"`{input_name}` must not be empty.")
    for index, text in enumerate(prompts):
        if not isinstance(text, str):
            raise ValueError(f"`{input_name}`[{index}] must be a string.")
        if input_name == "prompt" and len(text.strip()) == 0:
            raise ValueError("`prompt` entries must be non-empty strings.")
    return prompts


def _resolve_prompt_batches(
    *,
    prompt: PromptInput,
    negative_prompt: PromptInput | None,
    num_images_per_prompt: int,
) -> tuple[list[str], list[str]]:
    prompts = _normalize_prompt_list(prompt, input_name="prompt")
    if num_images_per_prompt < 1:
        raise ValueError("`num_images_per_prompt` must be >= 1.")

    if negative_prompt is None:
        negative_prompts = [""] * len(prompts)
    elif isinstance(negative_prompt, str):
        negative_prompts = [negative_prompt] * len(prompts)
    else:
        negative_prompts = _normalize_prompt_list(negative_prompt, input_name="negative_prompt")
        if len(negative_prompts) != len(prompts):
            raise ValueError(
                "`negative_prompt` list length must match `prompt` list length. "
                f"Got {len(negative_prompts)} and {len(prompts)}."
            )

    batched_prompts: list[str] = []
    batched_negative_prompts: list[str] = []
    for text, neg_text in zip(prompts, negative_prompts):
        for _ in range(num_images_per_prompt):
            batched_prompts.append(text)
            batched_negative_prompts.append(neg_text)
    return batched_prompts, batched_negative_prompts


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _normalize_dtype_name(value: str | torch.dtype, *, name: str) -> str:
    if isinstance(value, str):
        if value not in DTYPE_MAP:
            raise ValueError(f"Unsupported {name}: {value}")
        return value

    mapped = DTYPE_NAME_MAP.get(value)
    if mapped is None:
        raise ValueError(f"Unsupported {name}: {value}")
    return mapped


def _resolve_dtype(dtype: str, device: str) -> torch.dtype:
    mapped = DTYPE_MAP.get(dtype)
    if mapped is not None:
        return mapped
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def _normalize_proxies(proxies: Any) -> dict[str, str] | None:
    if proxies is None:
        return None
    if not isinstance(proxies, dict):
        raise ValueError("`proxies` must be a dictionary with string keys and values.")
    normalized: dict[str, str] = {}
    for key, value in proxies.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("`proxies` must be a dictionary with string keys and values.")
        normalized[key] = value
    return normalized


_ANIMA_COMPONENT_OVERRIDE_KEYS = {
    "text_encoder_weights",
    "text_encoder_config_repo",
    "qwen_tokenizer_repo",
    "t5_tokenizer_repo",
    "vae_repo",
}
_ANIMA_RUNTIME_OPTION_KEYS = {
    "device",
    "dtype",
    "torch_dtype",
    "text_encoder_dtype",
    "enable_model_cpu_offload",
    "enable_vae_slicing",
    "enable_vae_tiling",
    "enable_vae_xformers",
}
_ANIMA_LOADER_OPTION_KEYS = {
    "local_files_only",
    "cache_dir",
    "force_download",
    "token",
    "revision",
    "proxies",
}
_ANIMA_SINGLE_FILE_FROM_PRETRAINED_KEYS = (
    _ANIMA_COMPONENT_OVERRIDE_KEYS | _ANIMA_RUNTIME_OPTION_KEYS | _ANIMA_LOADER_OPTION_KEYS
)


def _loader_options_from_kwargs(kwargs: dict[str, Any], *, consume: bool) -> AnimaLoaderOptions:
    get_value = kwargs.pop if consume else kwargs.get
    cache_dir = get_value("cache_dir", None)
    revision = get_value("revision", None)
    return AnimaLoaderOptions(
        local_files_only=bool(get_value("local_files_only", False)),
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        force_download=bool(get_value("force_download", False)),
        token=get_value("token", None),
        revision=str(revision) if revision is not None else None,
        proxies=_normalize_proxies(get_value("proxies", None)),
    )


def _runtime_options_from_kwargs(kwargs: dict[str, Any], *, consume: bool) -> AnimaRuntimeOptions:
    get_value = kwargs.pop if consume else kwargs.get
    dtype_arg = get_value("dtype", "auto")
    torch_dtype_arg = get_value("torch_dtype", None)
    if torch_dtype_arg is not None:
        if dtype_arg != "auto":
            raise ValueError("Specify only one of `dtype` or `torch_dtype` for custom loading.")
        dtype_arg = torch_dtype_arg
    return AnimaRuntimeOptions(
        device=str(get_value("device", "auto")),
        dtype=_normalize_dtype_name(dtype_arg, name="dtype"),
        text_encoder_dtype=_normalize_dtype_name(
            get_value("text_encoder_dtype", "auto"),
            name="text_encoder_dtype",
        ),
        enable_model_cpu_offload=bool(get_value("enable_model_cpu_offload", False)),
        enable_vae_slicing=bool(get_value("enable_vae_slicing", False)),
        enable_vae_tiling=bool(get_value("enable_vae_tiling", False)),
        enable_vae_xformers=bool(get_value("enable_vae_xformers", False)),
    )


def _components_from_pretrained_kwargs(*, model_path: str, kwargs: dict[str, Any]) -> AnimaComponents:
    return AnimaComponents(
        model_path=model_path,
        text_encoder_weights=str(kwargs.get("text_encoder_weights", DEFAULT_TEXT_ENCODER_WEIGHTS)),
        text_encoder_config_repo=str(kwargs.get("text_encoder_config_repo", DEFAULT_TEXT_ENCODER_CONFIG_REPO)),
        qwen_tokenizer_repo=str(kwargs.get("qwen_tokenizer_repo", DEFAULT_QWEN_TOKENIZER_REPO)),
        t5_tokenizer_repo=str(kwargs.get("t5_tokenizer_repo", DEFAULT_T5_TOKENIZER_REPO)),
        vae_repo=str(kwargs.get("vae_repo", DEFAULT_VAE_REPO)),
    )


def _collect_unknown_single_file_from_pretrained_kwargs(kwargs: dict[str, Any]) -> list[str]:
    return sorted(key for key in kwargs if key not in _ANIMA_SINGLE_FILE_FROM_PRETRAINED_KEYS)


def _extract_hf_repo_id_and_filename(model_url: str) -> tuple[str, str]:
    stripped = model_url
    for prefix in HF_URL_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :]
            break

    match = re.match(r"([^/]+)/([^/]+)/(?:blob/main/)?(.+)", stripped)
    if match is None:
        raise ValueError(
            "URL model_path must be a Hugging Face file URL, for example "
            "'https://huggingface.co/<repo_id>/blob/main/<filename>'."
        )
    repo_id = f"{match.group(1)}/{match.group(2)}"
    filename = match.group(3)
    return repo_id, filename


def _download_model_file_via_diffusers(
    *,
    repo_id: str,
    filename: str,
    options: AnimaLoaderOptions,
) -> str:
    normalized_token = None if options.token is False else options.token
    return _get_model_file(
        repo_id,
        weights_name=filename,
        cache_dir=options.cache_dir,
        force_download=options.force_download,
        proxies=options.proxies,
        local_files_only=options.local_files_only,
        token=normalized_token,
        revision=options.revision,
    )


def _resolve_text_encoder_dtype(
    *,
    model_dtype: torch.dtype,
    text_encoder_dtype: str,
    execution_device: str,
) -> torch.dtype:
    mapped = DTYPE_MAP.get(text_encoder_dtype)
    if text_encoder_dtype != "auto" and mapped is None:
        raise ValueError(f"Unsupported text_encoder_dtype: {text_encoder_dtype}")

    resolved_dtype = model_dtype if mapped is None else mapped
    if execution_device == "cpu" and resolved_dtype == torch.float16:
        return torch.float32
    return resolved_dtype


def _warn_if_unsafe_fp16(*, resolved_device: str, resolved_dtype: torch.dtype) -> None:
    if resolved_device == "cuda" and resolved_dtype == torch.float16:
        warnings.warn(
            "dtype=float16 may cause NaN/Inf with Anima."
            " Use --dtype auto or --dtype bfloat16.",
            stacklevel=2,
        )


def _ensure_finite(tensor: torch.Tensor, *, name: str, runtime_dtype: torch.dtype) -> None:
    if torch.isfinite(tensor).all():
        return

    if runtime_dtype == torch.float16:
        raise RuntimeError(
            f"{name} contains NaN/Inf."
            " dtype=float16 is unstable for this model/environment."
            " Use --dtype auto, bfloat16, or float32."
        )
    raise RuntimeError(f"{name} contains NaN/Inf.")


@contextmanager
def _module_execution_context(
    module: torch.nn.Module,
    *,
    execution_device: str,
    execution_dtype: torch.dtype,
    enable_offload: bool,
) -> Any:
    if enable_offload and execution_device != "cpu":
        module.to(device=execution_device, dtype=execution_dtype)
        try:
            yield
        finally:
            module.to(device="cpu")
            if execution_device == "cuda":
                torch.cuda.empty_cache()
        return

    yield


def _resolve_single_file_path(
    model_path: str,
    *,
    options: AnimaLoaderOptions,
    input_label: str = "model_path",
    allow_remote_url: bool = False,
) -> str:
    if "::" in model_path and not model_path.startswith(("http://", "https://")):
        repo_id, filename = model_path.split("::", maxsplit=1)
        repo_id = repo_id.strip()
        filename = filename.strip()
        if not repo_id or not filename:
            raise ValueError(f"{input_label} in 'repo_id::filename' format requires both repo_id and filename.")
        return _download_model_file_via_diffusers(
            repo_id=repo_id,
            filename=filename,
            options=options,
        )

    path = Path(model_path)
    if path.exists() and path.is_file():
        return str(path)

    if model_path.startswith(("http://", "https://")):
        if not allow_remote_url:
            raise ValueError(
                f"Remote URL source is not supported for {input_label}. "
                "Use local path or 'repo_id::filename'."
            )
        if options.local_files_only:
            raise ValueError(
                f"`local_files_only=True` does not allow remote URL downloads for {input_label}."
            )
        repo_id, filename = _extract_hf_repo_id_and_filename(model_path)
        return _download_model_file_via_diffusers(
            repo_id=repo_id,
            filename=filename,
            options=options,
        )

    if allow_remote_url:
        raise ValueError(
            f"{input_label} must be a local file path, Hugging Face URL, or 'repo_id::filename'."
        )
    raise ValueError(f"{input_label} must be a local file path or 'repo_id::filename'.")


def _strip_net_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if all(key.startswith("net.") for key in state_dict.keys()):
        return {key[4:]: value for key, value in state_dict.items()}
    return dict(state_dict)


def _strip_model_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if all(key.startswith("model.") for key in state_dict.keys()):
        return {key[6:]: value for key, value in state_dict.items()}
    return dict(state_dict)


def _parse_repo_and_subfolder(source: str) -> tuple[str, str | None]:
    if "::" in source:
        repo, subfolder = source.split("::", maxsplit=1)
        repo = repo.strip()
        subfolder = subfolder.strip()
        if repo:
            return repo, (subfolder or None)
    return source, None


def _config_get_string(config: Any, key: str, default: str) -> str:
    value: Any | None = None
    if hasattr(config, "get"):
        value = config.get(key)
    elif isinstance(config, dict):
        value = config.get(key)
    else:
        value = getattr(config, key, None)
    if value is None:
        return default
    return str(value)


def _resolve_prompt_tokenizer_sources_for_local_dir(
    *,
    pipeline_dir: Path,
    pipeline_config: Any,
) -> tuple[str, str, bool]:
    local_qwen = pipeline_dir / LOCAL_QWEN_TOKENIZER_DIR
    local_t5 = pipeline_dir / LOCAL_T5_TOKENIZER_DIR
    if local_qwen.is_dir() and local_t5.is_dir():
        return str(local_qwen), str(local_t5), True

    qwen_source = _config_get_string(pipeline_config, "qwen_tokenizer_repo", DEFAULT_QWEN_TOKENIZER_REPO)
    t5_source = _config_get_string(pipeline_config, "t5_tokenizer_repo", DEFAULT_T5_TOKENIZER_REPO)
    return qwen_source, t5_source, False


def _load_tokenizer_from_source(
    source: str,
    *,
    options: AnimaLoaderOptions,
) -> Any:
    repo_or_path, subfolder = _parse_repo_and_subfolder(source)
    kwargs: dict[str, Any] = {
        "local_files_only": options.local_files_only,
        "cache_dir": options.cache_dir,
        "force_download": options.force_download,
    }
    if subfolder is not None:
        kwargs["subfolder"] = subfolder
    if options.token is not None:
        kwargs["token"] = options.token
    if options.revision is not None:
        kwargs["revision"] = options.revision
    if options.proxies is not None:
        kwargs["proxies"] = options.proxies
    return AutoTokenizer.from_pretrained(repo_or_path, **kwargs)


def _save_prompt_tokenizers_to_local_dir(*, prompt_tokenizer: Any, save_directory: Path) -> None:
    qwen_tokenizer = getattr(prompt_tokenizer, "qwen_tokenizer", None)
    t5_tokenizer = getattr(prompt_tokenizer, "t5_tokenizer", None)

    if qwen_tokenizer is None or not hasattr(qwen_tokenizer, "save_pretrained"):
        warnings.warn("Skipping bundled qwen tokenizer save because save_pretrained is not available.", stacklevel=2)
    else:
        qwen_dir = save_directory / LOCAL_QWEN_TOKENIZER_DIR
        qwen_dir.mkdir(parents=True, exist_ok=True)
        qwen_tokenizer.save_pretrained(str(qwen_dir))

    if t5_tokenizer is None or not hasattr(t5_tokenizer, "save_pretrained"):
        warnings.warn("Skipping bundled t5 tokenizer save because save_pretrained is not available.", stacklevel=2)
    else:
        t5_dir = save_directory / LOCAL_T5_TOKENIZER_DIR
        t5_dir.mkdir(parents=True, exist_ok=True)
        t5_tokenizer.save_pretrained(str(t5_dir))


class _AnimaPromptTokenizer:
    """Minimal prompt tokenizer for Anima.

    This intentionally models only the no-parentheses flow used in current tests:
    all produced token weights are fixed at 1.0.
    """

    def __init__(self, qwen_tokenizer: Any, t5_tokenizer: Any):
        self.qwen_tokenizer = qwen_tokenizer
        self.t5_tokenizer = t5_tokenizer

    def tokenize_with_weights(self, text: str) -> dict[str, list[list[tuple[int, float]]]]:
        qwen_ids = self.qwen_tokenizer(
            [text],
            add_special_tokens=False,
            truncation=False,
            return_tensors="pt",
        ).input_ids[0].tolist()
        t5_ids = self.t5_tokenizer(
            [text],
            add_special_tokens=False,
            truncation=False,
            return_tensors="pt",
        ).input_ids[0].tolist()

        qwen_pad = self.qwen_tokenizer.pad_token_id
        if qwen_pad is None:
            qwen_pad = 151643
        if len(qwen_ids) == 0:
            qwen_ids = [int(qwen_pad)]

        t5_eos = self.t5_tokenizer.eos_token_id
        if t5_eos is None:
            t5_eos = 1
        if len(t5_ids) == 0:
            t5_ids = [int(t5_eos)]
        elif int(t5_ids[-1]) != int(t5_eos):
            t5_ids = [*t5_ids, int(t5_eos)]

        return {
            "qwen3_06b": [[(int(token_id), 1.0) for token_id in qwen_ids]],
            "t5xxl": [[(int(token_id), 1.0) for token_id in t5_ids]],
        }


def _resolve_split_file_path(
    source: str,
    *,
    options: AnimaLoaderOptions,
    component_name: str,
) -> str:
    if source.startswith(("http://", "https://")):
        raise ValueError(
            f"Remote URL source is not supported for {component_name}. "
            "Use local path or 'repo_id::filename'."
        )
    return _resolve_single_file_path(
        source,
        options=options,
        input_label=component_name,
    )


_ANIMA_VAE_RESIDUAL_KEY_MAP = {
    "residual.0.gamma": "norm1.gamma",
    "residual.2.weight": "conv1.weight",
    "residual.2.bias": "conv1.bias",
    "residual.3.gamma": "norm2.gamma",
    "residual.6.weight": "conv2.weight",
    "residual.6.bias": "conv2.bias",
}

_ANIMA_VAE_DECODER_UP_RESNET_MAP = {
    0: "decoder.up_blocks.0.resnets.0",
    1: "decoder.up_blocks.0.resnets.1",
    2: "decoder.up_blocks.0.resnets.2",
    4: "decoder.up_blocks.1.resnets.0",
    5: "decoder.up_blocks.1.resnets.1",
    6: "decoder.up_blocks.1.resnets.2",
    8: "decoder.up_blocks.2.resnets.0",
    9: "decoder.up_blocks.2.resnets.1",
    10: "decoder.up_blocks.2.resnets.2",
    12: "decoder.up_blocks.3.resnets.0",
    13: "decoder.up_blocks.3.resnets.1",
    14: "decoder.up_blocks.3.resnets.2",
}

_ANIMA_VAE_DECODER_UP_UPSAMPLER_MAP = {
    3: "decoder.up_blocks.0.upsamplers.0",
    7: "decoder.up_blocks.1.upsamplers.0",
    11: "decoder.up_blocks.2.upsamplers.0",
}


def _map_residual_tail(tail: str) -> str | None:
    mapped = _ANIMA_VAE_RESIDUAL_KEY_MAP.get(tail)
    if mapped is not None:
        return mapped
    if tail.startswith("shortcut."):
        return "conv_shortcut." + tail.split(".", maxsplit=1)[1]
    if tail.startswith(("resample.", "time_conv.")):
        return tail
    return None


def _convert_anima_vae_key(key: str) -> str:
    if key.startswith("conv1."):
        return "quant_conv." + key.split(".", maxsplit=1)[1]
    if key.startswith("conv2."):
        return "post_quant_conv." + key.split(".", maxsplit=1)[1]

    if key.startswith("encoder.conv1."):
        return "encoder.conv_in." + key.split(".", maxsplit=2)[2]
    if key == "encoder.head.0.gamma":
        return "encoder.norm_out.gamma"
    if key.startswith("encoder.head.2."):
        return "encoder.conv_out." + key.split(".", maxsplit=3)[3]
    if key.startswith("encoder.middle.0."):
        mapped = _ANIMA_VAE_RESIDUAL_KEY_MAP.get(key.removeprefix("encoder.middle.0."))
        if mapped is not None:
            return "encoder.mid_block.resnets.0." + mapped
    if key.startswith("encoder.middle.1."):
        tail = key.removeprefix("encoder.middle.1.")
        if tail == "norm.gamma":
            return "encoder.mid_block.attentions.0.norm.gamma"
        if tail.startswith(("to_qkv.", "proj.")):
            return "encoder.mid_block.attentions.0." + tail
    if key.startswith("encoder.middle.2."):
        mapped = _ANIMA_VAE_RESIDUAL_KEY_MAP.get(key.removeprefix("encoder.middle.2."))
        if mapped is not None:
            return "encoder.mid_block.resnets.1." + mapped
    if key.startswith("encoder.downsamples."):
        rest = key.removeprefix("encoder.downsamples.")
        idx, tail = rest.split(".", maxsplit=1)
        mapped = _map_residual_tail(tail)
        if mapped is not None:
            return f"encoder.down_blocks.{idx}.{mapped}"

    if key.startswith("decoder.conv1."):
        return "decoder.conv_in." + key.split(".", maxsplit=2)[2]
    if key == "decoder.head.0.gamma":
        return "decoder.norm_out.gamma"
    if key.startswith("decoder.head.2."):
        return "decoder.conv_out." + key.split(".", maxsplit=3)[3]
    if key.startswith("decoder.middle.0."):
        mapped = _ANIMA_VAE_RESIDUAL_KEY_MAP.get(key.removeprefix("decoder.middle.0."))
        if mapped is not None:
            return "decoder.mid_block.resnets.0." + mapped
    if key.startswith("decoder.middle.1."):
        tail = key.removeprefix("decoder.middle.1.")
        if tail == "norm.gamma":
            return "decoder.mid_block.attentions.0.norm.gamma"
        if tail.startswith(("to_qkv.", "proj.")):
            return "decoder.mid_block.attentions.0." + tail
    if key.startswith("decoder.middle.2."):
        mapped = _ANIMA_VAE_RESIDUAL_KEY_MAP.get(key.removeprefix("decoder.middle.2."))
        if mapped is not None:
            return "decoder.mid_block.resnets.1." + mapped
    if key.startswith("decoder.upsamples."):
        rest = key.removeprefix("decoder.upsamples.")
        idx_text, tail = rest.split(".", maxsplit=1)
        idx = int(idx_text)
        mapped = _map_residual_tail(tail)
        if mapped is not None and idx in _ANIMA_VAE_DECODER_UP_RESNET_MAP:
            return f"{_ANIMA_VAE_DECODER_UP_RESNET_MAP[idx]}.{mapped}"
        if idx in _ANIMA_VAE_DECODER_UP_UPSAMPLER_MAP and tail.startswith(("resample.", "time_conv.")):
            return f"{_ANIMA_VAE_DECODER_UP_UPSAMPLER_MAP[idx]}.{tail}"

    raise KeyError(key)


def _convert_anima_vae_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if "encoder.conv_in.weight" in state_dict and "quant_conv.weight" in state_dict:
        return dict(state_dict)

    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        mapped_key = _convert_anima_vae_key(key)
        if mapped_key in converted:
            raise RuntimeError(f"Duplicate converted VAE key: {mapped_key}")
        converted[mapped_key] = value
    return converted


def _load_vae_single_file(file_path: str, device: str, dtype: torch.dtype) -> AutoencoderKLQwenImage:
    state_dict = load_file(file_path, device="cpu")
    state_dict = _convert_anima_vae_state_dict(state_dict)

    vae = AutoencoderKLQwenImage.from_config(ANIMA_VAE_CONFIG)
    missing, unexpected = vae.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Single-file VAE does not match expected AutoencoderKLQwenImage architecture. "
            f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )
    vae.to(dtype=dtype)
    vae.eval().requires_grad_(False)
    vae.to(device)
    return vae


def _load_text_encoder(
    *,
    weights_source: str,
    config_repo: str,
    device: str,
    dtype: torch.dtype,
    options: AnimaLoaderOptions,
) -> Any:
    file_path = _resolve_split_file_path(
        weights_source,
        options=options,
        component_name="text_encoder_weights",
    )
    state_dict = load_file(file_path, device="cpu")
    state_dict = _strip_model_prefix(state_dict)

    if config_repo == DEFAULT_TEXT_ENCODER_CONFIG_REPO:
        config = Qwen3Config(**QWEN3_06B_CONFIG)
    else:
        config_kwargs: dict[str, Any] = {
            "local_files_only": options.local_files_only,
            "cache_dir": options.cache_dir,
            "force_download": options.force_download,
        }
        if options.token is not None:
            config_kwargs["token"] = options.token
        if options.revision is not None:
            config_kwargs["revision"] = options.revision
        if options.proxies is not None:
            config_kwargs["proxies"] = options.proxies
        config = Qwen3Config.from_pretrained(config_repo, **config_kwargs)

    text_encoder = Qwen3Model(config)
    missing, unexpected = text_encoder.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Text encoder weights do not match expected Qwen3-0.6B architecture. "
            f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )

    text_encoder.to(dtype=dtype)
    text_encoder.eval().requires_grad_(False)
    text_encoder.to(device=device, dtype=dtype)
    return text_encoder


def _load_prompt_tokenizer(
    *,
    qwen_tokenizer_source: str,
    t5_tokenizer_source: str,
    options: AnimaLoaderOptions,
) -> _AnimaPromptTokenizer:
    qwen_tokenizer = _load_tokenizer_from_source(
        qwen_tokenizer_source,
        options=options,
    )
    t5_tokenizer = _load_tokenizer_from_source(
        t5_tokenizer_source,
        options=options,
    )
    return _AnimaPromptTokenizer(
        qwen_tokenizer=qwen_tokenizer,
        t5_tokenizer=t5_tokenizer,
    )


def _load_transformer_native(model_path: str, device: str, dtype: torch.dtype) -> torch.nn.Module:
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Anima checkpoint not found: {model_path}")

    state_dict = load_file(model_path, device="cpu")
    state_dict = _strip_net_prefix(state_dict)
    core_state_dict, llm_adapter_state_dict = _convert_anima_state_dict_to_diffusers(state_dict)

    transformer = AnimaTransformerModel()
    merged_state = {**core_state_dict, **llm_adapter_state_dict}

    missing, unexpected = transformer.load_state_dict(merged_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Anima checkpoint does not match native transformer architecture. "
            f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )

    transformer.to(dtype=dtype)
    transformer.eval().requires_grad_(False)
    transformer.to(device=device, dtype=dtype)
    return transformer


def _load_vae(
    vae_source: str,
    device: str,
    dtype: torch.dtype,
    options: AnimaLoaderOptions,
) -> AutoencoderKLQwenImage:
    file_path = _resolve_split_file_path(
        vae_source,
        options=options,
        component_name="vae_repo",
    )
    return _load_vae_single_file(file_path, device=device, dtype=dtype)


def _extract_ids_and_weights(token_weight_pairs: list[tuple[Any, float]]) -> tuple[list[int], list[float]]:
    token_ids: list[int] = []
    token_weights: list[float] = []
    for token, weight, *rest in token_weight_pairs:
        del rest
        if not isinstance(token, numbers.Integral):
            raise RuntimeError("Prompt tokenizer returned a non-integer token, which is not supported in this pipeline.")
        token_ids.append(int(token))
        token_weights.append(float(weight))
    return token_ids, token_weights


def _prepare_condition_inputs(
    runtime: AnimaPipeline,
    prompt: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if runtime.prompt_tokenizer is None:
        raise RuntimeError("prompt_tokenizer is not initialized on the pipeline.")
    if len(prompt) == 0:
        raise ValueError("`prompt` batch must not be empty.")

    qwen_pad = runtime.prompt_tokenizer.qwen_tokenizer.pad_token_id
    if qwen_pad is None:
        qwen_pad = 151643
    t5_pad = runtime.prompt_tokenizer.t5_tokenizer.pad_token_id
    if t5_pad is None:
        t5_pad = 0

    qwen_token_batches: list[list[int]] = []
    t5_token_batches: list[list[int]] = []
    t5_weight_batches: list[list[float]] = []
    max_qwen_len = 0
    max_t5_len = 0

    for text in prompt:
        tokenized = runtime.prompt_tokenizer.tokenize_with_weights(text)
        qwen_token_ids, _ = _extract_ids_and_weights(tokenized["qwen3_06b"][0])
        t5_token_ids, t5_token_weights = _extract_ids_and_weights(tokenized["t5xxl"][0])

        if len(qwen_token_ids) == 0:
            qwen_token_ids = [151643]
        if len(t5_token_ids) == 0:
            t5_token_ids = [1]
            t5_token_weights = [1.0]

        qwen_token_batches.append(qwen_token_ids)
        t5_token_batches.append(t5_token_ids)
        t5_weight_batches.append(t5_token_weights)
        max_qwen_len = max(max_qwen_len, len(qwen_token_ids))
        max_t5_len = max(max_t5_len, len(t5_token_ids))

    batch_size = len(prompt)
    qwen_ids = torch.full(
        (batch_size, max_qwen_len),
        int(qwen_pad),
        dtype=torch.long,
        device=runtime.execution_device,
    )
    qwen_mask = torch.zeros((batch_size, max_qwen_len), dtype=torch.long, device=runtime.execution_device)
    t5_ids = torch.full(
        (batch_size, max_t5_len),
        int(t5_pad),
        dtype=torch.int32,
        device=runtime.execution_device,
    )
    t5_weights = torch.zeros((batch_size, max_t5_len, 1), dtype=torch.float32, device=runtime.execution_device)

    for idx, (qwen_ids_item, t5_ids_item, t5_weights_item) in enumerate(
        zip(qwen_token_batches, t5_token_batches, t5_weight_batches)
    ):
        q_len = len(qwen_ids_item)
        t_len = len(t5_ids_item)
        qwen_ids[idx, :q_len] = torch.tensor(qwen_ids_item, dtype=torch.long, device=runtime.execution_device)
        qwen_mask[idx, :q_len] = 1
        t5_ids[idx, :t_len] = torch.tensor(t5_ids_item, dtype=torch.int32, device=runtime.execution_device)
        t5_weights[idx, :t_len, 0] = torch.tensor(t5_weights_item, dtype=torch.float32, device=runtime.execution_device)

    with torch.inference_mode():
        text_encoder_out = runtime.text_encoder(input_ids=qwen_ids, attention_mask=qwen_mask)
        if isinstance(text_encoder_out, tuple):
            qwen_hidden = text_encoder_out[0]
        else:
            qwen_hidden = text_encoder_out.last_hidden_state
        qwen_hidden = qwen_hidden.to(runtime.model_dtype)
    return qwen_hidden, t5_ids, t5_weights


def _build_condition(
    runtime: AnimaPipeline,
    *,
    qwen_hidden: torch.Tensor,
    t5_ids: torch.Tensor,
    t5_weights: torch.Tensor,
) -> torch.Tensor:
    with torch.inference_mode():
        cond = runtime.transformer.preprocess_text_embeds(qwen_hidden, t5_ids, t5xxl_weights=t5_weights)
    if cond.shape[1] < 512:
        cond = torch.nn.functional.pad(cond, (0, 0, 0, 512 - cond.shape[1]))
    return cond


def _latent_hw(height: int, width: int, vae_scale_factor: int = 8) -> tuple[int, int, int, int]:
    step = vae_scale_factor * 2
    height = step * (height // step)
    width = step * (width // step)
    return height, width, height // vae_scale_factor, width // vae_scale_factor


def _reshape_image_tensor_to_bchw(
    image: np.ndarray | torch.Tensor,
    *,
    input_label: str,
) -> torch.Tensor:
    if isinstance(image, np.ndarray):
        tensor = torch.from_numpy(image)
    else:
        tensor = image.detach()

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        if tensor.shape[0] in {1, 3} and tensor.shape[-1] not in {1, 3}:
            tensor = tensor.unsqueeze(0)
        elif tensor.shape[-1] in {1, 3}:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError(
                f"`{input_label}` must have channel size 1 or 3. Got shape {tuple(tensor.shape)}."
            )
    elif tensor.ndim == 4:
        if tensor.shape[1] in {1, 3}:
            pass
        elif tensor.shape[-1] in {1, 3}:
            tensor = tensor.permute(0, 3, 1, 2)
        else:
            raise ValueError(
                f"`{input_label}` must have channel size 1 or 3. Got shape {tuple(tensor.shape)}."
            )
    else:
        raise ValueError(f"`{input_label}` must be 2D/3D/4D. Got shape {tuple(tensor.shape)}.")

    if tensor.shape[0] < 1:
        raise ValueError(f"`{input_label}` batch size must be >= 1. Got {tensor.shape[0]}.")
    return tensor


def _normalize_tensor_to_unit_interval(
    tensor: torch.Tensor,
    *,
    input_label: str,
) -> torch.Tensor:
    tensor = tensor.to(dtype=torch.float32)
    value_min = float(tensor.min().item())
    value_max = float(tensor.max().item())

    if value_min >= 0.0 and value_max <= 1.0:
        normalized = tensor
    elif value_min >= -1.0 and value_max <= 1.0:
        normalized = (tensor + 1.0) / 2.0
    elif value_min >= 0.0 and value_max <= 255.0:
        normalized = tensor / 255.0
    else:
        raise ValueError(
            f"`{input_label}` value range is unsupported: min={value_min:.4f}, max={value_max:.4f}."
        )

    return normalized.clamp(0.0, 1.0)


def _prepare_init_image_tensor(
    image: Image.Image | np.ndarray | torch.Tensor | list[Image.Image] | tuple[Image.Image, ...],
    *,
    width: int,
    height: int,
) -> torch.Tensor:
    if isinstance(image, (list, tuple)):
        if len(image) == 0:
            raise ValueError("`image` list/tuple must not be empty.")
        pil_tensors: list[torch.Tensor] = []
        for item in image:
            if not isinstance(item, Image.Image):
                raise ValueError("`image` list/tuple must contain only PIL.Image.Image.")
            pil_image = item.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
            pil_tensors.append(torch.from_numpy(np.array(pil_image, copy=True)).permute(2, 0, 1).unsqueeze(0))
        tensor = torch.cat(pil_tensors, dim=0)
    elif isinstance(image, Image.Image):
        pil_image = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        tensor = torch.from_numpy(np.array(pil_image, copy=True)).permute(2, 0, 1).unsqueeze(0)
    else:
        tensor = _reshape_image_tensor_to_bchw(image, input_label="image")
        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)
        elif tensor.shape[1] != 3:
            raise ValueError(f"`image` must have 1 or 3 channels, got {tensor.shape[1]}.")

        tensor = _normalize_tensor_to_unit_interval(tensor, input_label="image")
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return tensor.mul(2.0).sub(1.0)

    tensor = _normalize_tensor_to_unit_interval(tensor, input_label="image")
    return tensor.mul(2.0).sub(1.0)


def _prepare_inpaint_mask_tensor(
    mask_image: Image.Image | np.ndarray | torch.Tensor | list[Image.Image] | tuple[Image.Image, ...],
    *,
    width: int,
    height: int,
) -> torch.Tensor:
    if isinstance(mask_image, (list, tuple)):
        if len(mask_image) == 0:
            raise ValueError("`mask_image` list/tuple must not be empty.")
        mask_tensors: list[torch.Tensor] = []
        for item in mask_image:
            if not isinstance(item, Image.Image):
                raise ValueError("`mask_image` list/tuple must contain only PIL.Image.Image.")
            pil_mask = item.convert("L").resize((width, height), Image.Resampling.LANCZOS)
            mask_tensors.append(torch.from_numpy(np.array(pil_mask, copy=True)).unsqueeze(0).unsqueeze(0))
        mask = torch.cat(mask_tensors, dim=0)
    elif isinstance(mask_image, Image.Image):
        pil_mask = mask_image.convert("L").resize((width, height), Image.Resampling.LANCZOS)
        mask = torch.from_numpy(np.array(pil_mask, copy=True)).unsqueeze(0).unsqueeze(0)
    else:
        mask = _reshape_image_tensor_to_bchw(mask_image, input_label="mask_image")
        if mask.shape[1] == 3:
            mask = mask.mean(dim=1, keepdim=True)
        elif mask.shape[1] != 1:
            raise ValueError(f"`mask_image` must have 1 or 3 channels, got {mask.shape[1]}.")
        mask = _normalize_tensor_to_unit_interval(mask, input_label="mask_image")
        mask = torch.nn.functional.interpolate(
            mask,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return mask.clamp(0.0, 1.0)

    mask = _normalize_tensor_to_unit_interval(mask, input_label="mask_image")
    return mask.clamp(0.0, 1.0)


def _retrieve_vae_latents(
    encoder_output: Any,
    *,
    generator: GeneratorInput | None,
) -> torch.Tensor:
    if hasattr(encoder_output, "latent_dist"):
        if generator is None:
            return encoder_output.latent_dist.sample()
        return encoder_output.latent_dist.sample(generator)  # type: ignore[arg-type]
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Failed to retrieve latents from VAE encoder output.")


def _encode_image_to_latents(
    pipe: AnimaPipeline,
    *,
    image_tensor: torch.Tensor,
    generator: GeneratorInput | None,
    sample_dtype: torch.dtype,
) -> torch.Tensor:
    image = image_tensor.to(device=pipe.execution_device, dtype=pipe.model_dtype).unsqueeze(2)
    with torch.inference_mode():
        encoded = pipe.vae.encode(image)
    image_latents = _retrieve_vae_latents(encoded, generator=generator).to(dtype=sample_dtype)

    latents_mean = torch.tensor(
        pipe.vae.config.latents_mean,
        dtype=image_latents.dtype,
        device=image_latents.device,
    ).view(1, 16, 1, 1, 1)
    latents_std = (1.0 / torch.tensor(
        pipe.vae.config.latents_std,
        dtype=image_latents.dtype,
        device=image_latents.device,
    )).view(1, 16, 1, 1, 1)
    return (image_latents - latents_mean) * latents_std


def _align_tensor_batch_size(
    tensor: torch.Tensor,
    *,
    target_batch_size: int,
    input_name: str,
) -> torch.Tensor:
    current_batch_size = int(tensor.shape[0])
    if current_batch_size == target_batch_size:
        return tensor
    if current_batch_size == 1:
        return tensor.repeat(target_batch_size, *([1] * (tensor.ndim - 1)))
    if target_batch_size % current_batch_size == 0:
        repeat_count = target_batch_size // current_batch_size
        return tensor.repeat_interleave(repeat_count, dim=0)
    raise ValueError(
        f"`{input_name}` batch size ({current_batch_size}) is incompatible with prompt batch size ({target_batch_size})."
    )


def _decode_latents(runtime: AnimaPipeline, latents: torch.Tensor) -> list[Image.Image]:
    _ensure_finite(latents, name="latents before decode", runtime_dtype=runtime.model_dtype)
    latents = latents.to(runtime.vae.dtype)
    latents_mean = torch.tensor(runtime.vae.config.latents_mean, dtype=latents.dtype, device=latents.device).view(
        1, 16, 1, 1, 1
    )
    latents_std = torch.tensor(runtime.vae.config.latents_std, dtype=latents.dtype, device=latents.device).view(
        1, 16, 1, 1, 1
    )
    latents = latents * latents_std + latents_mean

    with torch.inference_mode():
        image = runtime.vae.decode(latents, return_dict=False)[0][:, :, 0]

    _ensure_finite(image, name="VAE decode output", runtime_dtype=runtime.model_dtype)
    image = image.float().clamp(-1.0, 1.0)
    image = ((image + 1.0) / 2.0).clamp(0.0, 1.0)
    image_np = image.permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray((item * 255).round().astype("uint8")) for item in image_np]


def _randn_tensor(
    shape: tuple[int, ...],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator | list[torch.Generator] | None,
) -> torch.Tensor:
    target_device = torch.device(device)
    if generator is None:
        return torch.randn(shape, device=target_device, dtype=dtype)

    if isinstance(generator, list):
        if shape[0] != len(generator):
            raise ValueError(
                f"`generator` list length must match tensor batch size ({shape[0]}), got {len(generator)}."
            )
        samples: list[torch.Tensor] = []
        for item_generator in generator:
            sample = _randn_tensor(
                (1, *shape[1:]),
                device=target_device,
                dtype=dtype,
                generator=item_generator,
            )
            samples.append(sample)
        return torch.cat(samples, dim=0)

    generator_device = generator.device.type if hasattr(generator, "device") else target_device.type
    if generator_device == target_device.type:
        return torch.randn(shape, device=target_device, dtype=dtype, generator=generator)

    noise = torch.randn(shape, device=generator.device, dtype=torch.float32, generator=generator)
    return noise.to(device=target_device, dtype=dtype)


def _randn_like(sample: torch.Tensor, generator: torch.Generator | list[torch.Generator] | None) -> torch.Tensor:
    return _randn_tensor(
        tuple(sample.shape),
        device=sample.device,
        dtype=sample.dtype,
        generator=generator,
    )


def _time_snr_shift(alpha: float, t: torch.Tensor) -> torch.Tensor:
    if alpha == 1.0:
        return t
    numerator = torch.mul(t, alpha)
    denominator = torch.add(torch.mul(t, alpha - 1.0), 1.0)
    return torch.div(numerator, denominator)


def _build_beta_sigmas(
    *,
    num_inference_steps: int,
    num_train_timesteps: int,
    shift: float,
    beta_alpha: float,
    beta_beta: float,
    device: str,
) -> torch.Tensor:
    from scipy import stats

    t = (
        torch.arange(1, num_train_timesteps + 1, dtype=torch.float32, device=device)
        / float(num_train_timesteps)
    ) * ANIMA_SAMPLING_MULTIPLIER
    base_sigmas = _time_snr_shift(shift, t)

    total_timesteps = len(base_sigmas) - 1
    ts = 1.0 - np.linspace(0.0, 1.0, num_inference_steps, endpoint=False)
    mapped = stats.beta.ppf(ts, beta_alpha, beta_beta) * float(total_timesteps)
    mapped = np.nan_to_num(mapped, nan=0.0, posinf=float(total_timesteps), neginf=0.0)
    indices = np.clip(np.rint(mapped).astype(np.int64), 0, total_timesteps)

    sigmas: list[float] = []
    last_index: int | None = None
    for index in indices:
        if last_index is None or index != last_index:
            sigmas.append(float(base_sigmas[int(index)].item()))
        last_index = int(index)

    sigmas.append(0.0)
    return torch.tensor(sigmas, device=device, dtype=torch.float32)


def _build_sampling_sigmas(
    pipe: AnimaPipeline,
    *,
    num_inference_steps: int,
    sigma_schedule: str,
    beta_alpha: float,
    beta_beta: float,
) -> torch.Tensor:
    if sigma_schedule == "normal":
        shift = float(pipe.scheduler.config.shift)
        num_train_timesteps = int(pipe.scheduler.config.num_train_timesteps)
        multiplier = float(ANIMA_SAMPLING_MULTIPLIER)

        t = (
            torch.arange(1, num_train_timesteps + 1, dtype=torch.float32, device=pipe.execution_device)
            / float(num_train_timesteps)
        ) * multiplier
        base_sigmas = _time_snr_shift(shift, t / multiplier)
        sigma_min = base_sigmas[0]
        sigma_max = base_sigmas[-1]

        start = sigma_max * multiplier
        end = sigma_min * multiplier

        append_zero = True
        sigma_at_end = _time_snr_shift(shift, end / multiplier)
        if math.isclose(float(sigma_at_end.item()), 0.0, abs_tol=1e-5):
            num_inference_steps += 1
            append_zero = False

        timesteps = torch.linspace(start, end, num_inference_steps, device=pipe.execution_device, dtype=torch.float32)
        sigmas = _time_snr_shift(shift, timesteps / multiplier).to(dtype=torch.float32)
        if append_zero:
            sigmas = torch.cat([sigmas, torch.zeros(1, device=pipe.execution_device, dtype=torch.float32)])
        return sigmas

    if sigma_schedule == "simple":
        t = (
            torch.arange(
                1,
                pipe.scheduler.config.num_train_timesteps + 1,
                dtype=torch.float32,
                device=pipe.execution_device,
            )
            / float(pipe.scheduler.config.num_train_timesteps)
        ) * ANIMA_SAMPLING_MULTIPLIER
        base_sigmas = _time_snr_shift(float(pipe.scheduler.config.shift), t)
        return _build_simple_sigmas(base_sigmas, steps=num_inference_steps)

    if sigma_schedule == "beta":
        return _build_beta_sigmas(
            num_inference_steps=num_inference_steps,
            num_train_timesteps=pipe.scheduler.config.num_train_timesteps,
            shift=float(pipe.scheduler.config.shift),
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            device=pipe.execution_device,
        )

    pipe.scheduler.set_timesteps(num_inference_steps, device=pipe.execution_device)
    return pipe.scheduler.sigmas.to(device=pipe.execution_device, dtype=torch.float32)


def _resolve_strength_start_step(
    *,
    total_steps: int,
    strength: float,
) -> int:
    if total_steps < 1:
        raise ValueError("total_steps must be >= 1")
    init_timestep = min(total_steps * strength, total_steps)
    return int(max(total_steps - init_timestep, 0))


def _trim_flowmatch_timesteps_by_strength(
    pipe: AnimaPipeline,
    *,
    num_inference_steps: int,
    strength: float,
) -> torch.Tensor:
    pipe.scheduler.set_timesteps(num_inference_steps, device=pipe.execution_device)
    timesteps = pipe.scheduler.timesteps

    if math.isclose(strength, 1.0):
        return timesteps

    t_start = _resolve_strength_start_step(total_steps=num_inference_steps, strength=strength)
    begin_index = t_start * int(getattr(pipe.scheduler, "order", 1))
    trimmed = timesteps[begin_index:]
    if hasattr(pipe.scheduler, "set_begin_index"):
        pipe.scheduler.set_begin_index(begin_index)
    if len(trimmed) < 1:
        raise ValueError(
            f"After applying strength={strength}, no denoising steps remain. "
            "Increase `strength` or `num_inference_steps`."
        )
    return trimmed


def _trim_sigmas_by_strength(
    *,
    sigmas: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    if math.isclose(strength, 1.0):
        return sigmas

    total_steps = len(sigmas) - 1
    t_start = _resolve_strength_start_step(total_steps=total_steps, strength=strength)
    trimmed = sigmas[t_start:]
    if len(trimmed) < 2:
        raise ValueError(
            f"After applying strength={strength}, fewer than 1 denoising step remain. "
            "Increase `strength` or `num_inference_steps`."
        )
    return trimmed


def _predict_noise_cfg(
    pipe: AnimaPipeline,
    latents: torch.Tensor,
    *,
    model_timestep: torch.Tensor,
    pos_cond: torch.Tensor,
    neg_cond: torch.Tensor,
    guidance_scale: float,
    cfg_batch_mode: str,
) -> torch.Tensor:
    model_input = latents.to(dtype=pipe.model_dtype)
    timestep = model_timestep.to(device=model_input.device, dtype=torch.float32)
    if timestep.ndim == 0:
        timestep = timestep.expand(model_input.shape[0])

    if cfg_batch_mode == "concat":
        model_input = torch.cat([model_input, model_input], dim=0)
        timestep = torch.cat([timestep, timestep], dim=0)
        context = torch.cat(
            [
                pos_cond.to(device=model_input.device, dtype=pipe.model_dtype),
                neg_cond.to(device=model_input.device, dtype=pipe.model_dtype),
            ],
            dim=0,
        )
        with torch.inference_mode():
            noise_pred = pipe.transformer(
                model_input,
                timestep,
                context=context,
            )
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2, dim=0)
    elif cfg_batch_mode == "split":
        with torch.inference_mode():
            noise_pred_uncond = pipe.transformer(
                model_input,
                timestep,
                context=neg_cond.to(device=model_input.device, dtype=pipe.model_dtype),
            )
            noise_pred_text = pipe.transformer(
                model_input,
                timestep,
                context=pos_cond.to(device=model_input.device, dtype=pipe.model_dtype),
            )
    else:
        raise ValueError("cfg_batch_mode must be one of: split, concat.")

    noise = (noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)).float()
    if pipe.model_dtype == torch.float16:
        _ensure_finite(noise, name="noise prediction", runtime_dtype=pipe.model_dtype)
    return noise


def _predict_denoised_const(
    pipe: AnimaPipeline,
    latents: torch.Tensor,
    *,
    sigma: torch.Tensor,
    pos_cond: torch.Tensor,
    neg_cond: torch.Tensor,
    guidance_scale: float,
    cfg_batch_mode: str,
) -> torch.Tensor:
    model_timestep = (sigma * ANIMA_SAMPLING_MULTIPLIER).expand(latents.shape[0]).float()
    noise_pred = _predict_noise_cfg(
        pipe,
        latents,
        model_timestep=model_timestep,
        pos_cond=pos_cond,
        neg_cond=neg_cond,
        guidance_scale=guidance_scale,
        cfg_batch_mode=cfg_batch_mode,
    )
    return latents - sigma.float() * noise_pred


def _run_step_callback(
    pipe: AnimaPipeline,
    *,
    callback_on_step_end: Any | None,
    callback_on_step_end_tensor_inputs: list[str],
    step_index: int,
    timestep: torch.Tensor,
    latents: torch.Tensor,
) -> torch.Tensor:
    if callback_on_step_end is None:
        return latents

    callback_kwargs: dict[str, Any] = {}
    if "latents" in callback_on_step_end_tensor_inputs:
        callback_kwargs["latents"] = latents

    callback_outputs = callback_on_step_end(pipe, step_index, timestep, callback_kwargs)
    if callback_outputs is None:
        return latents
    if not isinstance(callback_outputs, dict):
        raise TypeError("callback_on_step_end must return dict[str, Any] or None.")

    return callback_outputs.pop("latents", latents)


def _sample_euler_const(
    pipe: AnimaPipeline,
    latents: torch.Tensor,
    *,
    sigmas: torch.Tensor,
    pos_cond: torch.Tensor,
    neg_cond: torch.Tensor,
    guidance_scale: float,
    cfg_batch_mode: str,
    callback_on_step_end: Any | None,
    callback_on_step_end_tensor_inputs: list[str],
    inpaint_mask: torch.Tensor | None = None,
    init_image_latents: torch.Tensor | None = None,
    init_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    if inpaint_mask is not None and (init_image_latents is None or init_noise is None):
        raise ValueError("inpaint sampling requires both `init_image_latents` and `init_noise`.")

    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        denoised = _predict_denoised_const(
            pipe,
            latents,
            sigma=sigma,
            pos_cond=pos_cond,
            neg_cond=neg_cond,
            guidance_scale=guidance_scale,
            cfg_batch_mode=cfg_batch_mode,
        )
        if float(sigma_next.item()) == 0.0:
            latents = denoised
            latents = _run_step_callback(
                pipe,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                step_index=i,
                timestep=sigma,
                latents=latents,
            )
            continue

        d = (latents - denoised) / sigma.to(latents.dtype)
        dt = (sigma_next - sigma).to(latents.dtype)
        latents = latents + d * dt
        if inpaint_mask is not None and init_image_latents is not None and init_noise is not None:
            sigma_next_value = sigma_next.to(init_image_latents.dtype)
            source_latents = sigma_next_value * init_noise + (1.0 - sigma_next_value) * init_image_latents
            latents = (1.0 - inpaint_mask) * source_latents + inpaint_mask * latents
        latents = _run_step_callback(
            pipe,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            step_index=i,
            timestep=sigma,
            latents=latents,
        )
    return latents


def _sample_euler_ancestral_rf_const(
    pipe: AnimaPipeline,
    latents: torch.Tensor,
    *,
    sigmas: torch.Tensor,
    pos_cond: torch.Tensor,
    neg_cond: torch.Tensor,
    guidance_scale: float,
    eta: float,
    s_noise: float,
    generator: torch.Generator | list[torch.Generator] | None,
    cfg_batch_mode: str,
    callback_on_step_end: Any | None,
    callback_on_step_end_tensor_inputs: list[str],
    inpaint_mask: torch.Tensor | None = None,
    init_image_latents: torch.Tensor | None = None,
    init_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    if inpaint_mask is not None and (init_image_latents is None or init_noise is None):
        raise ValueError("inpaint sampling requires both `init_image_latents` and `init_noise`.")

    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        denoised = _predict_denoised_const(
            pipe,
            latents,
            sigma=sigma,
            pos_cond=pos_cond,
            neg_cond=neg_cond,
            guidance_scale=guidance_scale,
            cfg_batch_mode=cfg_batch_mode,
        )
        if float(sigma_next.item()) == 0.0:
            latents = denoised
            latents = _run_step_callback(
                pipe,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                step_index=i,
                timestep=sigma,
                latents=latents,
            )
            continue

        downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta
        sigma_down = sigma_next * downstep_ratio
        alpha_ip1 = 1.0 - sigma_next
        alpha_down = 1.0 - sigma_down
        renoise_sq = sigma_next**2 - sigma_down**2 * alpha_ip1**2 / (alpha_down**2)
        renoise_coeff = renoise_sq.clamp_min(0).sqrt()

        sigma_down_ratio = sigma_down / sigma
        latents = sigma_down_ratio.to(latents.dtype) * latents + (1.0 - sigma_down_ratio).to(
            latents.dtype
        ) * denoised
        if eta > 0:
            noise = _randn_like(latents, generator=generator)
            latents = (alpha_ip1 / alpha_down).to(latents.dtype) * latents + noise * s_noise * renoise_coeff.to(
                latents.dtype
            )
        if inpaint_mask is not None and init_image_latents is not None and init_noise is not None:
            sigma_next_value = sigma_next.to(init_image_latents.dtype)
            source_latents = sigma_next_value * init_noise + (1.0 - sigma_next_value) * init_image_latents
            latents = (1.0 - inpaint_mask) * source_latents + inpaint_mask * latents
        latents = _run_step_callback(
            pipe,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            step_index=i,
            timestep=sigma,
            latents=latents,
        )
    return latents


def _configure_vae_runtime_features(
    vae: AutoencoderKLQwenImage,
    *,
    enable_vae_slicing: bool,
    enable_vae_tiling: bool,
    enable_vae_xformers: bool,
) -> None:
    if enable_vae_slicing:
        if hasattr(vae, "enable_slicing"):
            vae.enable_slicing()
        else:
            warnings.warn("VAE slicing is not supported by this VAE.", stacklevel=2)
    if enable_vae_tiling:
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        else:
            warnings.warn("VAE tiling is not supported by this VAE.", stacklevel=2)
    if enable_vae_xformers:
        if hasattr(vae, "set_use_memory_efficient_attention_xformers"):
            try:
                vae.set_use_memory_efficient_attention_xformers(True)
            except Exception as exc:
                warnings.warn(f"Failed to enable VAE xformers attention: {exc}", stacklevel=2)
        else:
            warnings.warn("VAE xformers is not supported by this VAE.", stacklevel=2)


def _apply_runtime_options_to_loaded_pipeline(
    runtime: AnimaPipeline,
    *,
    runtime_options: AnimaRuntimeOptions,
) -> None:
    resolved_device = _resolve_device(runtime_options.device)
    resolved_dtype = _resolve_dtype(runtime_options.dtype, resolved_device)
    resolved_text_encoder_dtype = _resolve_text_encoder_dtype(
        model_dtype=resolved_dtype,
        text_encoder_dtype=runtime_options.text_encoder_dtype,
        execution_device=resolved_device,
    )
    _warn_if_unsafe_fp16(resolved_device=resolved_device, resolved_dtype=resolved_dtype)

    use_module_cpu_offload = runtime_options.enable_model_cpu_offload
    load_device = "cpu" if use_module_cpu_offload and resolved_device != "cpu" else resolved_device

    runtime.execution_device = resolved_device
    runtime.model_dtype = resolved_dtype
    runtime.text_encoder_dtype = resolved_text_encoder_dtype
    runtime.use_module_cpu_offload = use_module_cpu_offload

    runtime.transformer.to(device=load_device, dtype=resolved_dtype)
    runtime.text_encoder.to(device=load_device, dtype=resolved_text_encoder_dtype)
    runtime.vae.to(device=load_device, dtype=resolved_dtype)

    _configure_vae_runtime_features(
        runtime.vae,
        enable_vae_slicing=runtime_options.enable_vae_slicing,
        enable_vae_tiling=runtime_options.enable_vae_tiling,
        enable_vae_xformers=runtime_options.enable_vae_xformers,
    )


def _build_anima_pipeline(
    components: AnimaComponents,
    *,
    device: str = "auto",
    dtype: str = "auto",
    text_encoder_dtype: str = "auto",
    local_files_only: bool = False,
    cache_dir: str | None = None,
    force_download: bool = False,
    token: str | bool | None = None,
    revision: str | None = None,
    proxies: dict[str, str] | None = None,
    enable_model_cpu_offload: bool = False,
    enable_vae_slicing: bool = False,
    enable_vae_tiling: bool = False,
    enable_vae_xformers: bool = False,
) -> AnimaPipeline:

    resolved_device = _resolve_device(device)
    load_options = AnimaLoaderOptions(
        local_files_only=local_files_only,
        cache_dir=cache_dir,
        force_download=force_download,
        token=token,
        revision=revision,
        proxies=proxies,
    )
    resolved_dtype = _resolve_dtype(dtype, resolved_device)
    resolved_text_encoder_dtype = _resolve_text_encoder_dtype(
        model_dtype=resolved_dtype,
        text_encoder_dtype=text_encoder_dtype,
        execution_device=resolved_device,
    )
    _warn_if_unsafe_fp16(resolved_device=resolved_device, resolved_dtype=resolved_dtype)
    use_module_cpu_offload = enable_model_cpu_offload
    load_device = "cpu" if use_module_cpu_offload and resolved_device != "cpu" else resolved_device
    resolved_model_path = _resolve_single_file_path(
        components.model_path,
        options=load_options,
        allow_remote_url=True,
    )

    transformer = _load_transformer_native(
        model_path=resolved_model_path,
        device=load_device,
        dtype=resolved_dtype,
    )
    vae = _load_vae(
        vae_source=components.vae_repo,
        device=load_device,
        dtype=resolved_dtype,
        options=load_options,
    )
    _configure_vae_runtime_features(
        vae,
        enable_vae_slicing=enable_vae_slicing,
        enable_vae_tiling=enable_vae_tiling,
        enable_vae_xformers=enable_vae_xformers,
    )

    text_encoder = _load_text_encoder(
        weights_source=components.text_encoder_weights,
        config_repo=components.text_encoder_config_repo,
        device=load_device,
        dtype=resolved_text_encoder_dtype,
        options=load_options,
    )
    prompt_tokenizer = _load_prompt_tokenizer(
        qwen_tokenizer_source=components.qwen_tokenizer_repo,
        t5_tokenizer_source=components.t5_tokenizer_repo,
        options=load_options,
    )

    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=3.0,
        use_dynamic_shifting=False,
    )

    runtime = AnimaPipeline(
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        text_encoder=text_encoder,
        prompt_tokenizer=prompt_tokenizer,
        execution_device=resolved_device,
        model_dtype=resolved_dtype,
        text_encoder_dtype=resolved_text_encoder_dtype,
        use_module_cpu_offload=use_module_cpu_offload,
        model_path=components.model_path,
        text_encoder_weights=components.text_encoder_weights,
        text_encoder_config_repo=components.text_encoder_config_repo,
        qwen_tokenizer_repo=components.qwen_tokenizer_repo,
        t5_tokenizer_repo=components.t5_tokenizer_repo,
        vae_repo=components.vae_repo,
    )
    return runtime


def _apply_extra_sampling_kwargs(
    *,
    sampler: str,
    sigma_schedule: str,
    beta_alpha: float,
    beta_beta: float,
    eta: float,
    s_noise: float,
    noise_seed_mode: str,
    cfg_batch_mode: str,
    extra_call_kwargs: dict[str, Any] | None,
) -> tuple[str, str, float, float, float, float, str, str]:
    if not extra_call_kwargs:
        return (
            sampler,
            sigma_schedule,
            beta_alpha,
            beta_beta,
            eta,
            s_noise,
            noise_seed_mode,
            cfg_batch_mode,
        )

    return (
        str(extra_call_kwargs.get("sampler", sampler)),
        str(extra_call_kwargs.get("sigma_schedule", sigma_schedule)),
        float(extra_call_kwargs.get("beta_alpha", beta_alpha)),
        float(extra_call_kwargs.get("beta_beta", beta_beta)),
        float(extra_call_kwargs.get("eta", eta)),
        float(extra_call_kwargs.get("s_noise", s_noise)),
        str(extra_call_kwargs.get("noise_seed_mode", noise_seed_mode)),
        str(extra_call_kwargs.get("cfg_batch_mode", cfg_batch_mode)),
    )


def _resolve_noise_runtime(
    *,
    execution_device: str,
    seed: int | None,
    noise_seed_mode: str,
    generator: GeneratorInput | None,
    batch_size: int,
) -> tuple[
    torch.Generator | list[torch.Generator] | None,
    torch.Generator | list[torch.Generator] | None,
    str,
    torch.dtype,
]:
    if noise_seed_mode not in {"comfy", "device"}:
        raise ValueError("noise_seed_mode must be one of: comfy, device.")

    init_generator: torch.Generator | list[torch.Generator] | None = None
    step_generator: torch.Generator | list[torch.Generator] | None = None
    noise_device = execution_device
    noise_dtype = torch.float32

    provided_generator = _normalize_generator(generator, batch_size=batch_size)
    if provided_generator is not None:
        if isinstance(provided_generator, list):
            generator_device = provided_generator[0].device.type if len(provided_generator) > 0 else "cpu"
        else:
            generator_device = provided_generator.device.type if hasattr(provided_generator, "device") else "cpu"
        return provided_generator, provided_generator, generator_device, noise_dtype

    if noise_seed_mode == "comfy":
        noise_device = "cpu"

    if seed is not None:
        init_generator, step_generator, noise_device = _create_seed_generators(
            seed=seed,
            model_device=execution_device,
            mode=noise_seed_mode,
        )

    return init_generator, step_generator, noise_device, noise_dtype


def _prepare_prompt_embedding_inputs(
    pipe: AnimaPipeline,
    *,
    prompt: list[str],
    negative_prompt: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with _module_execution_context(
        pipe.text_encoder,
        execution_device=pipe.execution_device,
        execution_dtype=pipe.text_encoder_dtype,
        enable_offload=pipe.use_module_cpu_offload,
    ):
        pos_hidden, pos_t5_ids, pos_t5_weights = _prepare_condition_inputs(pipe, prompt)
        neg_hidden, neg_t5_ids, neg_t5_weights = _prepare_condition_inputs(pipe, negative_prompt)

    return pos_hidden, pos_t5_ids, pos_t5_weights, neg_hidden, neg_t5_ids, neg_t5_weights


def _build_cfg_conditions_from_embeddings(
    pipe: AnimaPipeline,
    *,
    pos_hidden: torch.Tensor,
    pos_t5_ids: torch.Tensor,
    pos_t5_weights: torch.Tensor,
    neg_hidden: torch.Tensor,
    neg_t5_ids: torch.Tensor,
    neg_t5_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pos_cond = _build_condition(
        pipe,
        qwen_hidden=pos_hidden,
        t5_ids=pos_t5_ids,
        t5_weights=pos_t5_weights,
    )
    neg_cond = _build_condition(
        pipe,
        qwen_hidden=neg_hidden,
        t5_ids=neg_t5_ids,
        t5_weights=neg_t5_weights,
    )
    return pos_cond, neg_cond


def _sample_flowmatch_euler(
    pipe: AnimaPipeline,
    latents: torch.Tensor,
    *,
    timesteps: torch.Tensor,
    sigma_schedule: str,
    pos_cond: torch.Tensor,
    neg_cond: torch.Tensor,
    guidance_scale: float,
    cfg_batch_mode: str,
    callback_on_step_end: Any | None,
    callback_on_step_end_tensor_inputs: list[str],
    inpaint_mask: torch.Tensor | None = None,
    init_image_latents: torch.Tensor | None = None,
    init_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    if sigma_schedule != "uniform":
        raise ValueError("flowmatch_euler sampler only supports sigma_schedule='uniform'.")
    if inpaint_mask is not None and (init_image_latents is None or init_noise is None):
        raise ValueError("inpaint sampling requires both `init_image_latents` and `init_noise`.")

    for i, timestep in enumerate(timesteps):
        scheduler_timestep = timestep.expand(latents.shape[0]).float()
        model_timestep = (
            scheduler_timestep / float(pipe.scheduler.config.num_train_timesteps)
        ) * ANIMA_SAMPLING_MULTIPLIER

        noise_pred = _predict_noise_cfg(
            pipe,
            latents,
            model_timestep=model_timestep,
            pos_cond=pos_cond,
            neg_cond=neg_cond,
            guidance_scale=guidance_scale,
            cfg_batch_mode=cfg_batch_mode,
        )

        latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
        if inpaint_mask is not None and init_image_latents is not None and init_noise is not None:
            source_latents = init_image_latents
            if i < len(timesteps) - 1:
                next_timestep = timesteps[i + 1].expand(init_image_latents.shape[0]).to(
                    device=init_image_latents.device,
                    dtype=torch.float32,
                )
                source_latents = pipe.scheduler.scale_noise(init_image_latents, next_timestep, init_noise)
            latents = (1.0 - inpaint_mask) * source_latents + inpaint_mask * latents
        latents = _run_step_callback(
            pipe,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            step_index=i,
            timestep=timestep,
            latents=latents,
        )
    return latents


def _sample_const_sigma_samplers(
    pipe: AnimaPipeline,
    latents: torch.Tensor,
    *,
    sigmas: torch.Tensor,
    sampler: str,
    pos_cond: torch.Tensor,
    neg_cond: torch.Tensor,
    guidance_scale: float,
    eta: float,
    s_noise: float,
    generator: torch.Generator | list[torch.Generator] | None,
    cfg_batch_mode: str,
    callback_on_step_end: Any | None,
    callback_on_step_end_tensor_inputs: list[str],
    input_is_noisy_latents: bool = False,
    inpaint_mask: torch.Tensor | None = None,
    init_image_latents: torch.Tensor | None = None,
    init_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    if len(sigmas) < 2:
        raise ValueError("At least 1 denoising step is required.")
    if not input_is_noisy_latents:
        latents = latents * sigmas[0].to(latents.dtype)

    if sampler == "euler":
        return _sample_euler_const(
            pipe,
            latents,
            sigmas=sigmas,
            pos_cond=pos_cond,
            neg_cond=neg_cond,
            guidance_scale=guidance_scale,
            cfg_batch_mode=cfg_batch_mode,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            inpaint_mask=inpaint_mask,
            init_image_latents=init_image_latents,
            init_noise=init_noise,
        )

    if sampler in {"euler_a_rf", "euler_ancestral_rf"}:
        return _sample_euler_ancestral_rf_const(
            pipe,
            latents,
            sigmas=sigmas,
            pos_cond=pos_cond,
            neg_cond=neg_cond,
            guidance_scale=guidance_scale,
            eta=eta,
            s_noise=s_noise,
            generator=generator,
            cfg_batch_mode=cfg_batch_mode,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            inpaint_mask=inpaint_mask,
            init_image_latents=init_image_latents,
            init_noise=init_noise,
        )

    raise ValueError(
        f"Unsupported sampler '{sampler}'. Choose one of: "
        "flowmatch_euler, euler, euler_a_rf."
    )


def _generate_image(
    pipe: AnimaPipeline,
    *,
    prompt: PromptInput,
    negative_prompt: PromptInput | None = None,
    image: Image.Image | np.ndarray | torch.Tensor | list[Image.Image] | tuple[Image.Image, ...] | None = None,
    mask_image: Image.Image | np.ndarray | torch.Tensor | list[Image.Image] | tuple[Image.Image, ...] | None = None,
    strength: float = 1.0,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 32,
    num_images_per_prompt: int = 1,
    guidance_scale: float = 4.0,
    seed: int | None = 42,
    generator: GeneratorInput | None = None,
    sampler: str = "euler_a_rf",
    sigma_schedule: str = "beta",
    beta_alpha: float = FORGE_BETA_ALPHA,
    beta_beta: float = FORGE_BETA_BETA,
    eta: float = 1.0,
    s_noise: float = 1.0,
    noise_seed_mode: str = "comfy",
    cfg_batch_mode: str = "split",
    callback_on_step_end: Any | None = None,
    callback_on_step_end_tensor_inputs: list[str] | None = None,
    extra_call_kwargs: dict[str, Any] | None = None,
) -> list[Image.Image]:
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be >= 1")
    prompts, negative_prompts = _resolve_prompt_batches(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
    )
    batch_size = len(prompts)

    (
        sampler,
        sigma_schedule,
        beta_alpha,
        beta_beta,
        eta,
        s_noise,
        noise_seed_mode,
        cfg_batch_mode,
    ) = _apply_extra_sampling_kwargs(
        sampler=sampler,
        sigma_schedule=sigma_schedule,
        beta_alpha=beta_alpha,
        beta_beta=beta_beta,
        eta=eta,
        s_noise=s_noise,
        noise_seed_mode=noise_seed_mode,
        cfg_batch_mode=cfg_batch_mode,
        extra_call_kwargs=extra_call_kwargs,
    )

    height, width, latent_h, latent_w = _latent_hw(height=height, width=width, vae_scale_factor=8)
    sample_dtype = torch.float32

    resolved_callback_tensor_inputs = callback_on_step_end_tensor_inputs or ["latents"]
    init_generator, step_generator, noise_device, noise_dtype = _resolve_noise_runtime(
        execution_device=pipe.execution_device,
        seed=seed,
        noise_seed_mode=noise_seed_mode,
        generator=generator,
        batch_size=batch_size,
    )
    pos_hidden, pos_t5_ids, pos_t5_weights, neg_hidden, neg_t5_ids, neg_t5_weights = _prepare_prompt_embedding_inputs(
        pipe,
        prompt=prompts,
        negative_prompt=negative_prompts,
    )

    init_image_latents: torch.Tensor | None = None
    inpaint_mask: torch.Tensor | None = None
    init_noise: torch.Tensor | None = None

    if image is not None:
        init_image_tensor = _prepare_init_image_tensor(
            image,
            width=width,
            height=height,
        )
        init_image_tensor = _align_tensor_batch_size(
            init_image_tensor,
            target_batch_size=batch_size,
            input_name="image",
        )
        with _module_execution_context(
            pipe.vae,
            execution_device=pipe.execution_device,
            execution_dtype=pipe.model_dtype,
            enable_offload=pipe.use_module_cpu_offload,
        ):
            init_image_latents = _encode_image_to_latents(
                pipe,
                image_tensor=init_image_tensor,
                generator=init_generator,
                sample_dtype=sample_dtype,
            )
        init_image_latents = init_image_latents.to(device=pipe.execution_device, dtype=sample_dtype)

        if tuple(init_image_latents.shape[-2:]) != (latent_h, latent_w):
            raise RuntimeError(
                "Encoded image latent shape does not match target resolution. "
                f"Expected {(latent_h, latent_w)}, got {tuple(init_image_latents.shape[-2:])}."
            )

        if mask_image is not None:
            mask_tensor = _prepare_inpaint_mask_tensor(
                mask_image,
                width=width,
                height=height,
            )
            mask_latents = torch.nn.functional.interpolate(
                mask_tensor,
                size=(latent_h, latent_w),
                mode="nearest",
            )
            inpaint_mask = mask_latents.to(device=pipe.execution_device, dtype=sample_dtype).unsqueeze(2)
            inpaint_mask = inpaint_mask.repeat(1, init_image_latents.shape[1], 1, 1, 1)
            inpaint_mask = _align_tensor_batch_size(
                inpaint_mask,
                target_batch_size=batch_size,
                input_name="mask_image",
            )

    flowmatch_timesteps: torch.Tensor | None = None
    sigmas: torch.Tensor | None = None
    input_is_noisy_latents = False

    if sampler == "flowmatch_euler":
        flowmatch_timesteps = _trim_flowmatch_timesteps_by_strength(
            pipe,
            num_inference_steps=num_inference_steps,
            strength=strength if init_image_latents is not None else 1.0,
        )
        if init_image_latents is None:
            latents = _randn_tensor(
                (batch_size, 16, 1, latent_h, latent_w),
                device=noise_device,
                dtype=noise_dtype,
                generator=init_generator,
            )
            latents = latents.to(device=pipe.execution_device, dtype=sample_dtype)
        else:
            init_image_latents = _align_tensor_batch_size(
                init_image_latents,
                target_batch_size=batch_size,
                input_name="image",
            )
            init_noise = _randn_tensor(
                tuple(init_image_latents.shape),
                device=noise_device,
                dtype=noise_dtype,
                generator=init_generator,
            ).to(device=pipe.execution_device, dtype=sample_dtype)
            start_timestep = flowmatch_timesteps[:1].expand(init_image_latents.shape[0]).to(
                device=pipe.execution_device,
                dtype=torch.float32,
            )
            latents = pipe.scheduler.scale_noise(init_image_latents, start_timestep, init_noise)
            latents = latents.to(device=pipe.execution_device, dtype=sample_dtype)
    else:
        sigmas = _build_sampling_sigmas(
            pipe,
            num_inference_steps=num_inference_steps,
            sigma_schedule=sigma_schedule,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
        )
        if init_image_latents is None:
            latents = _randn_tensor(
                (batch_size, 16, 1, latent_h, latent_w),
                device=noise_device,
                dtype=noise_dtype,
                generator=init_generator,
            )
            latents = latents.to(device=pipe.execution_device, dtype=sample_dtype)
        else:
            init_image_latents = _align_tensor_batch_size(
                init_image_latents,
                target_batch_size=batch_size,
                input_name="image",
            )
            sigmas = _trim_sigmas_by_strength(
                sigmas=sigmas,
                strength=strength,
            )
            init_noise = _randn_tensor(
                tuple(init_image_latents.shape),
                device=noise_device,
                dtype=noise_dtype,
                generator=init_generator,
            ).to(device=pipe.execution_device, dtype=sample_dtype)
            sigma_start = sigmas[0].to(init_image_latents.dtype)
            latents = sigma_start * init_noise + (1.0 - sigma_start) * init_image_latents
            input_is_noisy_latents = True

    with _module_execution_context(
        pipe.transformer,
        execution_device=pipe.execution_device,
        execution_dtype=pipe.model_dtype,
        enable_offload=pipe.use_module_cpu_offload,
    ):
        pos_cond, neg_cond = _build_cfg_conditions_from_embeddings(
            pipe,
            pos_hidden=pos_hidden,
            pos_t5_ids=pos_t5_ids,
            pos_t5_weights=pos_t5_weights,
            neg_hidden=neg_hidden,
            neg_t5_ids=neg_t5_ids,
            neg_t5_weights=neg_t5_weights,
        )

        if sampler == "flowmatch_euler":
            if flowmatch_timesteps is None:
                raise RuntimeError("Internal error: flowmatch timesteps were not initialized.")
            latents = _sample_flowmatch_euler(
                pipe,
                latents,
                timesteps=flowmatch_timesteps,
                sigma_schedule=sigma_schedule,
                pos_cond=pos_cond,
                neg_cond=neg_cond,
                guidance_scale=guidance_scale,
                cfg_batch_mode=cfg_batch_mode,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=resolved_callback_tensor_inputs,
                inpaint_mask=inpaint_mask,
                init_image_latents=init_image_latents,
                init_noise=init_noise,
            )
        else:
            if sigmas is None:
                raise RuntimeError("Internal error: sigma schedule was not initialized.")
            latents = _sample_const_sigma_samplers(
                pipe,
                latents,
                sigmas=sigmas,
                sampler=sampler,
                pos_cond=pos_cond,
                neg_cond=neg_cond,
                guidance_scale=guidance_scale,
                eta=eta,
                s_noise=s_noise,
                generator=step_generator,
                cfg_batch_mode=cfg_batch_mode,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=resolved_callback_tensor_inputs,
                input_is_noisy_latents=input_is_noisy_latents,
                inpaint_mask=inpaint_mask,
                init_image_latents=init_image_latents,
                init_noise=init_noise,
            )

    with _module_execution_context(
        pipe.vae,
        execution_device=pipe.execution_device,
        execution_dtype=pipe.model_dtype,
        enable_offload=pipe.use_module_cpu_offload,
    ):
        return _decode_latents(pipe, latents)
