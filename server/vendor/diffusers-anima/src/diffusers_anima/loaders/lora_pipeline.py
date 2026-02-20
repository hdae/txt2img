from __future__ import annotations

import os
import re
from typing import Callable

import torch
from huggingface_hub.utils import validate_hf_hub_args

from diffusers.loaders.lora_base import LoraBaseMixin, _fetch_state_dict
from diffusers.utils import USE_PEFT_BACKEND, is_peft_version, logging


logger = logging.get_logger(__name__)

_LOW_CPU_MEM_USAGE_DEFAULT_LORA = False


_ANIMA_BLOCK_MODULE_MAP = {
    "self_attn.q_proj": "attn1.to_q",
    "self_attn.k_proj": "attn1.to_k",
    "self_attn.v_proj": "attn1.to_v",
    "self_attn.output_proj": "attn1.to_out.0",
    "cross_attn.q_proj": "attn2.to_q",
    "cross_attn.k_proj": "attn2.to_k",
    "cross_attn.v_proj": "attn2.to_v",
    "cross_attn.output_proj": "attn2.to_out.0",
    "mlp.layer1": "ff.net.0.proj",
    "mlp.layer2": "ff.net.2",
    "adaln_modulation_self_attn.1": "norm1.linear_1",
    "adaln_modulation_self_attn.2": "norm1.linear_2",
    "adaln_modulation_cross_attn.1": "norm2.linear_1",
    "adaln_modulation_cross_attn.2": "norm2.linear_2",
    "adaln_modulation_mlp.1": "norm3.linear_1",
    "adaln_modulation_mlp.2": "norm3.linear_2",
}

_ANIMA_ROOT_MODULE_MAP = {
    "x_embedder.proj.1": "core.patch_embed.proj",
    "t_embedder.1.linear_1": "core.time_embed.t_embedder.linear_1",
    "t_embedder.1.linear_2": "core.time_embed.t_embedder.linear_2",
    "final_layer.adaln_modulation.1": "core.norm_out.linear_1",
    "final_layer.adaln_modulation.2": "core.norm_out.linear_2",
    "final_layer.linear": "core.proj_out",
}

_LORA_PARAM_SUFFIXES = (
    "lora_A.weight",
    "lora_B.weight",
    "lora_A.bias",
    "lora_B.bias",
    "lora_down.weight",
    "lora_up.weight",
    "lora_down.bias",
    "lora_up.bias",
    "alpha",
)


def _normalize_lora_unet_path(path: str) -> str:
    protected = (
        "adaln_modulation_self_attn",
        "adaln_modulation_cross_attn",
        "adaln_modulation_mlp",
        "self_attn",
        "cross_attn",
        "output_proj",
        "q_proj",
        "k_proj",
        "v_proj",
    )
    placeholders = {item: f"__{idx}__" for idx, item in enumerate(protected)}

    normalized = path
    for key, token in placeholders.items():
        normalized = normalized.replace(key, token)
    normalized = normalized.replace("_", ".")
    for key, token in placeholders.items():
        normalized = normalized.replace(token, key)
    return normalized


def _split_lora_key_suffix(key: str) -> tuple[str, str]:
    for suffix in _LORA_PARAM_SUFFIXES:
        marker = f".{suffix}"
        if key.endswith(marker):
            return key[: -len(marker)], suffix
    raise ValueError(f"Unsupported LoRA key: {key}")


def _map_anima_module_path(raw_path: str) -> str:
    path = raw_path
    if path.startswith("transformer."):
        path = path.removeprefix("transformer.")

    if path.startswith("diffusion_model."):
        path = path.removeprefix("diffusion_model.")
    elif path.startswith("model."):
        path = path.removeprefix("model.")
    elif path.startswith("lora_unet_"):
        path = _normalize_lora_unet_path(path.removeprefix("lora_unet_"))

    if path.startswith("transformer_blocks."):
        return f"core.{path}"
    if path.startswith("core.") or path.startswith("llm_adapter."):
        return path

    match = re.match(r"^blocks\.(\d+)\.(.+)$", path)
    if match is not None:
        index = match.group(1)
        tail = match.group(2)
        mapped_tail = _ANIMA_BLOCK_MODULE_MAP.get(tail)
        if mapped_tail is None:
            raise ValueError(f"Unsupported Anima LoRA module path: {raw_path}")
        return f"core.transformer_blocks.{index}.{mapped_tail}"

    mapped_root = _ANIMA_ROOT_MODULE_MAP.get(path)
    if mapped_root is not None:
        return mapped_root

    raise ValueError(f"Unsupported Anima LoRA module path: {raw_path}")


def _to_float_alpha(alpha_value: torch.Tensor | float) -> float:
    if torch.is_tensor(alpha_value):
        return float(alpha_value.item())
    return float(alpha_value)


def _alpha_scales_from_down_weight(*, down_weight: torch.Tensor, alpha_value: float) -> tuple[float, float]:
    rank = down_weight.shape[0]
    if rank < 1:
        raise ValueError("Invalid LoRA down weight rank.")

    scale = alpha_value / float(rank)
    scale_down = scale
    scale_up = 1.0
    while scale_down * 2.0 < scale_up:
        scale_down *= 2.0
        scale_up /= 2.0
    return scale_down, scale_up


def _collect_lora_module_payloads(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, dict[str, torch.Tensor | float]]:
    payloads: dict[str, dict[str, torch.Tensor | float]] = {}
    for original_key, value in state_dict.items():
        if "dora_scale" in original_key:
            continue

        key = original_key.replace("default.", "")
        module_path, suffix = _split_lora_key_suffix(key)
        mapped_path = _map_anima_module_path(module_path)
        mapped_entry = payloads.setdefault(mapped_path, {})

        if suffix == "alpha":
            mapped_entry[suffix] = _to_float_alpha(value)
        else:
            mapped_entry[suffix] = value

    return payloads


def _write_direct_lora_payload(
    *,
    module_path: str,
    payload: dict[str, torch.Tensor | float],
    converted: dict[str, torch.Tensor],
) -> bool:
    has_direct = "lora_A.weight" in payload or "lora_B.weight" in payload
    if not has_direct:
        return False

    for suffix in ("lora_A.weight", "lora_B.weight", "lora_A.bias", "lora_B.bias"):
        tensor = payload.get(suffix)
        if tensor is None:
            continue
        if not torch.is_tensor(tensor):
            raise ValueError(f"Invalid LoRA tensor payload: {module_path}.{suffix}")
        converted[f"transformer.{module_path}.{suffix}"] = tensor
    return True


def _write_down_up_lora_payload(
    *,
    module_path: str,
    payload: dict[str, torch.Tensor | float],
    converted: dict[str, torch.Tensor],
) -> bool:
    has_down_up = "lora_down.weight" in payload or "lora_up.weight" in payload
    if not has_down_up:
        return False

    down_weight = payload.get("lora_down.weight")
    up_weight = payload.get("lora_up.weight")
    if down_weight is None or up_weight is None:
        raise ValueError(
            f"LoRA down/up keys must be paired for module '{module_path}', but one side is missing."
        )
    if not torch.is_tensor(down_weight) or not torch.is_tensor(up_weight):
        raise ValueError(f"Invalid LoRA tensor payload for module '{module_path}'.")

    alpha = payload.get("alpha")
    if alpha is None:
        converted[f"transformer.{module_path}.lora_A.weight"] = down_weight
        converted[f"transformer.{module_path}.lora_B.weight"] = up_weight
    else:
        if isinstance(alpha, torch.Tensor):
            alpha = _to_float_alpha(alpha)
        scale_down, scale_up = _alpha_scales_from_down_weight(down_weight=down_weight, alpha_value=float(alpha))
        converted[f"transformer.{module_path}.lora_A.weight"] = down_weight * scale_down
        converted[f"transformer.{module_path}.lora_B.weight"] = up_weight * scale_up

    down_bias = payload.get("lora_down.bias")
    up_bias = payload.get("lora_up.bias")
    if down_bias is not None:
        if not torch.is_tensor(down_bias):
            raise ValueError(f"Invalid LoRA tensor payload for module '{module_path}.lora_down.bias'.")
        converted[f"transformer.{module_path}.lora_A.bias"] = down_bias
    if up_bias is not None:
        if not torch.is_tensor(up_bias):
            raise ValueError(f"Invalid LoRA tensor payload for module '{module_path}.lora_up.bias'.")
        converted[f"transformer.{module_path}.lora_B.bias"] = up_bias
    return True


def _convert_single_lora_module_payload(
    *,
    module_path: str,
    payload: dict[str, torch.Tensor | float],
    converted: dict[str, torch.Tensor],
) -> None:
    has_down_up_keys = "lora_down.weight" in payload or "lora_up.weight" in payload
    wrote_direct = _write_direct_lora_payload(
        module_path=module_path,
        payload=payload,
        converted=converted,
    )

    if wrote_direct:
        if has_down_up_keys:
            logger.warning(
                "Both direct LoRA keys and down/up keys were found for module '%s'. Direct keys are preferred.",
                module_path,
            )
        return

    wrote_down_up = _write_down_up_lora_payload(
        module_path=module_path,
        payload=payload,
        converted=converted,
    )
    if wrote_down_up:
        return

    raise ValueError(f"No supported LoRA weights found for module '{module_path}'.")


def _convert_non_diffusers_anima_lora_to_diffusers(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    entries = _collect_lora_module_payloads(state_dict)
    converted: dict[str, torch.Tensor] = {}
    for module_path, payload in entries.items():
        _convert_single_lora_module_payload(
            module_path=module_path,
            payload=payload,
            converted=converted,
        )

    if len(converted) == 0:
        raise ValueError("No loadable LoRA weights were found in the provided state dict.")
    return converted


class AnimaLoraLoaderMixin(LoraBaseMixin):
    _lora_loadable_modules = ["transformer"]
    transformer_name = "transformer"

    @classmethod
    @validate_hf_hub_args
    def lora_state_dict(
        cls,
        pretrained_model_name_or_path_or_dict: str | dict[str, torch.Tensor],
        **kwargs,
    ):
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        return_lora_metadata = kwargs.pop("return_lora_metadata", False)

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        user_agent = {"file_type": "attn_procs_weights", "framework": "pytorch"}
        state_dict, metadata = _fetch_state_dict(
            pretrained_model_name_or_path_or_dict=pretrained_model_name_or_path_or_dict,
            weight_name=weight_name,
            use_safetensors=use_safetensors,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
            allow_pickle=allow_pickle,
        )

        converted_state_dict = _convert_non_diffusers_anima_lora_to_diffusers(state_dict)
        if return_lora_metadata:
            return converted_state_dict, metadata
        return converted_state_dict

    def load_lora_weights(
        self,
        pretrained_model_name_or_path_or_dict: str | dict[str, torch.Tensor],
        adapter_name: str | None = None,
        hotswap: bool = False,
        **kwargs,
    ):
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT_LORA)
        if low_cpu_mem_usage and is_peft_version("<", "0.13.0"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict.copy()

        kwargs["return_lora_metadata"] = True
        state_dict, metadata = self.lora_state_dict(pretrained_model_name_or_path_or_dict, **kwargs)

        is_correct_format = all("lora" in key for key in state_dict)
        if not is_correct_format:
            raise ValueError("Invalid LoRA checkpoint. Make sure all LoRA param names contain `'lora'` substring.")

        target_transformer = _resolve_pipeline_transformer(self, self.transformer_name)
        self.load_lora_into_transformer(
            state_dict,
            transformer=target_transformer,
            adapter_name=adapter_name,
            metadata=metadata,
            _pipeline=self,
            low_cpu_mem_usage=low_cpu_mem_usage,
            hotswap=hotswap,
        )

    @classmethod
    def load_lora_into_transformer(
        cls,
        state_dict: dict[str, torch.Tensor],
        transformer,
        adapter_name: str | None = None,
        _pipeline=None,
        low_cpu_mem_usage: bool = False,
        hotswap: bool = False,
        metadata: dict | None = None,
    ):
        if low_cpu_mem_usage and is_peft_version("<", "0.13.0"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        logger.info("Loading %s.", cls.transformer_name)
        transformer.load_lora_adapter(
            state_dict,
            network_alphas=None,
            adapter_name=adapter_name,
            metadata=metadata,
            _pipeline=_pipeline,
            low_cpu_mem_usage=low_cpu_mem_usage,
            hotswap=hotswap,
        )

    @classmethod
    def save_lora_weights(
        cls,
        save_directory: str | os.PathLike,
        transformer_lora_layers: dict[str, torch.nn.Module | torch.Tensor] | None = None,
        is_main_process: bool = True,
        weight_name: str | None = None,
        save_function: Callable | None = None,
        safe_serialization: bool = True,
        transformer_lora_adapter_metadata: dict | None = None,
    ):
        lora_layers = {}
        lora_metadata = {}
        if transformer_lora_layers:
            lora_layers[cls.transformer_name] = transformer_lora_layers
            lora_metadata[cls.transformer_name] = transformer_lora_adapter_metadata
        if not lora_layers:
            raise ValueError("`transformer_lora_layers` must be provided.")

        cls._save_lora_weights(
            save_directory=save_directory,
            lora_layers=lora_layers,
            lora_metadata=lora_metadata,
            is_main_process=is_main_process,
            weight_name=weight_name,
            save_function=save_function,
            safe_serialization=safe_serialization,
        )

    def fuse_lora(
        self,
        components: list[str] | None = None,
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
        adapter_names: list[str] | None = None,
        **kwargs,
    ):
        resolved_components = ["transformer"] if components is None else components
        super().fuse_lora(
            components=resolved_components,
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
            adapter_names=adapter_names,
            **kwargs,
        )

    def unfuse_lora(self, components: list[str] | None = None, **kwargs):
        resolved_components = ["transformer"] if components is None else components
        super().unfuse_lora(components=resolved_components, **kwargs)


def _resolve_pipeline_transformer(pipeline: object, transformer_name: str):
    transformer = getattr(pipeline, "transformer", None)
    if transformer is not None:
        return transformer

    transformer = getattr(pipeline, transformer_name, None)
    if transformer is None:
        raise ValueError(f"Could not resolve transformer component '{transformer_name}' from pipeline.")
    return transformer
