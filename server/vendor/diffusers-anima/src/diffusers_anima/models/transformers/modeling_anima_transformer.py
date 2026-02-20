from __future__ import annotations

import math
import numbers
import re
from typing import Any

from diffusers import CosmosTransformer3DModel, ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm
from diffusers.utils import USE_PEFT_BACKEND, set_weights_and_activate_adapters
import torch
from torch import nn
import torch.nn.functional as F


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    feature_dim = x.shape[-1]
    if feature_dim % 2 != 0:
        raise ValueError(f"RoPE rotate_half expects even feature dim, got {feature_dim}.")
    half = feature_dim // 2
    paired = x.reshape(*x.shape[:-1], 2, half)
    first, second = paired.unbind(dim=-2)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos.unsqueeze(1)) + (_rotate_half(x) * sin.unsqueeze(1))


class _AnimaRMSNorm(nn.Module):
    """RMSNorm implementation used by the Anima adapter blocks."""

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-6,
        *,
        elementwise_affine: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (int(normalized_shape),)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        else:
            self.register_parameter("weight", None)

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_diffusers(cls, module: DiffusersRMSNorm) -> "_AnimaRMSNorm":
        patched = cls(
            tuple(module.dim),
            eps=float(module.eps),
            elementwise_affine=module.weight is not None,
            bias=module.bias is not None,
        )
        with torch.no_grad():
            if module.weight is not None:
                patched.weight.copy_(module.weight)
            if module.bias is not None and patched.bias is not None:
                patched.bias.copy_(module.bias)
        return patched

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is None:
            out = F.rms_norm(x, self.normalized_shape, eps=self.eps)
        else:
            out = F.rms_norm(
                x,
                self.normalized_shape,
                weight=self.weight.to(dtype=x.dtype, device=x.device),
                eps=self.eps,
            )
        if self.bias is not None:
            out = out + self.bias.to(dtype=out.dtype, device=out.device)
        return out


def _patch_diffusers_rmsnorm_to_anima(module: nn.Module) -> None:
    """Recursively replace Diffusers RMSNorm modules with Anima RMSNorm."""
    for child_name, child in list(module.named_children()):
        if isinstance(child, DiffusersRMSNorm):
            setattr(module, child_name, _AnimaRMSNorm.from_diffusers(child))
            continue
        _patch_diffusers_rmsnorm_to_anima(child)


class _RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        exponents = torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim)
        inv = torch.exp((-math.log(theta)) * exponents)
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv = self.inv_freq[None, :, None].to(device=x.device, dtype=torch.float32).expand(positions.shape[0], -1, 1)
        pos = positions[:, None, :].to(dtype=torch.float32)
        freqs = (inv @ pos).transpose(1, 2)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


class _AdapterAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, heads: int):
        super().__init__()
        inner = query_dim
        head_dim = inner // heads
        self.heads = heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(query_dim, inner, bias=False)
        self.k_proj = nn.Linear(context_dim, inner, bias=False)
        self.v_proj = nn.Linear(context_dim, inner, bias=False)
        self.q_norm = _AnimaRMSNorm(head_dim)
        self.k_norm = _AnimaRMSNorm(head_dim)
        self.o_proj = nn.Linear(inner, query_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        pos_q: tuple[torch.Tensor, torch.Tensor] | None = None,
        pos_k: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        context = x if context is None else context

        q = self.q_proj(x).view(x.shape[0], x.shape[1], self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(context.shape[0], context.shape[1], self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(context.shape[0], context.shape[1], self.heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if pos_q is not None and pos_k is not None:
            q = _apply_rope(q, *pos_q)
            k = _apply_rope(k, *pos_k)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        y = y.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1).contiguous()
        return self.o_proj(y)


class _AdapterBlock(nn.Module):
    def __init__(self, model_dim: int = 1024, context_dim: int = 1024, heads: int = 16):
        super().__init__()
        self.norm_self_attn = _AnimaRMSNorm(model_dim)
        self.self_attn = _AdapterAttention(model_dim, model_dim, heads)
        self.norm_cross_attn = _AnimaRMSNorm(model_dim)
        self.cross_attn = _AdapterAttention(model_dim, context_dim, heads)
        self.norm_mlp = _AnimaRMSNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4, bias=True),
            nn.GELU(),
            nn.Linear(model_dim * 4, model_dim, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        context: torch.Tensor,
        target_mask: torch.Tensor | None,
        source_mask: torch.Tensor | None,
        pos_target: tuple[torch.Tensor, torch.Tensor],
        pos_source: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        x = x + self.self_attn(
            self.norm_self_attn(x),
            attn_mask=target_mask,
            pos_q=pos_target,
            pos_k=pos_target,
        )
        x = x + self.cross_attn(
            self.norm_cross_attn(x),
            context=context,
            attn_mask=source_mask,
            pos_q=pos_target,
            pos_k=pos_source,
        )
        x = x + self.mlp(self.norm_mlp(x))
        return x


class _LLMAdapter(nn.Module):
    def __init__(self, vocab_size: int = 32128, dim: int = 1024, layers: int = 6, heads: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([_AdapterBlock(model_dim=dim, context_dim=dim, heads=heads) for _ in range(layers)])
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.norm = _AnimaRMSNorm(dim)
        self.rope = _RotaryEmbedding(dim // heads)

    def forward(
        self,
        source_hidden_states: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor | None = None,
        source_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if target_attention_mask is not None and target_attention_mask.ndim == 2:
            target_attention_mask = target_attention_mask.to(torch.bool).unsqueeze(1).unsqueeze(1)
        if source_attention_mask is not None and source_attention_mask.ndim == 2:
            source_attention_mask = source_attention_mask.to(torch.bool).unsqueeze(1).unsqueeze(1)

        x = self.embed(target_input_ids)
        context = source_hidden_states
        pos_target = self.rope(x, torch.arange(x.shape[1], device=x.device).unsqueeze(0))
        pos_source = self.rope(context, torch.arange(context.shape[1], device=context.device).unsqueeze(0))

        for block in self.blocks:
            x = block(
                x,
                context=context,
                target_mask=target_attention_mask,
                source_mask=source_attention_mask,
                pos_target=pos_target,
                pos_source=pos_source,
            )
        return self.norm(self.out_proj(x))


class AnimaTransformerModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    @register_to_config
    def __init__(self):
        super().__init__()
        core = _create_anima_transformer_core_model()
        _patch_diffusers_rmsnorm_to_anima(core)
        self.core = core
        self.llm_adapter = _LLMAdapter()

    def preprocess_text_embeds(
        self,
        text_embeds: torch.Tensor,
        text_ids: torch.Tensor | None,
        t5xxl_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if text_ids is None:
            return text_embeds
        out = self.llm_adapter(text_embeds, text_ids)
        if t5xxl_weights is not None:
            out = out.mul(t5xxl_weights)
        pad_tokens = max(0, 512 - out.shape[1])
        if pad_tokens:
            out = F.pad(out, (0, 0, 0, pad_tokens))
        return out

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        t5xxl_ids = kwargs.pop("t5xxl_ids", None)
        if t5xxl_ids is not None:
            context = self.preprocess_text_embeds(context, t5xxl_ids, t5xxl_weights=kwargs.pop("t5xxl_weights", None))

        padding_mask = kwargs.pop("padding_mask", None)
        if padding_mask is None:
            # CosmosTransformer3DModel internally repeats this per batch, so keep batch=1 here.
            padding_mask = torch.zeros((1, 1, x.shape[-2], x.shape[-1]), device=x.device, dtype=x.dtype)

        return self.core(
            hidden_states=x,
            timestep=timesteps,
            encoder_hidden_states=context,
            padding_mask=padding_mask,
            return_dict=False,
        )[0]

    def set_adapters(
        self,
        adapter_names: list[str] | str,
        weights: float | dict[str, float] | list[float | dict[str, float] | None] | None = None,
    ) -> None:
        """Set active LoRA adapters without relying on Diffusers private model-name mappings."""
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for `set_adapters()`.")

        normalized_names = [adapter_names] if isinstance(adapter_names, str) else list(adapter_names)
        if not isinstance(weights, list):
            normalized_weights = [weights] * len(normalized_names)
        else:
            normalized_weights = list(weights)

        if len(normalized_names) != len(normalized_weights):
            raise ValueError(
                f"Length of adapter names {len(normalized_names)} is not equal to the length of their weights "
                f"{len(normalized_weights)}."
            )

        resolved_weights = [weight if weight is not None else 1.0 for weight in normalized_weights]
        set_weights_and_activate_adapters(self, normalized_names, resolved_weights)


def _create_anima_transformer_core_model() -> CosmosTransformer3DModel:
    return CosmosTransformer3DModel(
        in_channels=16,
        out_channels=16,
        num_attention_heads=16,
        attention_head_dim=128,
        num_layers=28,
        mlp_ratio=4.0,
        text_embed_dim=1024,
        adaln_lora_dim=256,
        max_size=(128, 240, 240),
        patch_size=(1, 2, 2),
        rope_scale=(1.0, 4.0, 4.0),
        concat_padding_mask=True,
        extra_pos_embed_type=None,
    )


def _convert_anima_state_dict_to_diffusers(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    core: dict[str, torch.Tensor] = {}
    adapter: dict[str, torch.Tensor] = {}

    root_map = {
        "x_embedder.proj.1.weight": "core.patch_embed.proj.weight",
        "t_embedder.1.linear_1.weight": "core.time_embed.t_embedder.linear_1.weight",
        "t_embedder.1.linear_2.weight": "core.time_embed.t_embedder.linear_2.weight",
        "t_embedding_norm.weight": "core.time_embed.norm.weight",
        "final_layer.adaln_modulation.1.weight": "core.norm_out.linear_1.weight",
        "final_layer.adaln_modulation.2.weight": "core.norm_out.linear_2.weight",
        "final_layer.linear.weight": "core.proj_out.weight",
    }

    block_maps = {
        "adaln_modulation_self_attn.1.weight": "norm1.linear_1.weight",
        "adaln_modulation_self_attn.2.weight": "norm1.linear_2.weight",
        "adaln_modulation_cross_attn.1.weight": "norm2.linear_1.weight",
        "adaln_modulation_cross_attn.2.weight": "norm2.linear_2.weight",
        "adaln_modulation_mlp.1.weight": "norm3.linear_1.weight",
        "adaln_modulation_mlp.2.weight": "norm3.linear_2.weight",
        "self_attn.q_norm.weight": "attn1.norm_q.weight",
        "self_attn.k_norm.weight": "attn1.norm_k.weight",
        "self_attn.q_proj.weight": "attn1.to_q.weight",
        "self_attn.k_proj.weight": "attn1.to_k.weight",
        "self_attn.v_proj.weight": "attn1.to_v.weight",
        "self_attn.output_proj.weight": "attn1.to_out.0.weight",
        "cross_attn.q_norm.weight": "attn2.norm_q.weight",
        "cross_attn.k_norm.weight": "attn2.norm_k.weight",
        "cross_attn.q_proj.weight": "attn2.to_q.weight",
        "cross_attn.k_proj.weight": "attn2.to_k.weight",
        "cross_attn.v_proj.weight": "attn2.to_v.weight",
        "cross_attn.output_proj.weight": "attn2.to_out.0.weight",
        "mlp.layer1.weight": "ff.net.0.proj.weight",
        "mlp.layer2.weight": "ff.net.2.weight",
    }

    block_re = re.compile(r"^blocks\.(\d+)\.(.+)$")
    for key, value in state_dict.items():
        if key.startswith("llm_adapter."):
            adapter[".".join(["llm_adapter", key.removeprefix("llm_adapter.")])] = value
            continue

        mapped = root_map.get(key)
        if mapped is not None:
            core[mapped] = value
            continue

        m = block_re.match(key)
        if m is not None:
            block_index = m.group(1)
            tail = m.group(2)
            mapped_tail = block_maps.get(tail)
            if mapped_tail is None:
                raise RuntimeError(f"Unsupported Anima checkpoint key in blocks: {key}")
            core[f"core.transformer_blocks.{block_index}.{mapped_tail}"] = value
            continue

        raise RuntimeError(f"Unsupported Anima checkpoint key: {key}")

    return core, adapter
