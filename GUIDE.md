# txt2img API Guide

This guide provides request body examples for different model types.

> **Note**: `steps`, `cfg_scale`, and `sampler` are now fixed per model and
> cannot be changed via API. Use `/api/info` to see the `parameter_schema` with
> fixed values for the current model.

## SDXL / Illustrious

Best for anime-style illustrations with detailed prompts.

```jsonc
{
    "prompt": "masterpiece, best quality, 1girl, solo, long blonde hair, blue eyes, white dress, flower garden, soft lighting",
    "negative_prompt": "worst quality, low quality, blurry, bad anatomy, bad hands, watermark",
    "width": 1024,
    "height": 1024,
    "seed": 42, // null for random
    "loras": [] // SDXL only: LoRA support
}
```

### With LoRA (SDXL only)

```jsonc
{
    "prompt": "masterpiece, best quality, 1girl, portrait, looking at viewer",
    "negative_prompt": "worst quality, low quality",
    "width": 1024,
    "height": 1024,
    "seed": null,
    "loras": [
        {
            "id": "civitai_1963644", // Get from /api/info
            "weight": 0.8, // LoRA strength (0.0-2.0)
            "trigger_weight": 0.5 // Trigger embedding weight
        }
    ]
}
```

> Get available LoRA IDs from `/api/info`

## Chroma

Lightweight Flux variant. Works well with natural language prompts. **LoRA not
supported.**

```jsonc
{
    "prompt": "a fluffy orange cat sleeping on a sunny windowsill, soft natural light",
    "width": 1024,
    "height": 1024,
    "seed": null
}
```

## Flux Dev

High quality with strong text understanding. Use natural language prompts.
**LoRA not supported.**

```jsonc
{
    "prompt": "A photorealistic landscape of snow-capped mountains reflected in a crystal clear alpine lake during golden hour",
    "width": 1024,
    "height": 1024,
    "seed": null
}
```

## Flux Schnell

Fast 4-step variant. **LoRA not supported.**

```jsonc
{
    "prompt": "minimalist logo design, geometric shapes, clean lines, modern branding",
    "width": 1024,
    "height": 1024,
    "seed": 999
}
```

## Z-Image Turbo

Fast 8-step model with excellent text rendering. **LoRA not supported.**

```jsonc
{
    "prompt": "A vintage coffee shop interior with warm lighting, wooden furniture, and a chalkboard menu displaying 'Today's Special'",
    "width": 1024,
    "height": 1024,
    "seed": null
}
```

## Parameter Reference

| Parameter         | Type          | Default  | Range    | Description                     |
| ----------------- | ------------- | -------- | -------- | ------------------------------- |
| `prompt`          | string        | required | -        | Image description               |
| `negative_prompt` | string        | `""`     | -        | What to avoid (SDXL only)       |
| `width`           | int           | `1024`   | 256-2048 | Image width in pixels           |
| `height`          | int           | `1024`   | 256-2048 | Image height in pixels          |
| `seed`            | int \| null   | `null`   | 0-2^32   | Random seed (null = random)     |
| `loras`           | array \| null | `null`   | -        | LoRA configurations (SDXL only) |

### Fixed Parameters (per model)

These parameters are fixed per model and retrieved via `/api/info`:

| Model         | Steps | CFG Scale | Sampler |
| ------------- | ----- | --------- | ------- |
| SDXL          | 20    | 7.0       | euler_a |
| Chroma        | 4     | 4.0       | -       |
| Flux Dev      | 50    | 3.5       | -       |
| Flux Schnell  | 4     | 0.0       | -       |
| Z-Image Turbo | 8     | 0.0       | -       |

### LoRA Object (SDXL only)

| Property         | Type   | Default  | Range   | Description              |
| ---------------- | ------ | -------- | ------- | ------------------------ |
| `id`             | string | required | -       | LoRA ID from `/api/info` |
| `weight`         | float  | `1.0`    | 0.0-2.0 | LoRA strength            |
| `trigger_weight` | float  | `0.5`    | 0.0-2.0 | Trigger embedding weight |

## Aspect Ratios

| Ratio | Resolution | Use Case         |
| ----- | ---------- | ---------------- |
| 1:1   | 1024×1024  | Square (default) |
| 4:3   | 1152×896   | Standard photo   |
| 16:9  | 1344×768   | Landscape/banner |
| 9:16  | 768×1344   | Portrait/mobile  |
| 3:2   | 1216×832   | Photography      |
