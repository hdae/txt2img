# txt2img API Guide

This guide provides complete request body examples for different model types.
Examples use JSONC format with comments to highlight model-specific parameters.

## SDXL / Illustrious

Best for anime-style illustrations with detailed prompts.

```jsonc
{
    "prompt": "masterpiece, best quality, 1girl, solo, long blonde hair, blue eyes, white dress, flower garden, soft lighting",
    "negative_prompt": "worst quality, low quality, blurry, bad anatomy, bad hands, watermark", // Important for SDXL
    "width": 1024,
    "height": 1024,
    "steps": 20, // Recommended: 20-30
    "cfg_scale": 7.0, // Recommended: 5.0-8.0
    "seed": 42, // null for random
    "sampler": "euler_a", // euler_a recommended for SDXL
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
    "steps": 20,
    "cfg_scale": 7.0,
    "seed": null,
    "sampler": "euler_a",
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

### Chroma Flash (4 steps)

```jsonc
{
    "prompt": "a fluffy orange cat sleeping on a sunny windowsill, soft natural light",
    "negative_prompt": "", // Optional for Chroma
    "width": 1024,
    "height": 1024,
    "steps": 4, // Flash: 4 steps optimal
    "cfg_scale": 4.0, // Recommended: 3.5-4.5
    "seed": null,
    "sampler": "euler"
}
```

### Chroma Base (20 steps)

```jsonc
{
    "prompt": "portrait of a woman with auburn hair, professional photography, studio lighting, magazine cover quality",
    "negative_prompt": "",
    "width": 1024,
    "height": 1024,
    "steps": 20, // Base: 20 steps recommended
    "cfg_scale": 4.0,
    "seed": 12345,
    "sampler": "euler"
}
```

### Chroma HD (28 steps)

```jsonc
{
    "prompt": "detailed cityscape at night, neon lights, cyberpunk aesthetic, rain reflections on wet streets",
    "negative_prompt": "",
    "width": 1024,
    "height": 1024,
    "steps": 28, // HD: 28 steps for highest quality
    "cfg_scale": 4.0,
    "seed": null,
    "sampler": "euler"
}
```

## Flux

High quality text understanding. Use natural language prompts. **LoRA not
supported.**

### Flux Dev (~50 steps)

```jsonc
{
    "prompt": "A photorealistic landscape of snow-capped mountains reflected in a crystal clear alpine lake during golden hour",
    "negative_prompt": "", // Flux ignores negative prompt
    "width": 1024,
    "height": 1024,
    "steps": 50, // Dev: 50 steps recommended
    "cfg_scale": 3.5, // Recommended: 3.0-4.0
    "seed": null,
    "sampler": "euler"
}
```

### Flux Schnell (4 steps)

```jsonc
{
    "prompt": "minimalist logo design, geometric shapes, clean lines, modern branding",
    "negative_prompt": "",
    "width": 1024,
    "height": 1024,
    "steps": 4, // Schnell: 4 steps optimal (max 10)
    "cfg_scale": 0.0, // Schnell: guidance-distilled, cfg ignored
    "seed": 999,
    "sampler": "euler"
}
```

## Z-Image Turbo

Fast 8-step model with excellent text rendering. **LoRA not supported.**

```jsonc
{
    "prompt": "A vintage coffee shop interior with warm lighting, wooden furniture, and a chalkboard menu displaying 'Today's Special'",
    "negative_prompt": "",
    "width": 1024,
    "height": 1024,
    "steps": 8, // 8 steps optimal
    "cfg_scale": 1.0, // Low cfg recommended
    "seed": null,
    "sampler": "euler"
}
```

### Landscape

```jsonc
{
    "prompt": "drone photography of winding river through autumn forest, vibrant fall colors, aerial view",
    "negative_prompt": "blurry, oversaturated",
    "width": 1344, // 16:9 aspect ratio
    "height": 768,
    "steps": 8,
    "cfg_scale": 1.0,
    "seed": 54321,
    "sampler": "euler"
}
```

## Parameter Reference

| Parameter         | Type          | Default   | Range     | Description                     |
| ----------------- | ------------- | --------- | --------- | ------------------------------- |
| `prompt`          | string        | required  | -         | Image description               |
| `negative_prompt` | string        | `""`      | -         | What to avoid                   |
| `width`           | int           | `1024`    | 256-2048  | Image width in pixels           |
| `height`          | int           | `1024`    | 256-2048  | Image height in pixels          |
| `steps`           | int           | `20`      | 1-100     | Inference steps                 |
| `cfg_scale`       | float         | `7.0`     | 1.0-30.0  | Prompt adherence strength       |
| `seed`            | int \| null   | `null`    | 0-2^32    | Random seed (null = random)     |
| `sampler`         | string        | `"euler"` | see below | Sampling algorithm              |
| `loras`           | array \| null | `null`    | -         | LoRA configurations (SDXL only) |

### Model-Specific Recommendations

| Model         | Steps | CFG Scale | Notes                        |
| ------------- | ----- | --------- | ---------------------------- |
| SDXL          | 20-30 | 5.0-8.0   | Use negative_prompt, LoRA OK |
| Chroma Flash  | 4     | 4.0       | Fast, lower quality          |
| Chroma Base   | 20    | 4.0       | Balanced                     |
| Chroma HD     | 28    | 4.0       | Highest quality              |
| Flux Dev      | 50    | 3.5       | High quality, slow           |
| Flux Schnell  | 4     | 0.0       | Fast, cfg ignored            |
| Z-Image Turbo | 8     | 1.0       | Fast, good text rendering    |

### Samplers

| Sampler        | Description                |
| -------------- | -------------------------- |
| `euler`        | Fast, general purpose      |
| `euler_a`      | Euler ancestral (creative) |
| `dpm++_2m`     | High quality, slower       |
| `dpm++_2m_sde` | DPM++ with SDE             |
| `dpm++_sde`    | DPM++ SDE variant          |
| `ddim`         | Deterministic              |
| `heun`         | Second-order accuracy      |
| `lms`          | Linear multi-step          |

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
