# Models & Configuration

## Supported Models

| Type             | Description                                                  | Example Preset     | LoRA |
| :--------------- | :----------------------------------------------------------- | :----------------- | :--- |
| **SDXL**         | SDXL and derivatives (Illustrious, etc.). Tag-based prompts. | `sdxl/illustrious` | ✅   |
| **Chroma**       | Lightweight Flux variant (8.9B). Natural language.           | `chroma/flash`     | ❌   |
| **Flux Dev**     | Flux.1 [dev]. High quality, ~50 steps.                       | `flux/dev`         | ❌   |
| **Flux Schnell** | Flux.1 [schnell]. Fast, 4 steps.                             | `flux/schnell`     | ❌   |
| **Z-Image**      | Z-Image Turbo. Fast 6B model, 8 steps.                       | `zimage/turbo`     | ❌   |
| **Anima**        | Anima Diffusion Model. Tag-based prompts.                    | `anima/preview`    | ❌   |

### Fixed Parameters

Certain parameters are fixed per model to ensure optimal quality and
performance. These cannot be changed via the API.

| Model            | Steps | CFG Scale | Sampler |
| :--------------- | :---- | :-------- | :------ |
| **SDXL**         | 20    | 7.0       | euler_a |
| **Chroma**       | 4     | 4.0       | -       |
| **Flux Dev**     | 50    | 3.5       | -       |
| **Flux Schnell** | 4     | 0.0       | -       |
| **Z-Image**      | 8     | 0.0       | -       |
| **Anima**        | 32    | 4.0       | -       |

## VRAM Profiles

Set the `VRAM_PROFILE` environment variable in `.env`.

| Profile        | VRAM Target | Optimization Strategy                                                          |
| :------------- | :---------- | :----------------------------------------------------------------------------- |
| **`full`**     | 24GB+       | No offloading. Maximum speed. Keep model in VRAM.                              |
| **`balanced`** | 12-16GB     | CPU Offload + VAE Tiling. Good balance.                                        |
| **`lowvram`**  | 8GB         | Sequential CPU Offload (streaming) + VAE Tiling. Slowest but memory efficient. |

## LoRA Configuration (SDXL)

Configure LoRAs in the preset JSON or request body.

- **`ref`**: Civitai AIR URN (`urn:air:sdxl:lora:civitai:ID@VERSION`) or URL.
- **`triggers`**: Trigger words list. If omitted, fetched automatically from
  Civitai.
- **`weight`**: Overall influence (0.0 - 2.0). Default: 1.0.
- **`trigger_weight`**: Embedding weight for trigger words. Default: 0.5.
