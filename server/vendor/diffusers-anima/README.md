# diffusers-anima

`diffusers-anima` provides an Anima pipeline implementation designed to align with Diffusers patterns.
The repository focuses on reusable Python pipeline code under `src/diffusers_anima`.

## Install

```bash
uv sync
```

## Python API

### 1. Basic generation

```python
import torch

from diffusers_anima import AnimaPipeline

pipe = AnimaPipeline.from_pretrained(
    "circlestone-labs/Anima::split_files/diffusion_models/anima-preview.safetensors",
    # Optional component overrides (defaults are built in):
    #   text_encoder_weights="circlestone-labs/Anima::split_files/text_encoders/qwen_3_06b_base.safetensors",
    #   text_encoder_config_repo="Qwen/Qwen3-0.6B-Base",
    #   qwen_tokenizer_repo="Qwen/Qwen-Image::tokenizer",
    #   t5_tokenizer_repo="google/flan-t5-xl",
    #   vae_repo="circlestone-labs/Anima::split_files/vae/qwen_image_vae.safetensors",
    # Optional runtime/loading options:
    #   device="cuda",
    #   dtype="auto",
    #   text_encoder_dtype="auto",
    #   local_files_only=False,
    #   enable_model_cpu_offload=False,
    #   enable_vae_slicing=False,
    #   enable_vae_tiling=False,
    #   enable_vae_xformers=False,
)

# Equivalent local single-file loader:
# pipe = AnimaPipeline.from_single_file(
#     "/absolute/path/to/anima-preview.safetensors",
#     # Optional kwargs are the same as from_pretrained(...)
# )

generator = torch.Generator(device="cpu").manual_seed(42)

result = pipe(
    prompt="masterpiece, best quality, 1girl",
    negative_prompt="worst quality, low quality",
    width=1024,
    height=1024,
    num_inference_steps=32,
    guidance_scale=4.0,
    generator=generator,  # Diffusers-style RNG input
)
image = result.images[0]
image.save("anima_from_pretrained.png")

# Multi-batch (prompt list + multiple images per prompt):
batch_generators = [torch.Generator(device="cpu").manual_seed(100 + i) for i in range(4)]
batch_result = pipe(
    prompt=["masterpiece, best quality, 1girl", "masterpiece, best quality, 1boy"],
    negative_prompt="worst quality, low quality",
    num_images_per_prompt=2,
    width=1024,
    height=1024,
    num_inference_steps=32,
    guidance_scale=4.0,
    generator=batch_generators,  # length must be batch_size * num_images_per_prompt
)
for idx, image in enumerate(batch_result.images):
    image.save(f"anima_batch_{idx}.png")

```

### 2. Img2Img and inpaint

```python
from PIL import Image
import torch

from diffusers_anima import AnimaPipeline

pipe = AnimaPipeline.from_pretrained(
    "circlestone-labs/Anima::split_files/diffusion_models/anima-preview.safetensors",
)
init_image = Image.open("/absolute/path/to/input.png").convert("RGB")
mask_image = Image.open("/absolute/path/to/mask.png").convert("L")

# Img2Img:
img2img_generator = torch.Generator(device="cpu").manual_seed(42)
img2img = pipe(
    prompt="masterpiece, best quality, 1girl",
    negative_prompt="worst quality, low quality",
    image=init_image,
    strength=0.65,
    width=1024,
    height=1024,
    num_inference_steps=32,
    guidance_scale=4.0,
    generator=img2img_generator,
)
img2img.images[0].save("anima_img2img.png")

# Inpaint:
inpaint_generator = torch.Generator(device="cpu").manual_seed(43)
inpaint = pipe(
    prompt="masterpiece, best quality, 1girl",
    negative_prompt="worst quality, low quality",
    image=init_image,
    mask_image=mask_image,  # white area is repainted
    strength=0.75,
    width=1024,
    height=1024,
    num_inference_steps=32,
    guidance_scale=4.0,
    generator=inpaint_generator,
)
inpaint.images[0].save("anima_inpaint.png")
```

### 3. LoRA example

```python
from diffusers_anima import AnimaPipeline
import torch

pipe = AnimaPipeline.from_pretrained(
    "circlestone-labs/Anima::split_files/diffusion_models/anima-preview.safetensors",
)
pipe.load_lora_weights("/absolute/path/to/anima_lora.safetensors", adapter_name="style")
pipe.set_adapters("style", adapter_weights=[0.8])

result = pipe(
    prompt="masterpiece, best quality, 1girl",
    negative_prompt="worst quality, low quality",
    width=1024,
    height=1024,
    num_inference_steps=32,
    guidance_scale=4.0,
    generator=torch.Generator(device="cpu").manual_seed(42),
)
image = result.images[0]
image.save("anima_with_lora.png")
```

## Notes

- `from_single_file` in this README assumes a local `.safetensors` model path.
- `from_single_file` accepts the same runtime/loading options as `from_pretrained` (including `enable_vae_slicing` and related flags).
- You can pass loading options such as `local_files_only`, `cache_dir`, `force_download`, `token`, `revision`, and `proxies`.
- `generator` accepts `torch.Generator` or a list/tuple of `torch.Generator` (length must match effective batch size).
- `prompt` and `negative_prompt` accept either a string or a list/tuple of strings.
- `num_images_per_prompt` is supported.
- `image` / `mask_image` accept `PIL.Image`, `numpy.ndarray`, `torch.Tensor`, or list/tuple of those types.
- `strength` is for `image`/`mask_image` workflows and must be in `(0, 1]`. Keep `strength=1.0` for text-to-image.
- `mask_image` requires `image`. White areas in the mask are regenerated.

## Integration Tests

```bash
ANIMA_RUN_INTEGRATION=1 uv run pytest tests/integration/test_regression_1girl.py -m integration -s
```

To refresh public baselines:

```bash
ANIMA_RUN_INTEGRATION=1 ANIMA_UPDATE_BASELINE=1 uv run pytest tests/integration/test_regression_1girl.py -m integration -s
```
