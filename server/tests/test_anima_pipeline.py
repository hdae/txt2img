from pathlib import Path

import torch

import txt2img.pipelines as pipelines
from txt2img.config import get_model_config, load_model_config
from txt2img.pipelines.anima import AnimaPipelineImpl


def test_anima_pipeline_is_selected_from_preset(monkeypatch):
    """Anima preset should select the Anima pipeline and expose expected schema defaults."""
    preset_path = Path(__file__).resolve().parents[1] / "presets" / "anima" / "preview.json"

    monkeypatch.setattr(pipelines, "_pipeline", None)
    load_model_config(f"file://{preset_path}")

    pipeline = pipelines.get_pipeline()
    schema = pipeline.get_parameter_schema()
    model_config = get_model_config()

    assert schema["model_type"] == "anima"
    assert schema["fixed"]["steps"] == 20
    assert schema["properties"]["cfg_scale"]["default"] == 4.0
    assert model_config.cfg_batch_mode == "concat"


def test_anima_sdpa_backends_are_enabled_on_cuda(monkeypatch):
    """Anima should explicitly enable SDPA backends when CUDA is available."""
    calls: list[tuple[str, bool]] = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.backends.cuda,
        "enable_flash_sdp",
        lambda enabled: calls.append(("flash", enabled)),
    )
    monkeypatch.setattr(
        torch.backends.cuda,
        "enable_mem_efficient_sdp",
        lambda enabled: calls.append(("mem_efficient", enabled)),
    )
    monkeypatch.setattr(
        torch.backends.cuda,
        "enable_math_sdp",
        lambda enabled: calls.append(("math", enabled)),
    )
    monkeypatch.setattr(torch.backends.cuda, "flash_sdp_enabled", lambda: True)
    monkeypatch.setattr(torch.backends.cuda, "mem_efficient_sdp_enabled", lambda: True)
    monkeypatch.setattr(torch.backends.cuda, "math_sdp_enabled", lambda: True)

    pipeline = AnimaPipelineImpl()
    pipeline._configure_sdpa_backends()

    assert calls == [
        ("flash", True),
        ("mem_efficient", True),
        ("math", True),
    ]
