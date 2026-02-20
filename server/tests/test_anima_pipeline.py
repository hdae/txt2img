from pathlib import Path

import txt2img.pipelines as pipelines
from txt2img.config import load_model_config


def test_anima_pipeline_is_selected_from_preset(monkeypatch):
    """Anima preset should select the Anima pipeline and expose expected schema defaults."""
    preset_path = Path(__file__).resolve().parents[1] / "presets" / "anima" / "preview.json"

    monkeypatch.setattr(pipelines, "_pipeline", None)
    load_model_config(f"file://{preset_path}")

    pipeline = pipelines.get_pipeline()
    schema = pipeline.get_parameter_schema()

    assert schema["model_type"] == "anima"
    assert schema["fixed"]["steps"] == 32
    assert schema["properties"]["cfg_scale"]["default"] == 4.0
