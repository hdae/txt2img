import sys
import logging
from txt2img.config import load_model_config
from txt2img.pipelines import get_pipeline

logging.basicConfig(level=logging.INFO)

print("Loading config...")
load_model_config('file://presets/anima/preview.json')

print("Getting pipeline...")
p = get_pipeline()

print("Schema ID:")
print(p.get_parameter_schema()["model_type"])

print("SUCCESS!")
