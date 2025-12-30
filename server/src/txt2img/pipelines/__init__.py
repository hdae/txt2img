"""Pipeline implementations for different model types."""

from txt2img.config import ModelType, get_model_config
from txt2img.core.base_pipeline import BasePipeline

_pipeline: BasePipeline | None = None


def get_pipeline() -> BasePipeline:
    """Get or create the pipeline instance based on model config.

    Returns:
        BasePipeline instance for the configured model type
    """
    global _pipeline
    if _pipeline is None:
        config = get_model_config()
        if config.type == ModelType.SDXL:
            from txt2img.pipelines.sdxl import SDXLPipeline

            _pipeline = SDXLPipeline()
        elif config.type == ModelType.SD3:
            raise NotImplementedError("SD3 pipeline not yet implemented")
        elif config.type == ModelType.FLUX:
            from txt2img.pipelines.flux_dev import FluxDevPipeline

            _pipeline = FluxDevPipeline()
        elif config.type == ModelType.FLUX_SCHNELL:
            from txt2img.pipelines.flux_schnell import FluxSchnellPipeline

            _pipeline = FluxSchnellPipeline()
        elif config.type == ModelType.CHROMA:
            from txt2img.pipelines.chroma import ChromaPipelineImpl

            _pipeline = ChromaPipelineImpl()
        elif config.type == ModelType.ZIMAGE:
            from txt2img.pipelines.zimage import ZImagePipelineImpl

            _pipeline = ZImagePipelineImpl()
        else:
            raise ValueError(f"Unsupported model type: {config.type}")
    return _pipeline
