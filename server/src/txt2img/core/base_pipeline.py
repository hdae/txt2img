"""Abstract base class for all pipelines."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from txt2img.core.image_processor import SavedImage
from txt2img.core.job_queue import GenerationParams


class BasePipeline(ABC):
    """Abstract base class for all model pipelines.

    Each model type (SDXL, Flux, SD3, etc.) should implement this interface.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the loaded model name."""
        pass

    @abstractmethod
    async def load_model(self) -> None:
        """Load model from ModelConfig."""
        pass

    @abstractmethod
    async def generate(
        self,
        params: GenerationParams,
        progress_callback: Callable[[int, str | None], Any] | None = None,
    ) -> SavedImage:
        """Generate image from parameters.

        Args:
            params: Generation parameters
            progress_callback: Optional callback (step_num, preview_base64)

        Returns:
            SavedImage with result
        """
        pass
