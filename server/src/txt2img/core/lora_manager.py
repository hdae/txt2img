"""LoRA manager for dynamic LoRA selection and trigger word extraction."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from safetensors import safe_open

from txt2img.config import get_model_config, get_settings
from txt2img.providers.civitai import download_model as civitai_download
from txt2img.providers.civitai import get_model_version_info
from txt2img.utils.air_parser import AIRResource, parse_model_ref

logger = logging.getLogger(__name__)


@dataclass
class LoraInfo:
    """Information about a registered LoRA."""

    id: str  # Unique identifier (derived from ref)
    name: str  # Display name
    ref: str  # Original reference (AIR URN, HF repo, URL)
    path: Path | None = None  # Local file path after download
    trigger_words: list[str] = field(default_factory=list)
    weight: float = 1.0  # Recommended weight
    trigger_weight: float = 0.5  # Recommended trigger weight
    loaded: bool = False


class LoraManager:
    """Manages LoRA registration, loading, and trigger word extraction."""

    def __init__(self):
        self.loras: dict[str, LoraInfo] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize LoRAs from config."""
        if self._initialized:
            return

        config = get_model_config()
        for lora_config in config.loras:
            await self.register_lora(
                lora_config.ref,
                manual_triggers=lora_config.triggers,
                weight=lora_config.weight,
                trigger_weight=lora_config.trigger_weight,
            )

        self._initialized = True

    async def register_lora(
        self,
        ref: str,
        manual_triggers: list[str] | None = None,
        weight: float = 1.0,
        trigger_weight: float = 0.5,
    ) -> LoraInfo:
        """Register a LoRA and fetch its metadata.

        Args:
            ref: LoRA reference (AIR URN, HF repo, or URL)
            manual_triggers: Manual trigger words (overrides auto-detection)
            weight: Recommended LoRA weight
            trigger_weight: Recommended trigger embedding weight

        Returns:
            LoraInfo with metadata
        """
        ref = ref.strip()
        if not ref:
            raise ValueError("Empty LoRA reference")

        # Generate ID from ref
        lora_id = self._generate_id(ref)

        if lora_id in self.loras:
            return self.loras[lora_id]

        # Parse reference and download
        model_ref = parse_model_ref(ref)
        path = await self._download_lora(model_ref)

        # Get name and trigger words
        name = path.stem if path else lora_id

        # Use manual triggers if provided, otherwise auto-detect
        if manual_triggers:
            trigger_words = manual_triggers
            logger.info(f"Using manual triggers: {trigger_words}")
        else:
            trigger_words = await self._get_trigger_words(model_ref, path)

        lora_info = LoraInfo(
            id=lora_id,
            name=name,
            ref=ref,
            path=path,
            trigger_words=trigger_words,
            weight=weight,
            trigger_weight=trigger_weight,
        )

        self.loras[lora_id] = lora_info
        logger.info(f"Registered LoRA: {name} (triggers: {trigger_words})")

        return lora_info

    def get_lora(self, lora_id: str) -> LoraInfo | None:
        """Get LoRA info by ID."""
        return self.loras.get(lora_id)

    def get_available_loras(self) -> list[LoraInfo]:
        """Get all registered LoRAs."""
        return list(self.loras.values())

    def _generate_id(self, ref: str) -> str:
        """Generate unique ID from reference."""
        model_ref = parse_model_ref(ref)
        if isinstance(model_ref, AIRResource):
            # Use version ID for Civitai
            return f"civitai_{model_ref.version or model_ref.id}"
        # Use hash for others
        import hashlib

        return hashlib.md5(ref.encode()).hexdigest()[:12]

    async def _download_lora(self, model_ref) -> Path | None:
        """Download LoRA file."""
        from txt2img.providers.huggingface import download_hf_lora
        from txt2img.utils.air_parser import AIRResource, HuggingFaceResource, URLResource

        if isinstance(model_ref, AIRResource):
            return await civitai_download(model_ref)
        elif isinstance(model_ref, HuggingFaceResource):
            return await download_hf_lora(model_ref)
        elif isinstance(model_ref, URLResource):
            # URL download - simplified
            settings = get_settings()
            target_dir = settings.model_cache_dir / "loras"
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / model_ref.filename
            if not target_path.exists():
                import httpx

                async with httpx.AsyncClient() as client:
                    response = await client.get(model_ref.url, follow_redirects=True)
                    response.raise_for_status()
                    target_path.write_bytes(response.content)
            return target_path
        return None

    async def _get_trigger_words(self, model_ref, path: Path | None) -> list[str]:
        """Get trigger words. Tries Civitai API first, then safetensors metadata.

        Args:
            model_ref: Parsed model reference
            path: Local file path

        Returns:
            List of trigger words
        """
        trigger_words = []

        # Try Civitai API first
        if isinstance(model_ref, AIRResource) and model_ref.version:
            try:
                info = await get_model_version_info(model_ref.version)
                trained_words = info.get("trainedWords", [])
                if trained_words:
                    trigger_words = trained_words
                    logger.info(f"Got trigger words from Civitai: {trigger_words}")
                    return trigger_words
            except Exception as e:
                logger.warning(f"Failed to get trigger words from Civitai: {e}")

        # Fallback to safetensors metadata
        if path and path.exists():
            trigger_words = self._extract_trigger_words_from_file(path)
            if trigger_words:
                logger.info(f"Got trigger words from file: {trigger_words}")

        return trigger_words

    def _extract_trigger_words_from_file(self, path: Path) -> list[str]:
        """Extract trigger words from safetensors metadata.

        Args:
            path: Path to safetensors file

        Returns:
            List of trigger words
        """
        try:
            with safe_open(str(path), framework="pt", device="cpu") as f:
                metadata = f.metadata()
                if not metadata:
                    return []

                # Try direct trigger_words field
                if "trigger_words" in metadata:
                    return [w.strip() for w in metadata["trigger_words"].split(",")]

                # Try kohya-ss format: ss_tag_frequency
                if "ss_tag_frequency" in metadata:
                    try:
                        tag_freq = json.loads(metadata["ss_tag_frequency"])
                        # Get most frequent tags (usually trigger words)
                        all_tags = {}
                        for dataset_tags in tag_freq.values():
                            for tag, freq in dataset_tags.items():
                                all_tags[tag] = all_tags.get(tag, 0) + freq

                        # Debug: log all tags with frequency
                        sorted_tags = sorted(all_tags.items(), key=lambda x: -x[1])
                        logger.info(f"Tags from {path.name} (top 10):")
                        for tag, freq in sorted_tags[:10]:
                            logger.info(f"  {tag}: {freq}")

                        # TODO: Use threshold instead of top N
                        # For now, take top 3
                        return [tag for tag, _ in sorted_tags[:3]]
                    except json.JSONDecodeError:
                        pass

        except Exception as e:
            logger.warning(f"Failed to extract trigger words from {path}: {e}")

        return []

    def build_trigger_prompt(self, lora_requests: list[dict], original_prompt: str) -> str:
        """Build prompt with trigger words prepended.

        Args:
            lora_requests: List of {"id": "...", "weight": 1.0}
            original_prompt: Original prompt

        Returns:
            Prompt with trigger words
        """
        trigger_parts = []

        for req in lora_requests:
            lora_id = req.get("id") if isinstance(req, dict) else req
            weight = req.get("weight", 1.0) if isinstance(req, dict) else 1.0

            lora_info = self.get_lora(lora_id)
            if lora_info and lora_info.trigger_words:
                for word in lora_info.trigger_words:
                    if weight != 1.0:
                        trigger_parts.append(f"({word}:{weight:.2f})")
                    else:
                        trigger_parts.append(word)

        if trigger_parts:
            return ", ".join(trigger_parts) + ", " + original_prompt

        return original_prompt


# Global instance
lora_manager = LoraManager()
