"""Tests for AIR parser."""

import pytest

from txt2img.models.air_parser import (
    AIRResource,
    HuggingFaceResource,
    ModelEcosystem,
    ModelSource,
    ModelType,
    URLResource,
    is_air_urn,
    is_huggingface_repo,
    is_url,
    parse_air_urn,
    parse_model_ref,
)


class TestAIRParser:
    """Tests for AIR URN parsing."""

    def test_parse_full_urn(self):
        """Test parsing full AIR URN with version."""
        urn = "urn:air:sdxl:checkpoint:civitai:827184@2514310"
        result = parse_air_urn(urn)

        assert result.ecosystem == ModelEcosystem.SDXL
        assert result.type == ModelType.CHECKPOINT
        assert result.source == ModelSource.CIVITAI
        assert result.id == "827184"
        assert result.version == "2514310"
        assert result.format is None

    def test_parse_urn_without_prefix(self):
        """Test parsing AIR URN without urn:air: prefix."""
        urn = "sdxl:lora:civitai:328553@368189"
        result = parse_air_urn(urn)

        assert result.ecosystem == ModelEcosystem.SDXL
        assert result.type == ModelType.LORA
        assert result.source == ModelSource.CIVITAI
        assert result.id == "328553"
        assert result.version == "368189"

    def test_parse_urn_with_format(self):
        """Test parsing AIR URN with format."""
        urn = "urn:air:sdxl:checkpoint:civitai:123@456.safetensors"
        result = parse_air_urn(urn)

        assert result.version == "456"
        assert result.format == "safetensors"

    def test_parse_invalid_urn(self):
        """Test parsing invalid URN raises error."""
        with pytest.raises(ValueError):
            parse_air_urn("invalid:urn")

    def test_is_air_urn(self):
        """Test AIR URN detection."""
        assert is_air_urn("urn:air:sdxl:checkpoint:civitai:123@456")
        assert is_air_urn("sdxl:lora:civitai:123@456")
        assert not is_air_urn("https://example.com/model.safetensors")
        assert not is_air_urn("stabilityai/stable-diffusion-xl-base-1.0")


class TestURLParsing:
    """Tests for URL parsing."""

    def test_is_url(self):
        """Test URL detection."""
        assert is_url("https://example.com/model.safetensors")
        assert is_url("http://localhost:8000/file.bin")
        assert not is_url("sdxl:checkpoint:civitai:123")
        assert not is_url("stabilityai/sdxl")


class TestHuggingFaceParsing:
    """Tests for HuggingFace repo parsing."""

    def test_is_huggingface_repo(self):
        """Test HuggingFace repo detection."""
        assert is_huggingface_repo("stabilityai/stable-diffusion-xl-base-1.0")
        assert is_huggingface_repo("user/model-name")
        assert not is_huggingface_repo("https://huggingface.co/user/model")
        assert not is_huggingface_repo("urn:air:sdxl:checkpoint:civitai:123")


class TestParseModelRef:
    """Tests for general model reference parsing."""

    def test_parse_air(self):
        """Test parsing AIR URN."""
        result = parse_model_ref("urn:air:sdxl:checkpoint:civitai:123@456")
        assert isinstance(result, AIRResource)

    def test_parse_url(self):
        """Test parsing URL."""
        result = parse_model_ref("https://example.com/model.safetensors")
        assert isinstance(result, URLResource)
        assert result.filename == "model.safetensors"

    def test_parse_hf_repo(self):
        """Test parsing HuggingFace repo."""
        result = parse_model_ref("stabilityai/stable-diffusion-xl-base-1.0")
        assert isinstance(result, HuggingFaceResource)
        assert result.repo_id == "stabilityai/stable-diffusion-xl-base-1.0"

    def test_parse_invalid(self):
        """Test parsing invalid reference."""
        with pytest.raises(ValueError):
            parse_model_ref("not-a-valid-reference")
