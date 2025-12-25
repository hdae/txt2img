"""Tests for prompt parser."""

import pytest

from txt2img.core.prompt_parser import (
    ParsedPrompt,
    ParserMode,
    WeightedChunk,
    parse_legacy_prompt,
    parse_prompt,
)


class TestLegacyParser:
    """Tests for legacy (StableDiffusionWebUI) prompt parser."""

    def test_simple_prompt(self):
        """Test parsing simple prompt without weights."""
        result = parse_legacy_prompt("a beautiful sunset")
        assert len(result.chunks) == 1
        assert len(result.chunks[0]) == 1
        assert result.chunks[0][0].text == "a beautiful sunset"
        assert result.chunks[0][0].weight == 1.0

    def test_explicit_weight(self):
        """Test parsing prompt with explicit weight."""
        result = parse_legacy_prompt("(masterpiece:1.2)")
        assert len(result.chunks) == 1
        assert result.chunks[0][0].text == "masterpiece"
        assert result.chunks[0][0].weight == pytest.approx(1.2)

    def test_multiple_words_weight(self):
        """Test parsing multiple words with weight."""
        result = parse_legacy_prompt("(best quality, high resolution:1.3)")
        assert len(result.chunks) == 1
        assert result.chunks[0][0].text == "best quality, high resolution"
        assert result.chunks[0][0].weight == pytest.approx(1.3)

    def test_emphasis_parentheses(self):
        """Test parsing emphasis with double parentheses."""
        result = parse_legacy_prompt("((important))")
        assert len(result.chunks) == 1
        assert result.chunks[0][0].text == "important"
        # ((text)) = weight * 1.1^2 = 1.21
        assert result.chunks[0][0].weight == pytest.approx(1.21)

    def test_deemphasis_brackets(self):
        """Test parsing de-emphasis with brackets."""
        result = parse_legacy_prompt("[less important]")
        assert len(result.chunks) == 1
        assert result.chunks[0][0].text == "less important"
        assert result.chunks[0][0].weight == pytest.approx(0.9)

    def test_break_syntax(self):
        """Test BREAK syntax for chunk separation."""
        result = parse_legacy_prompt("first part BREAK second part")
        assert len(result.chunks) == 2
        assert result.chunks[0][0].text == "first part"
        assert result.chunks[1][0].text == "second part"

    def test_break_case_insensitive(self):
        """Test BREAK is case insensitive."""
        result = parse_legacy_prompt("first break second")
        assert len(result.chunks) == 2

    def test_mixed_syntax(self):
        """Test mixed weight syntax."""
        result = parse_legacy_prompt("(masterpiece:1.4), best quality, ((detailed))")
        assert len(result.chunks) == 1
        chunks = result.chunks[0]
        # Should have multiple weighted parts
        assert any(c.weight == pytest.approx(1.4) for c in chunks)


class TestParsePrompt:
    """Tests for the main parse_prompt function."""

    def test_legacy_mode(self):
        """Test parsing in legacy mode."""
        result = parse_prompt("(test:1.2)", ParserMode.LEGACY)
        assert result.chunks[0][0].text == "test"
        assert result.chunks[0][0].weight == pytest.approx(1.2)

    def test_compel_mode(self):
        """Test parsing in compel mode keeps raw text."""
        result = parse_prompt("(test)1.2", ParserMode.COMPEL)
        # Compel mode should keep the raw text for Compel to parse
        assert result.chunks[0][0].text == "(test)1.2"
        assert result.chunks[0][0].weight == 1.0

    def test_empty_prompt(self):
        """Test parsing empty prompt."""
        result = parse_prompt("")
        assert len(result.chunks) == 1
        assert result.chunks[0][0].text == ""
