"""Tests for prompt parser."""

import pytest

from txt2img.core.prompt_parser import (
    PromptParser,
    convert_a1111_to_compel,
    parse_a1111_prompt,
    parse_prompt,
)


class TestA1111Parser:
    """Tests for A1111 (StableDiffusionWebUI) prompt parser."""

    def test_simple_prompt(self):
        """Test parsing simple prompt without weights."""
        result = parse_a1111_prompt("a beautiful sunset")
        assert len(result.chunks) == 1
        assert len(result.chunks[0]) == 1
        assert result.chunks[0][0].text == "a beautiful sunset"
        assert result.chunks[0][0].weight == 1.0

    def test_explicit_weight(self):
        """Test parsing prompt with explicit weight."""
        result = parse_a1111_prompt("(masterpiece:1.2)")
        assert len(result.chunks) == 1
        assert result.chunks[0][0].text == "masterpiece"
        assert result.chunks[0][0].weight == pytest.approx(1.2)

    def test_multiple_words_weight(self):
        """Test parsing multiple words with weight."""
        result = parse_a1111_prompt("(best quality, high resolution:1.3)")
        assert len(result.chunks) == 1
        assert result.chunks[0][0].text == "best quality, high resolution"
        assert result.chunks[0][0].weight == pytest.approx(1.3)

    def test_emphasis_parentheses(self):
        """Test parsing emphasis with double parentheses."""
        result = parse_a1111_prompt("((important))")
        assert len(result.chunks) == 1
        assert result.chunks[0][0].text == "important"
        # ((text)) = weight * 1.1^2 = 1.21
        assert result.chunks[0][0].weight == pytest.approx(1.21)

    def test_deemphasis_brackets(self):
        """Test parsing de-emphasis with brackets."""
        result = parse_a1111_prompt("[less important]")
        assert len(result.chunks) == 1
        assert result.chunks[0][0].text == "less important"
        assert result.chunks[0][0].weight == pytest.approx(0.9)

    def test_deemphasis_with_explicit_weight(self):
        """Test parsing de-emphasis with explicit weight."""
        result = parse_a1111_prompt("[less important:0.5]")
        assert len(result.chunks) == 1
        assert result.chunks[0][0].text == "less important"
        assert result.chunks[0][0].weight == pytest.approx(0.5)

    def test_break_syntax(self):
        """Test BREAK syntax for chunk separation."""
        result = parse_a1111_prompt("first part BREAK second part")
        assert len(result.chunks) == 2
        assert result.chunks[0][0].text == "first part"
        assert result.chunks[1][0].text == "second part"

    def test_break_case_insensitive(self):
        """Test BREAK is case insensitive."""
        result = parse_a1111_prompt("first break second")
        assert len(result.chunks) == 2

    def test_mixed_syntax(self):
        """Test mixed weight syntax."""
        result = parse_a1111_prompt("(masterpiece:1.4), best quality, ((detailed))")
        assert len(result.chunks) == 1
        chunks = result.chunks[0]
        # Should have multiple weighted parts
        assert any(c.weight == pytest.approx(1.4) for c in chunks)


class TestA1111ToCompelConversion:
    """Tests for A1111 to Compel syntax conversion."""

    def test_explicit_weight_conversion(self):
        """Test (word:1.5) converts to (word)1.50."""
        result = convert_a1111_to_compel("(masterpiece:1.5)")
        assert "(masterpiece)1.50" in result

    def test_emphasis_conversion(self):
        """Test ((word)) converts to (word)1.21."""
        result = convert_a1111_to_compel("((important))")
        assert "(important)1.21" in result

    def test_deemphasis_conversion(self):
        """Test [word] converts to (word)0.90."""
        result = convert_a1111_to_compel("[less important]")
        assert "(less important)0.90" in result

    def test_triple_brackets_conversion(self):
        """Test [[[word]]] converts to (word)0.73."""
        result = convert_a1111_to_compel("[[[word]]]")
        assert "(word)0.73" in result

    def test_no_weight_unchanged(self):
        """Test text without weight stays unchanged."""
        result = convert_a1111_to_compel("simple text")
        assert result == "simple text"


class TestParsePrompt:
    """Tests for the main parse_prompt function."""

    def test_lpw_mode(self):
        """Test parsing in LPW mode (A1111 syntax converted to Compel)."""
        result = parse_prompt("(test:1.2)", PromptParser.LPW)
        # LPW mode converts to Compel format
        assert "(test)1.20" in result.chunks[0][0].text

    def test_compel_mode(self):
        """Test parsing in compel mode keeps raw text."""
        result = parse_prompt("(test)1.2", PromptParser.COMPEL)
        # Compel mode should keep the raw text for Compel to parse
        assert result.chunks[0][0].text == "(test)1.2"
        assert result.chunks[0][0].weight == 1.0

    def test_empty_prompt(self):
        """Test parsing empty prompt."""
        result = parse_prompt("")
        assert len(result.chunks) == 1
        assert result.chunks[0][0].text == ""
