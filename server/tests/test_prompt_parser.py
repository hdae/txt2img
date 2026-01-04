"""Tests for prompt parser."""

import pytest

from txt2img.utils.prompt_parser import (
    PromptParser,
    build_compel_prompt,
    convert_a1111_to_compel,
    parse_a1111_prompt,
    parse_prompt,
    prepend_triggers,
)


class TestA1111Parser:
    """Tests for A1111 (StableDiffusionWebUI) prompt parser."""
    # ... (Tests remain same but verify import change) ...
    # Rewriting TestA1111Parser class to ensure it uses the new import
    # and to verify it didn't break.

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
        assert "(masterpiece)1.20" in result.chunks[0][0].text
        assert result.chunks[0][0].weight == 1.0

    def test_break_syntax(self):
        """Test BREAK syntax for chunk separation."""
        result = parse_a1111_prompt("first part BREAK second part")
        assert len(result.chunks) == 2
        assert "first part" in result.chunks[0][0].text
        assert "second part" in result.chunks[1][0].text


class TestA1111ToCompelConversion:
    """Tests for A1111 to Compel syntax conversion."""

    def test_explicit_weight_conversion(self):
        result = convert_a1111_to_compel("(masterpiece:1.5)")
        assert "(masterpiece)1.50" in result

    def test_emphasis_conversion(self):
        result = convert_a1111_to_compel("((important))")
        assert "(important)1.21" in result

    def test_deemphasis_conversion(self):
        result = convert_a1111_to_compel("[less important]")
        assert "(less important)0.90" in result

    def test_triple_brackets_conversion(self):
        result = convert_a1111_to_compel("[[[word]]]")
        assert "(word)0.73" in result

    def test_no_weight_unchanged(self):
        result = convert_a1111_to_compel("simple text")
        assert result == "simple text"

    def test_escaped_parentheses(self):
        result = convert_a1111_to_compel(r"a \(smile\)")
        assert r"\(smile\)" in result

    def test_nested_explicit_weights(self):
        result = convert_a1111_to_compel("((foo:1.1):1.2)")
        assert "(foo)1.32" in result

    def test_mixed_brackets(self):
        result = convert_a1111_to_compel("(foo [bar:0.5]:1.2)")
        assert "(foo )1.20" in result
        assert "(bar)0.60" in result


class TestHelperFunctions:
    """Tests for generic helper functions."""

    def test_prepend_triggers(self):
        """Test prepending triggers."""
        triggers = [("style lora", 0.8), ("trigger", 1.0)]
        result = prepend_triggers("user prompt", triggers)
        # Should be: (style lora)0.80, trigger BREAK user prompt
        assert "(style lora)0.80" in result
        assert "trigger" in result
        assert "BREAK user prompt" in result

    def test_build_compel_prompt_concat(self):
        """Test building prompt with concatenation."""
        prompt = "part1 BREAK part2"
        result = build_compel_prompt(prompt, use_concatenation=True)
        # Should be ("part1").and("part2")
        assert '("part1").and("part2")' == result

    def test_build_compel_prompt_no_concat(self):
        """Test building prompt without concatenation."""
        prompt = "part1 BREAK part2"
        result = build_compel_prompt(prompt, use_concatenation=False)
        assert "part1 part2" == result


class TestParsePrompt:
    """Tests for the main parse_prompt function."""

    def test_lpw_mode(self):
        result = parse_prompt("(test:1.2)", PromptParser.LPW)
        assert "(test)1.20" in result.chunks[0][0].text

    def test_compel_mode(self):
        result = parse_prompt("(test)1.2", PromptParser.COMPEL)
        assert result.chunks[0][0].text == "(test)1.2"
        assert result.chunks[0][0].weight == 1.0
