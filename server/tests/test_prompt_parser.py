"""Tests for prompt parser."""

import pytest

from txt2img.core.prompt_parser import (
    PromptParser,
    convert_a1111_to_compel,
    parse_a1111_prompt,
    parse_prompt,
)


class TestA1111Parser:
    """Tests for A1111 (StableDiffusionWebUI) prompt parser.

    Note: The parser now returns chunks prepared for Compel, so weights
    are embedded in the text string rather than in the WeightedChunk object.
    """

    def test_simple_prompt(self):
        """Test parsing simple prompt without weights."""
        result = parse_a1111_prompt("a beautiful sunset")
        assert len(result.chunks) == 1
        assert len(result.chunks[0]) == 1
        assert result.chunks[0][0].text == "a beautiful sunset"
        assert result.chunks[0][0].weight == 1.0

    def test_explicit_weight(self):
        """Test parsing prompt with explicit weight."""
        # (masterpiece:1.2) -> Compel: (masterpiece)1.20
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
        """Test (word:1.5) converts to (word)1.50."""
        result = convert_a1111_to_compel("(masterpiece:1.5)")
        assert "(masterpiece)1.50" in result

    def test_emphasis_conversion(self):
        """Test ((word)) converts to (word)1.21."""
        result = convert_a1111_to_compel("((important))")
        # 1.1 * 1.1 = 1.21
        assert "(important)1.21" in result

    def test_deemphasis_conversion(self):
        """Test [word] converts to (word)0.90."""
        result = convert_a1111_to_compel("[less important]")
        assert "(less important)0.90" in result

    def test_triple_brackets_conversion(self):
        """Test [[[word]]] converts to (word)0.73."""
        # 0.9 * 0.9 * 0.9 = 0.729
        result = convert_a1111_to_compel("[[[word]]]")
        assert "(word)0.73" in result

    def test_no_weight_unchanged(self):
        """Test text without weight stays unchanged."""
        result = convert_a1111_to_compel("simple text")
        assert result == "simple text"

    def test_escaped_parentheses(self):
        """Test escaped parentheses are preserved as text."""
        result = convert_a1111_to_compel(r"a \(smile\)")
        # Should be "a \(smile\)" (exact behavior depends on tokenizer impl)
        # Our tokenizer preserves escapes.
        # Compel sees \(smile\) which is good.
        assert r"\(smile\)" in result

    def test_nested_explicit_weights(self):
        """Test nested explicit weights: ((foo:1.1):1.2)."""
        # Inner: foo:1.1 -> weight 1.1
        # Outer: (...) : 1.2 -> multiplier 1.2
        # Total: 1.1 * 1.2 = 1.32
        result = convert_a1111_to_compel("((foo:1.1):1.2)")
        assert "(foo)1.32" in result

    def test_mixed_brackets(self):
        """Test mixed brackets and explicit weights."""
        # [bar:0.5] -> 0.5 (explicit overrides bracket default?)
        # OR [ ... ] applies 0.9, then :0.5 overrides?
        # A1111 logic: [text:0.5] means weight 0.5. The brackets just delineate scope if explicit weight exists.
        # My implementation:
        # LBRACKET starts group with 0.9.
        # COLON with number updates current group weight to 0.5.
        # So it becomes 0.5.
        # Then inside ( ... : 1.2 ).
        # So ( ... [bar:0.5] ... : 1.2 )
        # Inner bar is 0.5.
        # Outer applies 1.2 multiplier to everything?
        # A1111: ( a, b:1.1 ) -> applies 1.1 to a and b.
        # If one of them has explicit weight? (a:1.5, b:1.1)
        # Usually explicit weight is absolute or relative?
        # In A1111, `( (a:1.1) )` -> `1.1 * 1.1 = 1.21`.
        # So explicit weight sets base, then parents multiply.

        # In my parser:
        # COLON updates weight_multiplier of the node.
        # If node is children of another node, the parent multiplier applies recursively.
        # `(foo [bar:0.5]:1.2)`
        # Root->WeightedNode(1.1 default) -> updated to 1.2 by colon.
        #   Child: TextNode(foo) -> gets 1.2
        #   Child: WeightedNode(0.9 default) -> updated to 0.5 by colon.
        #       Child: TextNode(bar) -> gets 0.5 * 1.2 = 0.6.

        # Test: (foo [bar:0.5]:1.2)
        # foo: 1.2
        # bar: 0.5 * 1.2 = 0.6
        # Matches: (foo )1.20(bar)0.60
        result = convert_a1111_to_compel("(foo [bar:0.5]:1.2)")
        assert "(foo )1.20" in result
        assert "(bar)0.60" in result


class TestParsePrompt:
    """Tests for the main parse_prompt function."""

    def test_lpw_mode(self):
        """Test parsing in LPW mode (A1111 syntax converted to Compel)."""
        result = parse_prompt("(test:1.2)", PromptParser.LPW)
        assert "(test)1.20" in result.chunks[0][0].text

    def test_compel_mode(self):
        """Test parsing in compel mode keeps raw text."""
        result = parse_prompt("(test)1.2", PromptParser.COMPEL)
        # Compel mode should keep the raw text
        assert result.chunks[0][0].text == "(test)1.2"
        assert result.chunks[0][0].weight == 1.0
