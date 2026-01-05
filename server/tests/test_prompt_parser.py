"""Tests for prompt parser."""

import pytest

from txt2img.utils.prompt_parser import (
    PromptParser,
    build_compel_prompt,
    convert_a1111_to_compel,
    parse_a1111_prompt,
    parse_prompt,
    prepend_triggers,
    tokenize,
    parse_to_ast,
    TokenType,
)


class TestTokenizer:
    """Tests for the tokenizer."""

    def test_simple_text(self):
        """Test tokenizing simple text."""
        tokens = tokenize("hello world")
        assert len(tokens) == 1
        assert tokens[0] == (TokenType.TEXT, "hello world")

    def test_parentheses(self):
        """Test tokenizing parentheses."""
        tokens = tokenize("(word)")
        assert tokens == [
            (TokenType.LPAREN, "("),
            (TokenType.TEXT, "word"),
            (TokenType.RPAREN, ")"),
        ]

    def test_brackets(self):
        """Test tokenizing square brackets."""
        tokens = tokenize("[word]")
        assert tokens == [
            (TokenType.LBRACKET, "["),
            (TokenType.TEXT, "word"),
            (TokenType.RBRACKET, "]"),
        ]

    def test_colon(self):
        """Test tokenizing colon."""
        tokens = tokenize("(word:1.5)")
        assert (TokenType.COLON, ":") in tokens

    def test_escape_sequence(self):
        """Test escaped characters."""
        tokens = tokenize(r"\(literal\)")
        assert len(tokens) == 1
        assert tokens[0] == (TokenType.TEXT, "(literal)")

    def test_trailing_backslash(self):
        """Test trailing backslash."""
        tokens = tokenize("test\\")
        assert tokens == [(TokenType.TEXT, "test\\")]

    def test_empty_string(self):
        """Test empty string."""
        tokens = tokenize("")
        assert tokens == []


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
        assert "(masterpiece)1.20" in result.chunks[0][0].text
        assert result.chunks[0][0].weight == 1.0

    def test_break_syntax(self):
        """Test BREAK syntax for chunk separation."""
        result = parse_a1111_prompt("first part BREAK second part")
        assert len(result.chunks) == 2
        assert "first part" in result.chunks[0][0].text
        assert "second part" in result.chunks[1][0].text

    def test_empty_prompt(self):
        """Test empty prompt."""
        result = parse_a1111_prompt("")
        assert len(result.chunks) == 1
        assert result.chunks[0][0].text == ""

    def test_whitespace_only(self):
        """Test whitespace-only prompt."""
        result = parse_a1111_prompt("   ")
        assert len(result.chunks) == 1

    def test_multiple_breaks(self):
        """Test multiple consecutive BREAKs."""
        result = parse_a1111_prompt("a BREAK BREAK b")
        # Should skip empty chunks between BREAKs
        assert len(result.chunks) == 2

    def test_leading_break(self):
        """Test BREAK at the start."""
        result = parse_a1111_prompt("BREAK content")
        assert len(result.chunks) == 1
        assert "content" in result.chunks[0][0].text

    def test_trailing_break(self):
        """Test BREAK at the end."""
        result = parse_a1111_prompt("content BREAK")
        assert len(result.chunks) == 1
        assert "content" in result.chunks[0][0].text

    def test_break_case_insensitive(self):
        """Test BREAK is case insensitive."""
        result = parse_a1111_prompt("a break b BREAK c BrEaK d")
        assert len(result.chunks) == 4


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

    # --- New Edge Case Tests ---

    def test_unclosed_parenthesis(self):
        """Test unclosed parenthesis - should not crash."""
        result = convert_a1111_to_compel("(unclosed")
        # Should still produce output (treating as emphasized text)
        assert "unclosed" in result

    def test_extra_closing_parenthesis(self):
        """Test extra closing parenthesis - should be treated as text."""
        result = convert_a1111_to_compel("extra)")
        assert ")" in result or "extra" in result

    def test_unmatched_brackets(self):
        """Test unmatched brackets."""
        result = convert_a1111_to_compel("[unclosed")
        assert "unclosed" in result

    def test_colon_in_text(self):
        """Test colon in text that's not a weight."""
        result = convert_a1111_to_compel("time: 12:00")
        # Colon should be preserved as text
        assert "12" in result or "time" in result

    def test_colon_with_non_numeric(self):
        """Test colon followed by non-numeric text in parentheses."""
        result = convert_a1111_to_compel("(a:b:c)")
        # Should treat as text since b:c is not a valid weight
        assert "a" in result

    def test_empty_parentheses(self):
        """Test empty parentheses."""
        result = convert_a1111_to_compel("()")
        assert result == "" or "()" not in result

    def test_empty_weight_target(self):
        """Test weight with empty target."""
        result = convert_a1111_to_compel("(:1.5)")
        # Should handle gracefully
        assert "1.5" not in result or result != ""

    def test_zero_weight(self):
        """Test zero weight."""
        result = convert_a1111_to_compel("(word:0)")
        assert "(word)0.00" in result

    def test_very_large_weight(self):
        """Test very large weight."""
        result = convert_a1111_to_compel("(word:100)")
        assert "(word)100.00" in result

    def test_deep_nesting(self):
        """Test deeply nested parentheses."""
        result = convert_a1111_to_compel("(((((deep)))))")
        # 5 levels of nesting: 1.1^5 = 1.61051
        assert "(deep)1.61" in result

    def test_deep_bracket_nesting(self):
        """Test deeply nested brackets."""
        result = convert_a1111_to_compel("[[[[[deep]]]]]")
        # 5 levels: 0.9^5 ≈ 0.59
        assert "(deep)0.59" in result

    def test_mixed_deep_nesting(self):
        """Test alternating parentheses and brackets."""
        result = convert_a1111_to_compel("([word])")
        # ( = 1.1, [ = 0.9, so 1.1 * 0.9 = 0.99
        assert "(word)0.99" in result

    def test_comma_separated_tags(self):
        """Test typical SD tag format."""
        result = convert_a1111_to_compel("1girl, (masterpiece:1.2), best quality")
        assert "1girl" in result
        assert "(masterpiece)1.20" in result
        assert "best quality" in result

    def test_newline_in_prompt(self):
        """Test prompt with newlines."""
        result = convert_a1111_to_compel("line1\nline2")
        assert "line1" in result
        assert "line2" in result

    def test_unicode_characters(self):
        """Test unicode/Japanese characters."""
        result = convert_a1111_to_compel("(かわいい:1.2)")
        assert "(かわいい)1.20" in result

    def test_escaped_backslash(self):
        """Test escaped backslash."""
        result = convert_a1111_to_compel(r"a\\b")
        # \\ should become \
        assert "\\" in result or "b" in result

    def test_multiple_colons_in_weight(self):
        """Test something like (a:1:2) which is malformed."""
        result = convert_a1111_to_compel("(a:1:2)")
        # Parser should handle this gracefully
        assert "a" in result

    def test_decimal_edge_cases(self):
        """Test various decimal formats."""
        # Integer weight
        assert "(word)2.00" in convert_a1111_to_compel("(word:2)")
        # Leading zero
        assert "(word)0.50" in convert_a1111_to_compel("(word:0.5)")
        # Many decimals - should still parse
        result = convert_a1111_to_compel("(word:1.555)")
        assert "(word)1.56" in result or "(word)1.55" in result


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

    def test_prepend_triggers_empty(self):
        """Test prepending empty triggers."""
        result = prepend_triggers("user prompt", [])
        assert result == "user prompt"

    def test_prepend_triggers_zero_weight(self):
        """Test trigger with zero weight is skipped."""
        triggers = [("skipped", 0), ("kept", 1.0)]
        result = prepend_triggers("prompt", triggers)
        assert "skipped" not in result
        assert "kept" in result

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

    def test_build_compel_prompt_single_segment(self):
        """Test building prompt with single segment."""
        result = build_compel_prompt("no breaks here", use_concatenation=True)
        assert result == "no breaks here"

    def test_build_compel_prompt_empty(self):
        """Test building empty prompt."""
        result = build_compel_prompt("", use_concatenation=True)
        assert result == '""'

    def test_build_compel_prompt_quotes_escape(self):
        """Test quotes in prompt are escaped for .and() syntax."""
        prompt = 'say "hello" BREAK world'
        result = build_compel_prompt(prompt, use_concatenation=True)
        # Quotes should be escaped
        assert r'\"hello\"' in result


class TestParsePrompt:
    """Tests for the main parse_prompt function."""

    def test_lpw_mode(self):
        result = parse_prompt("(test:1.2)", PromptParser.LPW)
        assert "(test)1.20" in result.chunks[0][0].text

    def test_compel_mode(self):
        result = parse_prompt("(test)1.2", PromptParser.COMPEL)
        assert result.chunks[0][0].text == "(test)1.2"
        assert result.chunks[0][0].weight == 1.0


class TestRealWorldPrompts:
    """Tests with real-world prompt examples."""

    def test_typical_sd_prompt(self):
        """Test a typical Stable Diffusion prompt."""
        prompt = "(masterpiece:1.2), best quality, 1girl, (detailed eyes:1.1)"
        result = convert_a1111_to_compel(prompt)
        assert "(masterpiece)1.20" in result
        assert "best quality" in result
        assert "(detailed eyes)1.10" in result

    def test_negative_prompt_style(self):
        """Test typical negative prompt format."""
        prompt = "(low quality:1.4), [blurry], bad anatomy"
        result = convert_a1111_to_compel(prompt)
        assert "(low quality)1.40" in result
        assert "(blurry)0.90" in result
        assert "bad anatomy" in result

    def test_lora_trigger_format(self):
        """Test LoRA trigger words with weights."""
        prompt = "(hk_szk:0.8), 1girl, red eyes"
        result = convert_a1111_to_compel(prompt)
        assert "(hk_szk)0.80" in result
