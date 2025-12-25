"""Prompt parser for weighted prompts.

Supports two modes:
- Legacy: StableDiffusionWebUI compatible syntax
- Compel: Compel library syntax

Legacy syntax examples:
    (word:1.2)           # Weight 1.2
    ((word))             # Weight 1.1^2 = 1.21
    [word]               # Weight 0.9
    (multiple words:1.3) # Weight 1.3 for multiple words
    BREAK                # Chunk separator (encode separately)

Compel syntax examples:
    (word)1.2            # Weight 1.2
    word++               # Increase weight
    word--               # Decrease weight
"""

import re
from dataclasses import dataclass
from enum import Enum


class ParserMode(str, Enum):
    """Prompt parser mode."""

    LEGACY = "legacy"
    COMPEL = "compel"


@dataclass
class WeightedChunk:
    """A chunk of text with associated weight."""

    text: str
    weight: float = 1.0


@dataclass
class ParsedPrompt:
    """Parsed prompt with chunks separated by BREAK."""

    chunks: list[list[WeightedChunk]]  # List of chunks, each chunk is a list of weighted parts


# Legacy syntax patterns
LEGACY_WEIGHT_PATTERN = re.compile(r"\(([^()]+):([0-9.]+)\)")  # (text:weight)
LEGACY_EMPHASIS_PATTERN = re.compile(r"\(([^()]+)\)")  # (text) - increase weight
LEGACY_DEEMPHASIS_PATTERN = re.compile(r"\[([^\[\]]+)\]")  # [text] - decrease weight


def _count_nested_parens(text: str, open_char: str, close_char: str) -> int:
    """Count nested parentheses/brackets."""
    count = 0
    for char in text:
        if char == open_char:
            count += 1
        elif char == close_char:
            break
    return count


def _parse_legacy_weights(text: str, base_weight: float = 1.0) -> list[WeightedChunk]:
    """Parse legacy weight syntax recursively.

    Args:
        text: Text to parse
        base_weight: Base weight multiplier

    Returns:
        List of weighted chunks
    """
    chunks: list[WeightedChunk] = []

    # First, handle explicit weights: (text:1.2)
    last_end = 0
    for match in LEGACY_WEIGHT_PATTERN.finditer(text):
        # Add text before match
        if match.start() > last_end:
            before = text[last_end : match.start()]
            if before.strip():
                # Recursively parse for nested emphasis
                chunks.extend(_parse_legacy_emphasis(before, base_weight))

        # Add weighted chunk
        inner_text = match.group(1).strip()
        weight = float(match.group(2)) * base_weight
        if inner_text:
            chunks.append(WeightedChunk(text=inner_text, weight=weight))

        last_end = match.end()

    # Handle remaining text
    if last_end < len(text):
        remaining = text[last_end:]
        if remaining.strip():
            chunks.extend(_parse_legacy_emphasis(remaining, base_weight))

    return chunks if chunks else [WeightedChunk(text=text.strip(), weight=base_weight)]


def _parse_legacy_emphasis(text: str, base_weight: float = 1.0) -> list[WeightedChunk]:
    """Parse emphasis (parentheses) and de-emphasis (brackets).

    ((text)) -> weight * 1.1^2
    [text] -> weight * 0.9
    """
    chunks: list[WeightedChunk] = []
    i = 0

    while i < len(text):
        if text[i] == "(":
            # Find matching close paren
            depth = 1
            j = i + 1
            while j < len(text) and depth > 0:
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                j += 1

            if depth == 0:
                inner = text[i + 1 : j - 1]
                # Count how many nested parens at the start
                paren_count = 1
                while inner.startswith("(") and inner.endswith(")"):
                    inner = inner[1:-1]
                    paren_count += 1

                # Check if it's an explicit weight
                weight_match = re.match(r"^(.+):([0-9.]+)$", inner)
                if weight_match:
                    inner_text = weight_match.group(1).strip()
                    weight = float(weight_match.group(2)) * base_weight
                    if inner_text:
                        chunks.append(WeightedChunk(text=inner_text, weight=weight))
                else:
                    # Emphasis: multiply by 1.1 for each level
                    new_weight = base_weight * (1.1**paren_count)
                    if inner.strip():
                        chunks.append(WeightedChunk(text=inner.strip(), weight=new_weight))
                i = j
                continue

        elif text[i] == "[":
            # Find matching close bracket
            depth = 1
            j = i + 1
            while j < len(text) and depth > 0:
                if text[j] == "[":
                    depth += 1
                elif text[j] == "]":
                    depth -= 1
                j += 1

            if depth == 0:
                inner = text[i + 1 : j - 1]
                # Count nested brackets
                bracket_count = 1
                while inner.startswith("[") and inner.endswith("]"):
                    inner = inner[1:-1]
                    bracket_count += 1

                # De-emphasis: multiply by 0.9 for each level
                new_weight = base_weight * (0.9**bracket_count)
                if inner.strip():
                    chunks.append(WeightedChunk(text=inner.strip(), weight=new_weight))
                i = j
                continue

        # Regular text - find next special char
        j = i
        while j < len(text) and text[j] not in "()[]":
            j += 1

        if j > i:
            part = text[i:j].strip()
            if part:
                chunks.append(WeightedChunk(text=part, weight=base_weight))
        i = j if j > i else i + 1

    return chunks if chunks else [WeightedChunk(text=text.strip(), weight=base_weight)]


def parse_legacy_prompt(prompt: str) -> ParsedPrompt:
    """Parse prompt using legacy (StableDiffusionWebUI) syntax.

    Args:
        prompt: Raw prompt string

    Returns:
        ParsedPrompt with chunks and weights
    """
    # Split by BREAK
    raw_chunks = re.split(r"\bBREAK\b", prompt, flags=re.IGNORECASE)

    chunks: list[list[WeightedChunk]] = []
    for raw_chunk in raw_chunks:
        raw_chunk = raw_chunk.strip()
        if raw_chunk:
            weighted_parts = _parse_legacy_weights(raw_chunk)
            if weighted_parts:
                chunks.append(weighted_parts)

    # Ensure at least one empty chunk if prompt was empty
    if not chunks:
        chunks = [[WeightedChunk(text="", weight=1.0)]]

    return ParsedPrompt(chunks=chunks)


def parse_compel_prompt(prompt: str) -> ParsedPrompt:
    """Parse prompt for Compel library.

    For Compel, we do minimal parsing since Compel handles its own syntax.
    We only split by BREAK for chunk separation.

    Args:
        prompt: Raw prompt string

    Returns:
        ParsedPrompt with chunks (Compel will handle weights)
    """
    # Split by BREAK
    raw_chunks = re.split(r"\bBREAK\b", prompt, flags=re.IGNORECASE)

    chunks: list[list[WeightedChunk]] = []
    for raw_chunk in raw_chunks:
        raw_chunk = raw_chunk.strip()
        if raw_chunk:
            # For Compel, keep the raw text - Compel will parse weights
            chunks.append([WeightedChunk(text=raw_chunk, weight=1.0)])

    if not chunks:
        chunks = [[WeightedChunk(text="", weight=1.0)]]

    return ParsedPrompt(chunks=chunks)


def parse_prompt(prompt: str, mode: ParserMode = ParserMode.LEGACY) -> ParsedPrompt:
    """Parse prompt according to specified mode.

    Args:
        prompt: Raw prompt string
        mode: Parser mode (legacy or compel)

    Returns:
        ParsedPrompt with chunks and weights
    """
    if mode == ParserMode.LEGACY:
        return parse_legacy_prompt(prompt)
    else:
        return parse_compel_prompt(prompt)
