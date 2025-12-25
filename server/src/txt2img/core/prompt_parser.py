"""Prompt parser for weighted prompts.

Supports two modes:
- LPW: Long Prompt Weighting (A1111/StableDiffusionWebUI compatible syntax)
- Compel: Compel library native syntax

LPW (A1111) syntax examples:
    (word:1.2)           # Weight 1.2
    ((word))             # Weight 1.1^2 = 1.21
    [word]               # Weight 0.9
    (multiple words:1.3) # Weight 1.3 for multiple words
    BREAK                # Chunk separator (encode separately)

Compel syntax examples:
    (word)1.2            # Weight 1.2
    word++               # Increase weight
    word--               # Decrease weight
    .and()               # Conjunction
    .blend()             # Blending

Note: LPW mode parses A1111 syntax and converts to Compel format internally.
"""

import re
from dataclasses import dataclass
from enum import Enum


class PromptParser(str, Enum):
    """Prompt parser mode."""

    LPW = "lpw"  # A1111/WebUI compatible (converted to Compel internally)
    COMPEL = "compel"  # Native Compel syntax


@dataclass
class WeightedChunk:
    """A chunk of text with associated weight."""

    text: str
    weight: float = 1.0


@dataclass
class ParsedPrompt:
    """Parsed prompt with chunks separated by BREAK."""

    chunks: list[list[WeightedChunk]]  # List of chunks, each chunk is a list of weighted parts


# A1111/Legacy syntax patterns
A1111_WEIGHT_PATTERN = re.compile(r"\(([^()]+):([0-9.]+)\)")  # (text:weight)
A1111_BRACKET_WEIGHT_PATTERN = re.compile(
    r"\[([^\[\]]+):([0-9.]+)\]"
)  # [text:weight] (de-emphasis with weight)


def _parse_a1111_emphasis(text: str, base_weight: float = 1.0) -> list[WeightedChunk]:
    """Parse A1111 emphasis (parentheses) and de-emphasis (brackets).

    ((text)) -> weight * 1.1^2
    [text] -> weight * 0.9
    (text:1.5) -> explicit weight
    [text:0.5] -> explicit weight (de-emphasis variant)
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
                # Check for explicit weight first: [text:0.5]
                weight_match = re.match(r"^(.+):([0-9.]+)$", inner)
                if weight_match:
                    inner_text = weight_match.group(1).strip()
                    weight = float(weight_match.group(2)) * base_weight
                    if inner_text:
                        chunks.append(WeightedChunk(text=inner_text, weight=weight))
                else:
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


def parse_a1111_prompt(prompt: str) -> ParsedPrompt:
    """Parse prompt using A1111 (StableDiffusionWebUI) syntax.

    Args:
        prompt: Raw prompt string with A1111 syntax

    Returns:
        ParsedPrompt with chunks and weights
    """
    # Split by BREAK
    raw_chunks = re.split(r"\bBREAK\b", prompt, flags=re.IGNORECASE)

    chunks: list[list[WeightedChunk]] = []
    for raw_chunk in raw_chunks:
        raw_chunk = raw_chunk.strip()
        if raw_chunk:
            weighted_parts = _parse_a1111_emphasis(raw_chunk)
            if weighted_parts:
                chunks.append(weighted_parts)

    # Ensure at least one empty chunk if prompt was empty
    if not chunks:
        chunks = [[WeightedChunk(text="", weight=1.0)]]

    return ParsedPrompt(chunks=chunks)


def convert_a1111_to_compel(prompt: str) -> str:
    """Convert A1111 prompt syntax to Compel syntax.

    A1111:  (word:1.5)   -> Compel: (word)1.5
    A1111:  ((word))     -> Compel: (word)1.21
    A1111:  [word]       -> Compel: (word)0.9
    A1111:  [[[word]]]   -> Compel: (word)0.729
    A1111:  BREAK        -> (handled separately, not converted here)

    Args:
        prompt: A1111 formatted prompt

    Returns:
        Compel formatted prompt
    """
    # Parse A1111 and convert to Compel format
    # Don't split by BREAK here - caller should handle BREAK chunks
    parsed = _parse_a1111_emphasis(prompt)

    compel_parts = []
    for chunk in parsed:
        if chunk.weight == 1.0:
            compel_parts.append(chunk.text)
        else:
            # Compel format: (text)weight
            compel_parts.append(f"({chunk.text}){chunk.weight:.2f}")

    return " ".join(compel_parts)


def parse_compel_prompt(prompt: str) -> ParsedPrompt:
    """Parse prompt for Compel library.

    For Compel, we do minimal parsing since Compel handles its own syntax.
    We only split by BREAK for chunk separation.

    Args:
        prompt: Raw prompt string with Compel syntax

    Returns:
        ParsedPrompt with chunks (Compel will handle weights internally)
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


def parse_lpw_prompt(prompt: str) -> ParsedPrompt:
    """Parse prompt using LPW (A1111) syntax and prepare for Compel.

    This parses A1111 syntax and converts each chunk to Compel format.

    Args:
        prompt: Raw prompt string with A1111 syntax

    Returns:
        ParsedPrompt with Compel-converted chunks
    """
    # Split by BREAK
    raw_chunks = re.split(r"\bBREAK\b", prompt, flags=re.IGNORECASE)

    chunks: list[list[WeightedChunk]] = []
    for raw_chunk in raw_chunks:
        raw_chunk = raw_chunk.strip()
        if raw_chunk:
            # Convert A1111 chunk to Compel format
            compel_text = convert_a1111_to_compel(raw_chunk)
            # Return as single weighted chunk (Compel handles the rest)
            chunks.append([WeightedChunk(text=compel_text, weight=1.0)])

    if not chunks:
        chunks = [[WeightedChunk(text="", weight=1.0)]]

    return ParsedPrompt(chunks=chunks)


def parse_prompt(prompt: str, mode: PromptParser = PromptParser.LPW) -> ParsedPrompt:
    """Parse prompt according to specified mode.

    Args:
        prompt: Raw prompt string
        mode: Parser mode (lpw or compel)

    Returns:
        ParsedPrompt with chunks and weights
    """
    if mode == PromptParser.LPW:
        return parse_lpw_prompt(prompt)
    else:
        return parse_compel_prompt(prompt)
