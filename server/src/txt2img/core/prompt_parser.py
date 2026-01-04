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
    \\(word\\)             # Escaped parenthesis (literal text)

Compel syntax examples:
    (word)1.2            # Weight 1.2
    word++               # Increase weight
    word--               # Decrease weight
    .and()               # Conjunction
    .blend()             # Blending

Note: LPW mode parses A1111 syntax and converts to Compel format internally.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union


# --- Data Classes ---

@dataclass
class WeightedChunk:
    """A chunk of text with associated weight.
    Used for backwards compatibility with ParsedPrompt."""
    text: str
    weight: float = 1.0


@dataclass
class ParsedPrompt:
    """Parsed prompt with chunks separated by BREAK."""
    chunks: list[list[WeightedChunk]]  # List of chunks, each chunk is a list of weighted parts


class PromptParser(str, Enum):
    """Prompt parser mode."""
    LPW = "lpw"  # A1111/WebUI compatible (converted to Compel internally)
    COMPEL = "compel"  # Native Compel syntax


# --- AST Nodes ---

class Node:
    """Base AST node."""
    pass


@dataclass
class TextNode(Node):
    """Leaf node representing raw text."""
    text: str


@dataclass
class WeightedNode(Node):
    """Node representing a weighted group of children."""
    children: List[Node]
    weight_multiplier: float = 1.0


# --- Tokenizer ---

class TokenType(Enum):
    TEXT = "text"
    LPAREN = "("
    RPAREN = ")"
    LBRACKET = "["
    RBRACKET = "]"
    COLON = ":"
    COMMA = ","
    ESCAPE = "escape"  # Internal use logic


def tokenize(text: str) -> List[tuple[TokenType, str]]:
    """Tokenize prompt string."""
    tokens = []
    i = 0
    length = len(text)

    # buffer for accumulating text
    text_buffer = []

    def flush_buffer():
        if text_buffer:
            tokens.append((TokenType.TEXT, "".join(text_buffer)))
            text_buffer.clear()

    while i < length:
        char = text[i]

        if char == "\\":
            if i + 1 < length:
                # Escape sequence: keep the escaped character as text
                text_buffer.append(text[i + 1])
                i += 2
            else:
                # Trailing backslash
                text_buffer.append(char)
                i += 1
            continue

        if char == "(":
            flush_buffer()
            tokens.append((TokenType.LPAREN, "("))
        elif char == ")":
            flush_buffer()
            tokens.append((TokenType.RPAREN, ")"))
        elif char == "[":
            flush_buffer()
            tokens.append((TokenType.LBRACKET, "["))
        elif char == "]":
            flush_buffer()
            tokens.append((TokenType.RBRACKET, "]"))
        elif char == ":":
            flush_buffer()
            tokens.append((TokenType.COLON, ":"))
        else:
            text_buffer.append(char)

        i += 1

    flush_buffer()
    return tokens


# --- Parser ---

def parse_to_ast(tokens: List[tuple[TokenType, str]]) -> WeightedNode:
    """Parse tokens into an AST."""
    # Root node
    root = WeightedNode(children=[], weight_multiplier=1.0)

    # Stack stores (node, closing_token_type)
    stack: List[tuple[WeightedNode, Optional[TokenType]]] = [(root, None)]

    i = 0
    while i < len(tokens):
        token_type, token_value = tokens[i]

        if token_type == TokenType.TEXT:
            stack[-1][0].children.append(TextNode(text=token_value))

        elif token_type == TokenType.LPAREN:
            # Start of ( ... ) group
            node = WeightedNode(children=[], weight_multiplier=1.1) # Default emphasis
            stack[-1][0].children.append(node)
            stack.append((node, TokenType.RPAREN))

        elif token_type == TokenType.LBRACKET:
            # Start of [ ... ] group
            node = WeightedNode(children=[], weight_multiplier=0.9) # Default de-emphasis
            stack[-1][0].children.append(node)
            stack.append((node, TokenType.RBRACKET))

        elif token_type in (TokenType.RPAREN, TokenType.RBRACKET):
            # Check if this closes the current group
            if len(stack) > 1 and stack[-1][1] == token_type:
                # Check for explicit weight immediately following: (text):1.5
                # Lookahead for : then numbers
                has_explicit_weight = False
                weight_val = 1.0

                # Simple weight parser lookahead
                j = i + 1
                if j < len(tokens) and tokens[j][0] == TokenType.COLON:
                    # Found colon, look for text that looks like a number
                    k = j + 1
                    if k < len(tokens) and tokens[k][0] == TokenType.TEXT:
                        # Try parsing number
                        try:
                            # It might be "1.5" or "1.5, next"
                            # We only want the number part if it's strictly a number or at end?
                            # A1111 allows (text:1.5)
                            # Here we are at (text):1.5 which is valid too?
                            # Actually (text):1.5 is NOT standard A1111 text emphasis syntax usually.
                            # Standard is (text:1.5).
                            # But WAIT, we just closed a group.
                            # If we had (text:1.5), we would encounter COLON *inside*.
                            pass
                        except ValueError:
                            pass

                # Pop the stack
                finished_node, _ = stack.pop()

            else:
                # Mismatched or extra closing bracket, treat as text
                stack[-1][0].children.append(TextNode(text=token_value))

        elif token_type == TokenType.COLON:
            # Colon encountered.
            # If we are inside a group, this might be a weight separator for THAT group.
            # e.g. (text:1.5)
            # We need to see if the REST of the group is just a number.

            # Use simple heuristic: if we are in a group (len(stack)>1),
            # and what follows until the closing token is a number, then set weight.

            if len(stack) > 1:
                current_node, closing_type = stack[-1]

                # Check remaining tokens until matching closer
                # lookahead
                temp_j = i + 1
                is_valid_weight = False
                weight_val = 1.0
                consumed_tokens = 0

                # We expect [NUMBER] then [CLOSER]
                if temp_j < len(tokens) and tokens[temp_j][0] == TokenType.TEXT:
                    try:
                        possible_weight = float(tokens[temp_j][1].strip())
                        # Check next token
                        next_j = temp_j + 1
                        if next_j < len(tokens) and tokens[next_j][0] == closing_type:
                            # Yes, it is ( ... : 1.5 )
                            is_valid_weight = True
                            weight_val = possible_weight
                            consumed_tokens = 1 # The number token
                        elif next_j >= len(tokens):
                             # EOF? weird but ok
                             is_valid_weight = True
                             weight_val = possible_weight
                             consumed_tokens = 1
                    except ValueError:
                        pass

                if is_valid_weight:
                    # Modify the current node's weight
                    # Note: The current node started with default weight (1.1 or 0.9).
                    # If explicit weight is provided, it OVERRIDES the default multiplier logic check?
                    # No, in A1111:
                    # (text) -> 1.1
                    # (text:1.5) -> 1.5 (Total weight 1.5, not 1.1*1.5)
                    # [text] -> 0.9
                    # [text:0.5] -> 0.5

                    current_node.weight_multiplier = weight_val
                    i += consumed_tokens # Skip the number
                    # The loop will increment i one more time, and next iteration handles closer
                else:
                    # Colon is just text
                    stack[-1][0].children.append(TextNode(text=":"))
            else:
                 # Colon at top level is just text
                 stack[-1][0].children.append(TextNode(text=":"))

        else:
            # Should not happen with current tokenizer
            pass

        i += 1

    return root


# --- Compel Converter ---

def _convert_ast_to_compel(node: Node, accumulated_weight: float = 1.0) -> List[str]:
    """Recursively convert AST to Compel fragments."""
    fragments = []

    if isinstance(node, TextNode):
        if not node.text:
            return []

        # Escape parentheses to prevent Compel from interpreting them as syntax
        # We also need to escape + and - if Compel uses them?
        # Compel uses +++ and ---.
        # But primarily parens are the structural elements.
        clean_text = node.text.replace("(", r"\(").replace(")", r"\)")

        # If absolute weight is 1.0, just output the safely escaped text
        if abs(accumulated_weight - 1.0) < 0.001:
            fragments.append(clean_text)
        else:
            fragments.append(f"({clean_text}){accumulated_weight:.2f}")

    elif isinstance(node, WeightedNode):
        new_weight = accumulated_weight * node.weight_multiplier
        for child in node.children:
            fragments.extend(_convert_ast_to_compel(child, new_weight))

    return fragments


def convert_a1111_to_compel(prompt: str) -> str:
    """Convert A1111 prompt syntax to Compel syntax.

    A1111:  (word:1.5)   -> Compel: (word)1.5
    A1111:  ((word))     -> Compel: (word)1.21
    A1111:  [word]       -> Compel: (word)0.9
    A1111:  \\(word\\)     -> Compel: \\(word\\)

    Args:
        prompt: A1111 formatted prompt

    Returns:
        Compel formatted prompt
    """
    tokens = tokenize(prompt)
    ast = parse_to_ast(tokens)
    fragments = _convert_ast_to_compel(ast)

    # Post-process: Join fragments.
    # We need spaces between fragments?
    # AST preserved structure. TextNodes were creating from split buffer.
    # tokens did not discard spaces.
    # So direct join should work.

    return "".join(fragments)


def normalize_grouped_weights(prompt: str) -> str:
    """Legacy function, no longer strictly needed but kept for interface compatibility.
    The new parser handles grouping naturally via recursion.
    However, if we want to ensure (a, b:1.1) becomes (a:1.1), (b:1.1) explicitly BEFORE parsing?
    No, the new parser converts (a, b:1.1) to (a, b)1.1 in Compel.
    Compel treats "a, b" with weight 1.1 correctly (weights the whole sequence).
    So we can just return the prompt as is, OR implement the explicit split if preferred.

    Returning as is for now as it's cleaner.
    """
    return prompt


# --- Legacy Interfaces for Compatibility ---

def parse_a1111_prompt(prompt: str) -> ParsedPrompt:
    """Legacy parser using new backend."""
    # Split by BREAK first
    raw_chunks = re.split(r"\bBREAK\b", prompt, flags=re.IGNORECASE)

    chunks: list[list[WeightedChunk]] = []
    for raw_chunk in raw_chunks:
        raw_chunk = raw_chunk.strip()
        if raw_chunk:
            # Convert to Compel string to satisfy tests and ensure consistent behavior
            compel_text = convert_a1111_to_compel(raw_chunk)
            chunks.append([WeightedChunk(text=compel_text, weight=1.0)])

    if not chunks:
        chunks = [[WeightedChunk(text="", weight=1.0)]]

    return ParsedPrompt(chunks=chunks)


def parse_lpw_prompt(prompt: str) -> ParsedPrompt:
    """Parse prompt using LPW (A1111) syntax."""
    # Normalize?
    # prompt = normalize_grouped_weights(prompt) # No longer needed

    # Split by BREAK
    raw_chunks = re.split(r"\bBREAK\b", prompt, flags=re.IGNORECASE)

    chunks: list[list[WeightedChunk]] = []
    for raw_chunk in raw_chunks:
        raw_chunk = raw_chunk.strip()
        if raw_chunk:
            # Convert to Compel string
            compel_text = convert_a1111_to_compel(raw_chunk)
            chunks.append([WeightedChunk(text=compel_text, weight=1.0)])

    if not chunks:
        chunks = [[WeightedChunk(text="", weight=1.0)]]

    return ParsedPrompt(chunks=chunks)


def parse_compel_prompt(prompt: str) -> ParsedPrompt:
    """Parse prompt using Compel syntax."""
    raw_chunks = re.split(r"\bBREAK\b", prompt, flags=re.IGNORECASE)
    chunks: list[list[WeightedChunk]] = []
    for raw_chunk in raw_chunks:
        raw_chunk = raw_chunk.strip()
        if raw_chunk:
            chunks.append([WeightedChunk(text=raw_chunk, weight=1.0)])

    if not chunks:
         chunks = [[WeightedChunk(text="", weight=1.0)]]

    return ParsedPrompt(chunks=chunks)


def parse_prompt(prompt: str, mode: PromptParser = PromptParser.LPW) -> ParsedPrompt:
    """Parse prompt according to specified mode."""
    if mode == PromptParser.LPW:
        return parse_lpw_prompt(prompt)
    else:
        return parse_compel_prompt(prompt)
