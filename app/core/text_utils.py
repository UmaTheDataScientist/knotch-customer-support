"""Defensive Markdown stripping for API responses.

Prompt instructions ("respond in plain text") are not a guarantee -- real
models occasionally ignore formatting instructions anyway. Since the
`response` field in ChatMessageOut is a raw string displayed as-is (not
rendered through a Markdown renderer), any Markdown syntax that slips
through would show up as literal asterisks/hashes/backticks to the user.

This module is a deterministic, dependency-free regex-based cleanup applied
to every final response right before it's returned -- regardless of whether
the text came from an LLM or from one of our own hardcoded template strings
(for which it's simply a no-op, since there's nothing to strip).
"""
from __future__ import annotations

import re

_BOLD_DOUBLE_STAR = re.compile(r"\*\*(.+?)\*\*")
_BOLD_DOUBLE_UNDERSCORE = re.compile(r"__(.+?)__")
_ITALIC_SINGLE_STAR = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")
_ITALIC_SINGLE_UNDERSCORE = re.compile(r"(?<!_)_([^_\n]+?)_(?!_)")
_INLINE_CODE = re.compile(r"`([^`]+)`")
_CODE_FENCE = re.compile(r"```[a-zA-Z]*\n?")
_HEADER = re.compile(r"^\s{0,3}#{1,6}\s+", flags=re.MULTILINE)
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BLOCKQUOTE = re.compile(r"^\s{0,3}>\s?", flags=re.MULTILINE)


def strip_markdown(text: str) -> str:
    """Removes common Markdown syntax, keeping the underlying text intact.

    Idempotent and safe to call on plain text with no Markdown at all
    (which is most of our tool-generated responses -- clarification
    questions, refusals, status lookups -- this is a no-op for those).
    """
    if not text:
        return text

    result = text
    result = _CODE_FENCE.sub("", result)
    result = _MARKDOWN_LINK.sub(r"\1 (\2)", result)
    result = _BOLD_DOUBLE_STAR.sub(r"\1", result)
    result = _BOLD_DOUBLE_UNDERSCORE.sub(r"\1", result)
    result = _INLINE_CODE.sub(r"\1", result)
    result = _HEADER.sub("", result)
    result = _BLOCKQUOTE.sub("", result)
    # Italics last: stripping bold first avoids a stray single '*' left
    # behind by "**bold**" being mistaken for the start of an italic run.
    result = _ITALIC_SINGLE_STAR.sub(r"\1", result)
    result = _ITALIC_SINGLE_UNDERSCORE.sub(r"\1", result)

    # Collapse any blank-line runs left behind by removed header/fence lines.
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()
