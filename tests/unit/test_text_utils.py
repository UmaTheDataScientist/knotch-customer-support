from __future__ import annotations

from app.core.text_utils import strip_markdown


def test_strips_double_star_bold():
    assert strip_markdown("Go to **Settings** now") == "Go to Settings now"


def test_strips_the_exact_real_bug_case():
    # This is the literal text that surfaced the bug during live testing.
    original = (
        "To find your API key, please navigate to **Settings** -> **Developer** "
        "-> **API Keys**. From there, you can create and copy a secret key."
    )
    expected = (
        "To find your API key, please navigate to Settings -> Developer "
        "-> API Keys. From there, you can create and copy a secret key."
    )
    assert strip_markdown(original) == expected


def test_strips_double_underscore_bold():
    assert strip_markdown("This is __important__ info") == "This is important info"


def test_strips_markdown_headers():
    assert strip_markdown("# Password Reset\nGo to settings.") == "Password Reset\nGo to settings."


def test_strips_inline_code():
    assert strip_markdown("Run `git status` first") == "Run git status first"


def test_strips_markdown_links():
    assert strip_markdown("See [our docs](https://example.com) for more") == "See our docs (https://example.com) for more"


def test_strips_code_fences():
    assert strip_markdown("```python\nprint(1)\n```") == "print(1)"


def test_strips_blockquotes():
    assert strip_markdown("> Note: this is important") == "Note: this is important"


def test_strips_single_star_italics():
    assert strip_markdown("This is *really* important") == "This is really important"


def test_plain_text_is_unchanged():
    plain = "Go to account settings, select Change Password, enter your current password."
    assert strip_markdown(plain) == plain


def test_empty_string_is_safe():
    assert strip_markdown("") == ""


def test_idempotent_on_already_clean_text():
    text = "Go to Settings then Developer then API Keys."
    once = strip_markdown(text)
    twice = strip_markdown(once)
    assert once == twice == text


def test_idempotent_when_applied_twice_to_markdown_text():
    original = "Go to **Settings** and click *here*"
    once = strip_markdown(original)
    twice = strip_markdown(once)
    assert once == twice
