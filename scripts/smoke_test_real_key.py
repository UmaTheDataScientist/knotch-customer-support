"""Minimal real-key smoke test.

Makes exactly TWO API calls total (one chat, one embedding) to confirm the
real OPENAI_API_KEY actually works against both permitted endpoints, without
touching anything close to the 50 req/min limit. This is deliberately NOT
the eval harness or test suite -- those make dozens of calls per run.

Usage (from the project root, with .env containing your real key):
    python scripts/smoke_test_real_key.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.core.llm_client import build_llm_client


def main() -> None:
    settings = get_settings()

    if settings.llm_provider != "openai":
        print(f"LLM_PROVIDER is currently '{settings.llm_provider}', not 'openai'.")
        print("Set LLM_PROVIDER=openai in your .env file before running this script.")
        sys.exit(1)

    if not settings.openai_api_key:
        print("OPENAI_API_KEY is empty. Add it to your .env file before running this script.")
        sys.exit(1)

    print(f"Provider: {settings.llm_provider}")
    print(f"Chat model: {settings.openai_chat_model}")
    print(f"Embed model: {settings.openai_embed_model}")
    print()

    llm = build_llm_client(settings)

    # --- Call 1 of 2: chat completion ---
    print("Call 1/2: chat completion...")
    chat_result = llm.chat(
        [
            {"role": "system", "content": "Reply with exactly the word: pong"},
            {"role": "user", "content": "ping"},
        ],
        temperature=0.0,
    )
    print(f"  response: {chat_result.content!r}")
    print(f"  tokens: {chat_result.prompt_tokens} prompt / {chat_result.completion_tokens} completion")
    print(f"  estimated cost: ${chat_result.estimated_cost_usd:.6f}")
    print()

    # --- Call 2 of 2: embedding ---
    print("Call 2/2: embedding...")
    vectors = llm.embed(["How do I reset my password?"])
    vec = vectors[0]
    print(f"  embedding dimensions: {len(vec)}")
    print(f"  first 5 values: {[round(v, 4) for v in vec[:5]]}")
    print()

    print("Both endpoints responded successfully. Key is working for chat + embeddings.")
    print("Total API calls made by this script: 2")


if __name__ == "__main__":
    main()
