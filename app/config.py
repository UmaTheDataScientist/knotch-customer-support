"""Centralized application configuration.

All environment-driven knobs live here so the rest of the codebase never
reaches for os.environ directly. This is what makes provider-swapping
("OpenAI for Anthropic") a one-line change instead of a grep-and-replace.
"""
from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM provider abstraction ---
    llm_provider: str = "fake"  # openai | anthropic | fake
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    # --- Agent loop bounds ---
    max_agent_iterations: int = 6
    max_verification_retries: int = 2

    # --- Conversation state / context window management ---
    max_turns_in_context: int = 8  # older turns get summarized, not dropped silently

    # --- Retrieval ---
    faq_top_k: int = 3
    faq_min_score: float = 0.15

    # --- Observability ---
    trace_log_dir: str = "traces"


@lru_cache
def get_settings() -> Settings:
    return Settings()
