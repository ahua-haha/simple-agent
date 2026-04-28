"""Custom model registrations for MiniMax-CN and DeepSeek providers."""

from __future__ import annotations

import os

from pi.ai.models import register_models, get_model
from pi.ai import Model



# API key environment variable mapping
PROVIDER_API_KEYS = {
    "minimax-cn": "MINIMAX_CN_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def get_api_key(provider: str) -> str | None:
    """Get API key for a provider from environment variables.

    Args:
        provider: The provider name (e.g., "minimax-cn", "deepseek")

    Returns:
        API key string if found, None otherwise
    """
    env_var = PROVIDER_API_KEYS.get(provider)
    if env_var:
        return os.environ.get(env_var)
    return None


def register_custom_models() -> None:
    """Register MiniMax-CN and DeepSeek models.

    Call this once per session before using the models.
    Safe to call multiple times - will skip if models already registered.
    """
    # MiniMax-M2.7
    if not get_model("minimax-cn", "MiniMax-M2.7"):
        register_models(
            "minimax-cn",
            {
                "MiniMax-M2.7": Model(
                    id="MiniMax-M2.7",
                    provider="minimax-cn",
                    api="anthropic-messages",
                    base_url="https://api.minimaxi.com/anthropic",
                    name="MiniMax-M2.7",
                    reasoning=True,
                    input=["text"],
                    cost=dict(input=0.3, output=1.2, cache_read=0.06, cache_write=0.375),
                    context_window=204800,
                    max_tokens=131072,
                ),
            },
        )

    # DeepSeek V4 Pro
    if not get_model("deepseek", "deepseek-v4-pro"):
        register_models(
            "deepseek",
            {
                "deepseek-v4-pro": Model(
                    id="deepseek-v4-pro",
                    provider="deepseek",
                    api="anthropic-messages",
                    base_url="https://api.deepseek.com/anthropic",
                    name="DeepSeek V4 Pro",
                    reasoning=True,
                    input=["text"],
                    cost=dict(input=1.74, output=3.48, cache_read=0.145, cache_write=0),
                    context_window=1000000,
                    max_tokens=384000,
                ),
            },
        )