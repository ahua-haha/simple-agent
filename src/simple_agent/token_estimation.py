"""Heuristic token estimation for pi-ai messages.

This is intentionally a rough preflight estimator, not a provider-specific
tokenizer. It mirrors the common chars/4 approximation used elsewhere.
"""

from __future__ import annotations

from pi.ai.types import (
    AssistantMessage,
    ImageContent,
    Message,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from simple_agent.json_utils import stable_json

IMAGE_ESTIMATED_CHARS = 4800


def estimate_text_tokens(text: str) -> int:
    return len(text) // 4


def estimate_user_message_tokens(message: UserMessage) -> int:
    if isinstance(message.content, str):
        return estimate_text_tokens(message.content)
    return _estimate_content_items_chars(message.content) // 4


def estimate_assistant_message_tokens(message: AssistantMessage) -> int:
    return _estimate_content_items_chars(message.content) // 4


def estimate_tool_result_message_tokens(message: ToolResultMessage) -> int:
    chars = len(message.tool_name)
    chars += _estimate_content_items_chars(message.content)
    return chars // 4


def estimate_message_tokens(message: Message) -> int:
    if isinstance(message, UserMessage):
        return estimate_user_message_tokens(message)
    if isinstance(message, AssistantMessage):
        return estimate_assistant_message_tokens(message)
    if isinstance(message, ToolResultMessage):
        return estimate_tool_result_message_tokens(message)
    return 0


def estimate_messages_tokens(messages: list[Message]) -> int:
    return sum(estimate_message_tokens(message) for message in messages)


def _estimate_content_items_chars(items: list[object]) -> int:
    total = 0
    for item in items:
        if isinstance(item, TextContent):
            total += len(item.text)
        elif isinstance(item, ThinkingContent):
            total += len(item.thinking)
        elif isinstance(item, ToolCall):
            total += len(item.name)
            total += len(stable_json(item.arguments))
        elif isinstance(item, ImageContent):
            total += IMAGE_ESTIMATED_CHARS
    return total
