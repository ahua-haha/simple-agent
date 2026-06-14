"""Tests for project-local message token estimation."""

from dataclasses import dataclass

from pi.ai.types import (
    AssistantMessage,
    ImageContent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

from simple_agent.token_estimation import (
    IMAGE_ESTIMATED_CHARS,
    estimate_assistant_message_tokens,
    estimate_message_tokens,
    estimate_messages_tokens,
    estimate_tool_result_message_tokens,
    estimate_user_message_tokens,
)


@dataclass
class _NestedDetails:
    count: int


@dataclass
class _ToolDetails:
    ok: bool
    nested: _NestedDetails


def test_estimate_user_message_tokens_for_text():
    message = UserMessage(content="abcdefgh", timestamp=1)

    assert estimate_user_message_tokens(message) == 2


def test_estimate_user_message_tokens_for_content_items():
    message = UserMessage(
        content=[
            TextContent(text="abcdefgh"),
            ImageContent(data="base64", mimeType="image/png"),
        ],
        timestamp=1,
    )

    assert estimate_user_message_tokens(message) == (8 + IMAGE_ESTIMATED_CHARS) // 4


def test_estimate_assistant_message_tokens_for_content_items():
    message = AssistantMessage(
        content=[
            TextContent(text="abcdefgh"),
            ThinkingContent(thinking="think"),
            ToolCall(id="call_1", name="read_file", arguments={"path": "app.py"}),
        ]
    )

    assert estimate_assistant_message_tokens(message) > 3


def test_estimate_tool_result_message_tokens_counts_content_and_metadata():
    message = ToolResultMessage(
        toolCallId="call_1",
        toolName="read_file",
        content=[TextContent(text="abcdefghijkl")],
        details={"ok": True},
        timestamp=1,
    )

    assert estimate_tool_result_message_tokens(message) > 3


def test_estimate_tool_result_message_tokens_ignores_details():
    without_details = ToolResultMessage(
        toolCallId="call_1",
        toolName="ls",
        content=[TextContent(text="files")],
        details=None,
        timestamp=1,
    )
    with_details = ToolResultMessage(
        toolCallId="call_1",
        toolName="ls",
        content=[TextContent(text="files")],
        details=_ToolDetails(ok=True, nested=_NestedDetails(count=2)),
        timestamp=1,
    )

    assert estimate_tool_result_message_tokens(with_details) == estimate_tool_result_message_tokens(without_details)


def test_estimate_message_tokens_dispatches_by_role():
    user = UserMessage(content="abcdefgh", timestamp=1)
    assistant = AssistantMessage(content=[TextContent(text="abcdefghijkl")])
    tool_result = ToolResultMessage(
        toolCallId="call_1",
        toolName="read_file",
        content=[TextContent(text="abcdefghijklmnop")],
        timestamp=1,
    )

    assert estimate_message_tokens(user) == estimate_user_message_tokens(user)
    assert estimate_message_tokens(assistant) == estimate_assistant_message_tokens(assistant)
    assert estimate_message_tokens(tool_result) == estimate_tool_result_message_tokens(tool_result)


def test_estimate_messages_tokens_sums_each_message():
    messages = [
        UserMessage(content="abcdefgh", timestamp=1),
        AssistantMessage(content=[TextContent(text="abcdefghijkl")]),
    ]

    assert estimate_messages_tokens(messages) == 5
