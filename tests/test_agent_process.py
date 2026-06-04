"""Tests for AgentProcess executor contract."""

from __future__ import annotations

import asyncio

import pytest

from pi.agent import AgentTool, AgentToolResult
from pi.ai.types import AssistantMessage, TextContent, ToolCall, UserMessage

from simple_agent.process.agent_process import AgentProcess


@pytest.mark.asyncio
async def test_agent_process_call_llm_step_returns_one_assistant_message_and_emits_events(monkeypatch):
    from pi.agent.types import MessageEndEvent

    message = AssistantMessage(role="assistant", content=[TextContent(text="done")])
    existing_message = UserMessage(content=[TextContent(text="previous")], timestamp=1)

    captured = {}

    async def fake_stream_assistant_response(context, loop_config, cancel_event, stream, stream_fn):
        captured["context_messages"] = context.messages
        captured["on_event"] = loop_config.on_event
        stream.push(MessageEndEvent(message=message))
        return message

    monkeypatch.setattr(
        "simple_agent.process.agent_process._stream_assistant_response",
        fake_stream_assistant_response,
    )

    process = AgentProcess(model=object())
    calls = []
    process.subscribe(lambda event: calls.append(("listener", event.type)))

    messages = [existing_message]

    result = await process.call_llm_step(
        system_prompt="system",
        messages=messages,
        tools=[],
    )

    assert result is message
    assert captured["context_messages"] is messages
    assert captured["on_event"] is None
    assert calls == [("listener", "message_end")]


@pytest.mark.asyncio
async def test_agent_process_run_tool_calls_step_returns_tool_results_and_emits_events():
    tool = AgentTool(
        name="echo",
        description="Echo text",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

    async def execute(tool_call_id, params, cancel_event=None, on_update=None):
        return AgentToolResult(content=[TextContent(text=params["text"])])

    tool.execute = execute
    assistant = AssistantMessage(
        role="assistant",
        content=[
            ToolCall(
                id="call_1",
                name="echo",
                arguments={"text": "hello"},
            )
        ],
    )
    process = AgentProcess(model=object())
    calls = []
    process.subscribe(lambda event: calls.append(("listener", event.type)))

    results = await process.run_tool_calls_step(
        tools=[tool],
        assistant_message=assistant,
    )

    assert len(results) == 1
    assert results[0].tool_call_id == "call_1"
    assert results[0].tool_name == "echo"
    assert results[0].content[0].text == "hello"
    assert ("listener", "tool_execution_end") in calls
    assert ("listener", "message_end") in calls


def test_agent_process_does_not_expose_run_method():
    assert not hasattr(AgentProcess(model=object()), "run")
