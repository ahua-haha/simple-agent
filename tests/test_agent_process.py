"""Tests for AgentProcess executor contract."""

from __future__ import annotations

import asyncio

import pytest

from pi.agent import AgentTool, AgentToolResult
from pi.ai.types import AssistantMessage, TextContent, ToolCall, UserMessage

from simple_agent.process.agent_process import AgentProcess


@pytest.mark.asyncio
async def test_agent_process_run_returns_messages_without_agent_state(monkeypatch):
    message = AssistantMessage(role="assistant", content=[TextContent(text="done")])

    class FakeStream:
        _background_task = None

        def __aiter__(self):
            return self

        async def __anext__(self):
            from pi.agent.types import AgentEndEvent

            if getattr(self, "_sent", False):
                raise StopAsyncIteration
            self._sent = True
            return AgentEndEvent(messages=[message])

    captured = {}

    def fake_agent_loop(input_messages, context, loop_config, cancel_event=None):
        captured["cancel_event"] = cancel_event
        captured["tools"] = context.tools
        return FakeStream()

    monkeypatch.setattr("simple_agent.process.agent_process.agent_loop", fake_agent_loop)

    cancel_event = asyncio.Event()
    process = AgentProcess(model=object())
    result = await process.run(
        system_prompt="system",
        messages=[],
        tools=[],
        user_prompt="hello",
        cancel_event=cancel_event,
    )

    assert result == [message]
    assert captured["cancel_event"] is cancel_event
    assert captured["tools"] == []


@pytest.mark.asyncio
async def test_agent_process_run_with_none_user_prompt_continues_existing_messages(monkeypatch):
    message = AssistantMessage(role="assistant", content=[TextContent(text="continued")])
    existing_message = UserMessage(content=[TextContent(text="previous")], timestamp=1)

    class FakeStream:
        _background_task = None

        def __aiter__(self):
            return self

        async def __anext__(self):
            from pi.agent.types import AgentEndEvent

            if getattr(self, "_sent", False):
                raise StopAsyncIteration
            self._sent = True
            return AgentEndEvent(messages=[message])

    captured = {}

    def fake_agent_loop_continue(context, loop_config, cancel_event=None):
        captured["cancel_event"] = cancel_event
        captured["messages"] = context.messages
        captured["tools"] = context.tools
        return FakeStream()

    def fake_agent_loop(input_messages, context, loop_config, cancel_event=None):
        raise AssertionError("run should use agent_loop_continue when user_prompt is None")

    monkeypatch.setattr("simple_agent.process.agent_process.agent_loop", fake_agent_loop)
    monkeypatch.setattr("simple_agent.process.agent_process.agent_loop_continue", fake_agent_loop_continue)

    cancel_event = asyncio.Event()
    process = AgentProcess(model=object())
    result = await process.run(
        system_prompt="system",
        messages=[existing_message],
        tools=[],
        user_prompt=None,
        cancel_event=cancel_event,
    )

    assert result == [message]
    assert captured["cancel_event"] is cancel_event
    assert captured["messages"] == [existing_message]
    assert captured["tools"] == []


@pytest.mark.asyncio
async def test_agent_process_call_llm_step_returns_one_assistant_message_and_emits_events(monkeypatch):
    from pi.agent.types import MessageEndEvent

    message = AssistantMessage(role="assistant", content=[TextContent(text="done")])
    existing_message = UserMessage(content=[TextContent(text="previous")], timestamp=1)

    captured = {}

    async def fake_stream_assistant_response(context, loop_config, cancel_event, stream, stream_fn):
        captured["context_messages"] = context.messages
        loop_config.on_event(MessageEndEvent(message=message))
        return message

    monkeypatch.setattr(
        "simple_agent.process.agent_process._stream_assistant_response",
        fake_stream_assistant_response,
    )

    process = AgentProcess(model=object())
    calls = []
    process.subscribe(lambda event: calls.append(("listener", event.type)))

    result = await process.call_llm_step(
        system_prompt="system",
        messages=[existing_message],
        tools=[],
        hooks={"message_end": [lambda event: calls.append(("hook", event.type))]},
    )

    assert result is message
    assert captured["context_messages"] == [existing_message]
    assert calls == [("hook", "message_end"), ("listener", "message_end")]


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
        hooks={"tool_execution_end": [lambda event: calls.append(("hook", event.type))]},
    )

    assert len(results) == 1
    assert results[0].tool_call_id == "call_1"
    assert results[0].tool_name == "echo"
    assert results[0].content[0].text == "hello"
    assert ("hook", "tool_execution_end") in calls
    assert ("listener", "tool_execution_end") in calls
    assert ("listener", "message_end") in calls


@pytest.mark.asyncio
async def test_agent_process_runs_hooks_before_listeners(monkeypatch):
    from pi.agent.types import AgentEndEvent

    message = AssistantMessage(role="assistant", content=[TextContent(text="done")])
    event = AgentEndEvent(messages=[message])
    calls = []

    class FakeStream:
        _background_task = None

        def __init__(self):
            self._sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._sent:
                raise StopAsyncIteration
            self._sent = True
            return event

    captured = {}

    def fake_agent_loop(input_messages, context, loop_config, cancel_event=None):
        captured["messages"] = context.messages
        loop_config.on_event(event)
        return FakeStream()

    monkeypatch.setattr("simple_agent.process.agent_process.agent_loop", fake_agent_loop)

    process = AgentProcess(model=object())

    def hook(seen_event):
        calls.append(("hook", seen_event is event))

    def wrong_type_hook(seen_event):
        calls.append(("wrong_type_hook", seen_event is event))

    def listener(seen_event):
        calls.append(("listener", seen_event is event))

    process.subscribe(listener)

    result = await process.run(
        system_prompt="system",
        messages=[],
        tools=[],
        user_prompt="hello",
        hooks={"agent_end": [hook], "message_start": [wrong_type_hook]},
    )

    assert result == [message]
    assert captured["messages"] == []
    assert calls == [("hook", True), ("listener", True)]
