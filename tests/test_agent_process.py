"""Tests for AgentProcess executor contract."""

from __future__ import annotations

import asyncio

import pytest

from pi.ai.types import AssistantMessage, TextContent

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
