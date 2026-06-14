"""Tests for the Agent class."""

import pytest

from pi.agent.agent import Agent


def test_agent_creation():
    agent = Agent()
    assert agent.state.is_streaming is False
    assert agent.state.messages == []


def test_agent_set_system_prompt():
    agent = Agent()
    agent.set_system_prompt("Be helpful")
    assert agent.state.system_prompt == "Be helpful"


def test_agent_set_thinking_level():
    agent = Agent()
    agent.set_thinking_level("high")
    assert agent.state.thinking_level == "high"


def test_agent_queue_messages():
    from pi.ai.types import UserMessage

    agent = Agent()
    msg = UserMessage(content="test", timestamp=123)
    agent.steer(msg)
    assert agent.has_queued_messages()

    agent.clear_all_queues()
    assert not agent.has_queued_messages()


def test_agent_reset():
    from pi.ai.types import UserMessage

    agent = Agent()
    agent.append_message(UserMessage(content="test", timestamp=123))
    agent.steer(UserMessage(content="steer", timestamp=456))
    assert len(agent.state.messages) == 1
    assert agent.has_queued_messages()

    agent.reset()
    assert agent.state.messages == []
    assert not agent.has_queued_messages()


def test_agent_subscribe():
    agent = Agent()
    events = []
    unsub = agent.subscribe(lambda e: events.append(e))
    assert callable(unsub)

    # Unsubscribe
    unsub()
    # No error calling unsub twice
    unsub()


@pytest.mark.asyncio
async def test_agent_loop_runs_on_event_hook_synchronously():
    import asyncio

    from pi.agent.loop import agent_loop
    from pi.agent.types import AgentContext, AgentLoopConfig
    from pi.ai.events import AssistantMessageEventStream
    from pi.ai.types import AssistantMessage, DoneEvent, Model, TextContent, UserMessage

    assistant_message = AssistantMessage(content=[TextContent(text="done")])
    hook_events = []
    yielded_events = []

    def stream_fn(model, context, options):
        stream = AssistantMessageEventStream()
        stream.push(DoneEvent(reason="stop", message=assistant_message))
        return stream

    config = AgentLoopConfig(
        model=Model(
            id="fake",
            name="Fake",
            api="fake-api",
            provider="fake-provider",
            baseUrl="",
        ),
        convert_to_llm=lambda messages: messages,
        on_event=lambda event: hook_events.append(event),
    )
    messages = []
    stream = agent_loop(
        [UserMessage(content="hello", timestamp=1)],
        AgentContext(messages=messages),
        config,
        cancel_event=asyncio.Event(),
        stream_fn=stream_fn,
    )

    async for event in stream:
        yielded_events.append(event)

    assert hook_events == yielded_events
    assert messages == []
