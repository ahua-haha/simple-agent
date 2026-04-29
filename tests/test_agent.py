"""Tests for Agent with custom providers (e.g., MiniMax-CN)."""

from __future__ import annotations

import asyncio
import os
import time

import anthropic
import pytest

from pi.agent import Agent, AgentState
from pi.ai.env import get_env_api_key
from pi.ai.events import AssistantMessageEventStream
from pi.ai.models import get_model

from simple_agent.models import get_api_key, register_custom_models


@pytest.fixture(scope="session", autouse=True)
def register_models_fixture():
    """Register custom models once for all tests."""
    register_custom_models()


def on_event(event):
    """Print events in streaming mode."""
    if event.type == "message_update":
        ae = event.assistant_message_event
        if ae.type == "thinking_start":
            print("<thinking>", end="\n", flush=True)
        if ae.type == "text_start":
            print("<resp>", end="\n", flush=True)
        if ae.type == "thinking_end":
            print("\n</thinking>", end="\n", flush=True)
        if ae.type == "text_end":
            print("\n</resp>", end="\n", flush=True)
        if ae.type == "text_delta":
            print(ae.delta, end="", flush=True)
        elif ae.type == "thinking_delta":
            print(ae.delta, end="", flush=True)
    elif event.type == "tool_execution_start":
        print(f"\n[tool start: {event.tool_name}]", flush=True)
    elif event.type == "tool_execution_end":
        print(f"\n[tool end: {event.tool_name} → result={event.result}]", flush=True)
    elif event.type == "agent_end":
        print("\n[agent done]", flush=True)


class TestMinimaxProvider:
    """Tests for MiniMax-CN as a custom provider."""

    def test_minimax_model_exists(self):
        """MiniMax-M2.7 model should be registered and retrievable."""
        model = get_model("minimax-cn", "MiniMax-M2.7")
        assert model is not None
        assert model.id == "MiniMax-M2.7"
        assert model.provider == "minimax-cn"
        assert model.api == "anthropic-messages"

    def test_minimax_model_has_correct_base_url(self):
        """MiniMax model should have the correct base_url."""
        model = get_model("minimax-cn", "MiniMax-M2.7")
        assert model is not None
        assert model.base_url == "https://api.minimaxi.com/anthropic"

    def test_minimax_model_has_reasoning(self):
        """MiniMax-M2.7 should have reasoning enabled."""
        model = get_model("minimax-cn", "MiniMax-M2.7")
        assert model is not None
        assert model.reasoning is True

    def test_minimax_model_cost(self):
        """MiniMax-M2.7 should have correct cost values."""
        model = get_model("minimax-cn", "MiniMax-M2.7")
        assert model is not None
        assert model.cost.input == 0.3
        assert model.cost.output == 1.2
        assert model.cost.cache_read == 0.06
        assert model.cost.cache_write == 0.375

    @pytest.mark.asyncio
    async def test_agent_with_minimax_model(self):
        """Agent should work with MiniMax-M2.7 model."""
        if not os.environ.get("DEEPSEEK_API_KEY"):
            pytest.skip("MINIMAX_CN_API_KEY not set")

        print("hello world")
        model = get_model("deepseek", "deepseek-v4-pro")
        assert model is not None

        agent = Agent(get_api_key=get_api_key)
        agent.set_model(model)
        agent.set_system_prompt("You are a helpful assistant. Respond in one sentence.")

        collected_events = []
        collected_events_append = collected_events.append

        def on_event_with_collect(event):
            on_event(event)
            collected_events_append(event)

        agent.subscribe(on_event_with_collect)

        print("\n--- test_agent_with_minimax_model ---")
        await agent.prompt("who is the president of American now")
        print("\n--- done ---")

        # Check that we got some text output
        text_events = [
            e for e in collected_events
            if hasattr(e, "assistant_message_event") and hasattr(e.assistant_message_event, "type")
        ]
        assert len(text_events) > 0

    @pytest.mark.asyncio
    async def test_agent_state_persists_after_minimax_call(self):
        """AgentState should be updated after agent runs."""
        if not os.environ.get("MINIMAX_CN_API_KEY"):
            pytest.skip("MINIMAX_CN_API_KEY not set")

        model = get_model("minimax-cn", "MiniMax-M2.7")
        state = AgentState()
        initial_msg_count = len(state.messages)

        agent = Agent(initial_state=state, get_api_key=get_api_key)
        agent.set_model(model)
        agent.set_system_prompt("You are a helpful assistant.")

        agent.subscribe(on_event)

        print("\n--- test_agent_state_persists ---")
        await agent.prompt("What is 1+1? Answer in one word.")
        print("\n--- done ---")

        # State should have more messages
        assert len(state.messages) > initial_msg_count
        # The last message should be from the assistant
        assert state.messages[-1].role == "assistant"
