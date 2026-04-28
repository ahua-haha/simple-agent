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
from pi.ai.models import register_models, get_model
from pi.ai.providers.anthropic import (
    AnthropicOptions,
    _convert_messages,
    _convert_tools,
    _get_cache_control,
    _map_stop_reason,
    _supports_adaptive_thinking,
    _map_thinking_level_to_effort,
    adjust_max_tokens_for_thinking,
    build_base_options,
    stream_anthropic,
)
from pi.ai.types import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    Model,
    ModelCost,
    SimpleStreamOptions,
    StartEvent,
    TextContent,
    ThinkingContent,
    ToolCall,
    TextStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    ThinkingStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ToolCallStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    Usage,
)
from pi.ai.utils.json import parse_streaming_json
from pi.ai.models import calculate_cost


def get_minimax_api_key(provider: str) -> str | None:
    """Custom API key getter for MiniMax-CN."""
    return os.environ.get("MINIMAX_CN_API_KEY")

def get_deepseek_api_key(provider: str) -> str | None:
    """Custom API key getter for MiniMax-CN."""
    return os.environ.get("DEEPSEEK_API_KEY")

def setup_minimax_provider():
    """Register MiniMax-CN as a provider (called once per session)."""
    if get_model("minimax-cn", "MiniMax-M2.7"):
        return  # Already registered

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
                cost=ModelCost(input=0.3, output=1.2, cache_read=0.06, cache_write=0.375),
                context_window=204800,
                max_tokens=131072,
            ),
        },
    )

def setup_deepseek_model():
    """Register MiniMax-CN as a provider (called once per session)."""
    if get_model("deepseek", "deepseek-v4-pro"):
        return  # Already registered

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
                cost=ModelCost(input=1.74, output=3.48, cache_read=0.145, cache_write=0),
                context_window= 1000000,
                max_tokens= 384000,
            ),
        },
    )

@pytest.fixture(scope="session", autouse=True)
def register_minimax():
    """Register MiniMax-CN provider once for all tests."""
    setup_minimax_provider()
    setup_deepseek_model()


def on_event(event):
    """Print events in streaming mode."""
    if event.type == "message_update":
        ae = event.assistant_message_event
        if ae.type == "text_delta":
            print(ae.delta, end="", flush=True)
        # elif ae.type == "thinking_delta":
            # print(ae.delta, end="", flush=True)
        elif ae.type == "tool_call_delta":
            print(f"\n[tool call: {ae.name} → {ae.delta}]", end="", flush=True)
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

        agent = Agent(get_api_key=get_deepseek_api_key)
        agent.set_model(model)
        agent.set_system_prompt("You are a helpful assistant. Respond in one sentence.")

        collected_events = []
        collected_events_append = collected_events.append

        def on_event_with_collect(event):
            on_event(event)
            collected_events_append(event)

        agent.subscribe(on_event_with_collect)

        print("\n--- test_agent_with_minimax_model ---")
        await agent.prompt("tell me your model version, and knowledge cut off time")
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

        agent = Agent(initial_state=state, get_api_key=get_minimax_api_key)
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
