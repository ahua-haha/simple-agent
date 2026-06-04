"""AgentProcess — pure agent executor."""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

from pi.agent import AgentTool
from pi.agent.loop import (
    _create_agent_stream,
    _execute_tool_calls,
    _stream_assistant_response,
)
from pi.agent.types import AgentContext, AgentLoopConfig, AgentMessage
from pi.ai.types import AssistantMessage, ToolResultMessage

from simple_agent.models import register_custom_models, get_api_key

_log = logging.getLogger(__name__)


class AgentProcess:
    """Pure agent executor.

    ``__init__`` sets the model. Step methods take prompts, messages,
    caller-owned tools, and an optional cancel event. The caller owns the
    loop and state transitions.
    """

    def __init__(self, model):
        register_custom_models()
        self._model = model
        self._api_key = get_api_key
        self._listeners: list[Callable] = []

    def _create_loop_config(self) -> AgentLoopConfig:
        return AgentLoopConfig(
            model=self._model,
            convert_to_llm=lambda msgs: [m for m in msgs if m.role in ("user", "assistant", "tool_result")],
            get_api_key=self._api_key,
        )

    async def _consume_stream_events(self, stream) -> None:
        async for event in stream:
            self._emit(event)

    async def call_llm_step(
        self,
        *,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        cancel_event: asyncio.Event | None = None,
    ) -> AssistantMessage:
        """Call the LLM once and return the assistant message without running tools."""
        context = AgentContext(
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
        )
        stream = _create_agent_stream()
        consumer = asyncio.create_task(self._consume_stream_events(stream))
        try:
            return await _stream_assistant_response(
                context,
                self._create_loop_config(),
                cancel_event,
                stream,
                stream_fn=None,
            )
        finally:
            stream.end()
            await consumer

    async def run_tool_calls_step(
        self,
        *,
        tools: list[AgentTool],
        assistant_message: AssistantMessage,
        cancel_event: asyncio.Event | None = None,
    ) -> list[ToolResultMessage]:
        """Execute tool calls from one assistant message and return tool results."""
        stream = _create_agent_stream()
        consumer = asyncio.create_task(self._consume_stream_events(stream))
        try:
            execution = await _execute_tool_calls(
                tools,
                assistant_message,
                cancel_event,
                stream,
                self._create_loop_config(),
                get_steering_messages=None,
            )
            return execution["tool_results"]
        finally:
            stream.end()
            await consumer

    def subscribe(self, callback: Callable) -> None:
        self._listeners.append(callback)

    def unsubscribe(self, callback: Callable) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _emit(self, event) -> None:
        for listener in self._listeners:
            listener(event)
