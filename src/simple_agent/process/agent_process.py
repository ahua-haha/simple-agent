"""AgentProcess — pure agent executor."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable

from pi.agent import AgentTool
from pi.agent.loop import agent_loop, agent_loop_continue
from pi.agent.types import AgentContext, AgentEndEvent, AgentEvent, AgentLoopConfig, AgentMessage
from pi.ai.types import UserMessage, TextContent

from simple_agent.log import logged
from simple_agent.models import register_custom_models, get_api_key

_log = logging.getLogger(__name__)

AgentProcessHook = Callable[[AgentEvent], None]
AgentProcessHooks = dict[str, list[AgentProcessHook]]


class AgentProcess:
    """Pure agent executor.

    ``__init__`` sets the model. ``run()`` takes prompts, messages,
    caller-owned tools, and an optional cancel event, then returns the
    messages produced by the agent loop.
    """

    def __init__(self, model):
        register_custom_models()
        self._model = model
        self._api_key = get_api_key
        self._listeners: list[Callable] = []

    @logged(_log)
    async def run(
        self,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        user_prompt: str | None = "",
        cancel_event: asyncio.Event | None = None,
        hooks: AgentProcessHooks | None = None,
    ) -> list[AgentMessage]:
        """Execute a single agent run and return the new messages."""
        context = AgentContext(
            system_prompt=system_prompt,
            messages=list(messages),
            tools=tools,
        )
        new_messages: list[AgentMessage] = []

        def on_loop_event(event: AgentEvent) -> None:
            for hook in (hooks or {}).get(event.type, []):
                hook(event)

        loop_config = AgentLoopConfig(
            model=self._model,
            convert_to_llm=lambda msgs: [m for m in msgs if m.role in ("user", "assistant", "tool_result")],
            get_api_key=self._api_key,
            on_event=on_loop_event,
        )

        if user_prompt is None:
            stream = agent_loop_continue(
                context,
                loop_config,
                cancel_event=cancel_event,
            )
        else:
            now_ms = int(time.time() * 1000)
            input_messages = []
            if user_prompt:
                input_messages.append(UserMessage(content=[TextContent(text=user_prompt)], timestamp=now_ms))
            stream = agent_loop(
                input_messages,
                context,
                loop_config,
                cancel_event=cancel_event,
            )

        async for event in stream:
            if isinstance(event, AgentEndEvent):
                new_messages = event.messages
            self._emit(event)

        if stream._background_task is not None:
            exc = stream._background_task.exception()
            if exc is not None:
                raise exc

        return new_messages

    def subscribe(self, callback: Callable) -> None:
        self._listeners.append(callback)

    def unsubscribe(self, callback: Callable) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _emit(self, event) -> None:
        for listener in self._listeners:
            listener(event)
