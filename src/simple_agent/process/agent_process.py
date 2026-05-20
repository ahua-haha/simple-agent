"""AgentProcess — composable agent execution with owned state and injectable stop logic."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from pi.agent import AgentTool, AgentToolResult, AgentToolUpdateCallback
from pi.agent.loop import agent_loop
from pi.agent.types import AgentContext, AgentEndEvent, AgentLoopConfig, AgentMessage, ToolExecutionEndEvent, TurnEndEvent
from pi.ai.types import UserMessage, TextContent

from simple_agent.models import register_custom_models, get_api_key
from simple_agent.state.agent_run_state import AgentRunState

HookFn = Callable[["AgentProcess"], None]


class _AgentCompat:
    """Backward-compat shim so subclasses can call proc.agent.subscribe() / proc.agent.reset()."""

    def __init__(self, process: "AgentProcess"):
        self._process = process
        self.state = None

    def subscribe(self, callback: Callable) -> None:
        self._process.subscribe(callback)

    def reset(self) -> None:
        self._process.reset()

    def set_model(self, _model) -> None:
        pass  # handled by AgentProcess.__init__


class AgentProcess:
    """Owns messages, state, and tools. Reuses agent_loop as a stateless engine."""

    finish_reason: str | None
    _results: dict[str, list]

    def __init__(self, model):
        register_custom_models()
        self._model = model
        self._api_key = get_api_key
        self._tools: list[AgentTool] = []
        self._results: dict[str, list] = {}
        self._listeners: list[Callable] = []
        self.message: list[AgentMessage] = []
        self.state = AgentRunState()
        self.finish_reason = None

        self.agent = _AgentCompat(self)

    def subscribe(self, callback: Callable) -> None:
        self._listeners.append(callback)

    def _emit(self, event) -> None:
        for listener in self._listeners:
            listener(event)

    def stop_agent(self, reason: str):
        self.state.finish_reason = reason
        self.finish_reason = reason
        self.state.set()

    def add_tool(self, tool: AgentTool | list[AgentTool], on_call: HookFn | None = None, store: bool = False) -> "AgentProcess":
        if isinstance(tool, AgentTool):
            tool = [tool]
        for t in tool:
            if on_call or store:
                original = t.execute
                async def wrapped(
                    tool_call_id: str,
                    params: dict[str, Any],
                    cancel_event: asyncio.Event | None = None,
                    on_update: AgentToolUpdateCallback | None = None,
                ) -> AgentToolResult:
                    res = await original(tool_call_id, params, cancel_event, on_update)
                    if store and t.result is not None:
                        self._results.setdefault(t.name, []).append(t.result)
                        t.result = None
                    if on_call:
                        on_call(self)
                    return res
                t.execute = wrapped
            self._tools.append(t)
        return self

    def reset(self):
        self.message = []
        self.state = AgentRunState()
        self._results.clear()
        self.finish_reason = None

    async def step(
        self,
        system_prompt: str,
        messages: list[AgentMessage],
        user_prompt: str,
        stop_condition: Callable[[AgentRunState], bool] | None = None,
    ) -> "AgentProcess":
        """Run the agent. Returns self for chaining.

        If *stop_condition* is provided, it replaces state.stop_condition for
        this run. Otherwise the default (checks finish_reason) is used.
        """
        self.reset()

        if stop_condition is not None:
            self.state.stop_condition = stop_condition

        now_ms = int(time.time() * 1000)
        user_msg = UserMessage(content=[TextContent(text=user_prompt)], timestamp=now_ms)
        self.message = list(messages)

        context = AgentContext(
            system_prompt=system_prompt,
            messages=list(self.message),
            tools=self._tools,
        )
        config = AgentLoopConfig(
            model=self._model,
            convert_to_llm=lambda msgs: [m for m in msgs if m.role in ("user", "assistant", "tool_result")],
            get_api_key=self._api_key,
        )

        stream = agent_loop(
            [user_msg],
            context,
            config,
            cancel_event=self.state,
        )

        async for event in stream:
            if isinstance(event, AgentEndEvent):
                self.message.extend(event.messages)
            if isinstance(event, ToolExecutionEndEvent):
                self.state.tool_calls.setdefault(event.tool_name, []).append(
                    {"tool_call_id": event.tool_call_id, "args": event.args}
                )
            if isinstance(event, TurnEndEvent):
                self.state.turn_count += 1
            self._emit(event)

        self.finish_reason = self.state.finish_reason
        return self

    def prune(self, tool_name: str) -> "AgentProcess":
        """Remove the last tool call pair if it matches tool_name. Returns self for chaining."""
        last_two = self.message[-2:]
        if (
            len(last_two) >= 2
            and hasattr(last_two[0], "content")
            and hasattr(last_two[1], "tool_name")
            and last_two[1].tool_name == tool_name
        ):
            from pi.ai.types import AssistantMessage
            if isinstance(last_two[0], AssistantMessage):
                del self.message[-2:]
        return self

    def result(self) -> tuple:
        """Return the recorded result: (messages, finish_reason, stored_results)."""
        return (self.message, self.finish_reason, self._results)
