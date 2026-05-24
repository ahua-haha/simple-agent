"""AgentProcess — pure agent executor.  Caller owns AgentState and tools."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from pi.agent import AgentTool, AgentToolResult, AgentToolUpdateCallback
from pi.agent.loop import agent_loop
from pi.agent.types import AgentContext, AgentEndEvent, AgentLoopConfig, AgentMessage, ToolExecutionEndEvent, TurnEndEvent
from pi.ai.types import UserMessage, TextContent

from simple_agent.models import register_custom_models, get_api_key

HookFn = Callable[["AgentProcess"], None]


class AgentState(asyncio.Event):
    """Per-run state owned by the caller.

    The caller creates this, binds tools to write into ``tool_results``
    and to set ``finish_reason``, configures ``stop_condition``, then
    passes it to ``AgentProcess.run()`` as the ``cancel_event``.

    While the agent runs, ``agent_loop`` checks ``is_set()`` to decide
    whether to stop.  After the run the caller reads results from the
    same object.
    """

    def __init__(self):
        super().__init__()
        self.new_messages: list[AgentMessage] = []
        self.tool_results: dict[str, list] = {}
        self.finish_reason: str | None = None
        self.tool_calls: dict[str, list[dict]] = {}
        self.turn_count: int = 0
        self.error: str | None = None
        self.stop_condition: Callable[["AgentState"], bool] | None = None

    def is_set(self) -> bool:
        if self.stop_condition is not None and self.stop_condition(self):
            return True
        if self.finish_reason is not None:
            return True
        return super().is_set()

    def bind_tool(self, tool: AgentTool, *, stop: bool = False) -> AgentTool:
        """Wrap *tool* so its result is recorded into ``tool_results``.

        After each execution, if ``tool.result`` is set (e.g. by a
        ``ToolMgr.create_record_tool`` tool), it is appended to
        ``state.tool_results[tool.name]``.

        If *stop* is True, the tool also sets ``finish_reason`` and
        triggers ``set()`` to stop the agent loop.
        """
        _state = self
        _original = tool.execute

        async def _wrapped(
            tool_call_id: str,
            params: dict[str, Any],
            cancel_event: asyncio.Event | None = None,
            on_update: AgentToolUpdateCallback | None = None,
        ) -> AgentToolResult:
            res = await _original(tool_call_id, params, cancel_event, on_update)
            if tool.result is not None:
                _state.tool_results.setdefault(tool.name, []).append(tool.result)
                tool.result = None
            if stop:
                _state.finish_reason = tool.name
                _state.set()
            return res

        tool.execute = _wrapped
        return tool


class AgentProcess:
    """Pure agent executor.

    ``__init__`` sets the model.  ``run()`` takes a caller-owned
    ``AgentState`` and pre-configured tools, executes the agent loop,
    returns the (now populated) state.  AgentProcess does **not** store
    the state — the caller owns it end-to-end.
    """

    _results: dict[str, list]

    def __init__(self, model):
        register_custom_models()
        self._model = model
        self._api_key = get_api_key
        self._tools: list[AgentTool] = []
        self._results: dict[str, list] = {}
        self._listeners: list[Callable] = []
        self.state: AgentState = AgentState()

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        state: AgentState,
        user_prompt: str = "",
    ) -> AgentState:
        """Execute a single agent run.

        The caller creates and owns *state* and *tools*:
        - Tools are pre-wrapped to write results into *state* and set
          ``finish_reason`` / ``set()`` for stop behaviour.
        - *state* is the ``cancel_event`` — ``agent_loop`` checks
          ``state.is_set()`` each iteration.

        AgentProcess does **not** store *state* internally.
        """
        now_ms = int(time.time() * 1000)
        user_msg = UserMessage(content=[TextContent(text=user_prompt)], timestamp=now_ms)

        context = AgentContext(
            system_prompt=system_prompt,
            messages=list(messages),
            tools=tools,
        )
        loop_config = AgentLoopConfig(
            model=self._model,
            convert_to_llm=lambda msgs: [m for m in msgs if m.role in ("user", "assistant", "tool_result")],
            get_api_key=self._api_key,
        )

        stream = agent_loop(
            [user_msg],
            context,
            loop_config,
            cancel_event=state,
        )

        async for event in stream:
            if isinstance(event, AgentEndEvent):
                state.new_messages = event.messages
            if isinstance(event, ToolExecutionEndEvent):
                state.tool_calls.setdefault(event.tool_name, []).append(
                    {"tool_call_id": event.tool_call_id}
                )
            if isinstance(event, TurnEndEvent):
                state.turn_count += 1
            self._emit(event)

        return state

    # ------------------------------------------------------------------
    # backward-compat entry point (used by existing process classes)
    # ------------------------------------------------------------------

    async def step(
        self,
        system_prompt: str,
        messages: list[AgentMessage],
        user_prompt: str,
        stop_condition: Callable[[AgentState], bool] | None = None,
    ) -> tuple[list[AgentMessage], str | None, dict[str, list]]:
        """Backward-compatible wrapper.

        Existing process classes call this with tools pre-registered via
        ``add_tool()``.  Creates its own AgentState internally.
        """
        self._results.clear()

        state = AgentState()
        if stop_condition is not None:
            state.stop_condition = stop_condition
        self.state = state

        now_ms = int(time.time() * 1000)
        user_msg = UserMessage(content=[TextContent(text=user_prompt)], timestamp=now_ms)

        context = AgentContext(
            system_prompt=system_prompt,
            messages=list(messages),
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
            cancel_event=state,
        )

        appended: list[AgentMessage] = []

        async for event in stream:
            if isinstance(event, AgentEndEvent):
                appended = event.messages
            if isinstance(event, ToolExecutionEndEvent):
                state.tool_calls.setdefault(event.tool_name, []).append(
                    {"tool_call_id": event.tool_call_id, "args": event.args}
                )
            if isinstance(event, TurnEndEvent):
                state.turn_count += 1
            self._emit(event)

        state.new_messages = appended
        self.state = AgentState()  # clear for next run
        return appended, state.finish_reason, self._results

    # ------------------------------------------------------------------
    # tools & listeners
    # ------------------------------------------------------------------

    def subscribe(self, callback: Callable) -> None:
        self._listeners.append(callback)

    def _emit(self, event) -> None:
        for listener in self._listeners:
            listener(event)

    def stop_agent(self, reason: str):
        self.state.finish_reason = reason
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
        self.state = AgentState()
        self._results.clear()

    @staticmethod
    def prune_messages(messages: list[AgentMessage], tool_name: str) -> list[AgentMessage]:
        """Remove the last tool call pair if it matches tool_name. Returns a new list."""
        if len(messages) < 2:
            return list(messages)
        last_two = messages[-2:]
        if (
            hasattr(last_two[0], "content")
            and hasattr(last_two[1], "tool_name")
            and last_two[1].tool_name == tool_name
        ):
            from pi.ai.types import AssistantMessage
            if isinstance(last_two[0], AssistantMessage):
                return messages[:-2]
        return list(messages)
