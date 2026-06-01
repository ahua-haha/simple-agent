"""AgentProcess — pure agent executor.  Caller owns AgentState and tools."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, TYPE_CHECKING

from pi.agent import AgentTool, AgentToolResult, AgentToolUpdateCallback
from pi.agent.loop import agent_loop
from pi.agent.types import AgentContext, AgentEndEvent, AgentLoopConfig, AgentMessage, ToolExecutionEndEvent, TurnEndEvent
from pi.ai.types import UserMessage, TextContent

from simple_agent.log import logged
from simple_agent.models import register_custom_models, get_api_key

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from simple_agent.tool.execution_logger import ToolExecutionLogger


class AgentState(asyncio.Event):
    """Per-run state owned by the caller.

    The caller creates this, binds tools via ``bind_tool`` to record
    results into ``tool_results``, configures ``stop_condition``, then
    passes it to ``AgentProcess.run()`` as the ``cancel_event``.

    Stop is controlled exclusively through ``stop_condition`` (state-driven)
    and explicit ``set()`` (for pause/cancel).  ``finish_reason`` is a
    purely informational field — the framework never reads it.

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
        return super().is_set()

    def bind_tool(self, tool: AgentTool) -> AgentTool:
        """Wrap *tool* so its result is recorded into ``tool_results``.

        After each execution, if ``tool.result`` is set (e.g. by a
        record tool created by ``AgentState.create_record_tool``), it is appended to
        ``state.tool_results[tool.name]``.

        Does NOT set ``finish_reason`` or call ``set()`` — stop behavior
        is controlled exclusively through ``stop_condition``.
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
            return res

        tool.execute = _wrapped
        return tool

    def create_record_tool(
        self,
        model_class: type,
        name: str,
        description: str,
        parameters: dict[str, Any],
        execution_logger: "ToolExecutionLogger | None" = None,
    ) -> AgentTool:
        tool = AgentTool(name=name, description=description, parameters=parameters)

        async def execute(
            tool_call_id: str,
            params: dict[str, Any],
            cancel_event: asyncio.Event | None = None,
            on_update: AgentToolUpdateCallback | None = None,
        ) -> AgentToolResult:
            try:
                item = model_class.model_validate(params)
                tool.result = item
                return AgentToolResult(content=[TextContent(text="ok")])
            except Exception as exc:
                return AgentToolResult(content=[TextContent(text=f"validation failed: {exc}")])

        tool.execute = execute
        tool.result = None
        wrapped = self.bind_tool(tool)
        if execution_logger is not None:
            wrapped = execution_logger.wrap_tool(wrapped)
        return wrapped

    def create_determine_state_tool(self, execution_logger: "ToolExecutionLogger | None" = None) -> AgentTool:
        from simple_agent.state.state import StateClarification

        return self.create_record_tool(
            model_class=StateClarification,
            name="determine_state",
            description="Determine the current state based on context.",
            parameters={
                "type": "object",
                "properties": {
                    "state": {"type": "string", "enum": ["finished", "error"]},
                    "reason": {"type": "string", "description": "Reason for choosing this state"},
                },
                "required": ["state", "reason"],
            },
            execution_logger=execution_logger,
        )

    def create_define_task_tool(self, execution_logger: "ToolExecutionLogger | None" = None) -> AgentTool:
        from simple_agent.state.state import Task

        return self.create_record_tool(
            model_class=Task,
            name="define_task",
            description="Define a sub-task to be executed. Include all necessary context.",
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "The full input for this sub-task"},
                },
                "required": ["input"],
            },
            execution_logger=execution_logger,
        )

    def create_record_textresult_tool(self, execution_logger: "ToolExecutionLogger | None" = None) -> AgentTool:
        from simple_agent.state.state import TEXT_RESULT_JSON_SCHEMA, TextResult

        return self.create_record_tool(
            model_class=TextResult,
            name="record_textresult",
            description="Record a TextResult instance capturing a final outcome.",
            parameters=TEXT_RESULT_JSON_SCHEMA,
            execution_logger=execution_logger,
        )


class AgentProcess:
    """Pure agent executor.

    ``__init__`` sets the model.  ``run()`` takes a caller-owned
    ``AgentState`` and pre-configured tools, executes the agent loop,
    returns the (now populated) state.  AgentProcess does **not** store
    the state — the caller owns it end-to-end.
    """

    def __init__(self, model):
        register_custom_models()
        self._model = model
        self._api_key = get_api_key
        self._listeners: list[Callable] = []
        self.state: AgentState = AgentState()

    @logged(_log)
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

        if stream._background_task is not None:
            exc = stream._background_task.exception()
            if exc is not None:
                raise exc

        return state

    def subscribe(self, callback: Callable) -> None:
        self._listeners.append(callback)

    def _emit(self, event) -> None:
        for listener in self._listeners:
            listener(event)
