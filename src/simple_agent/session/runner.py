"""SessionRunner owns the persisted Session.run workflow."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Literal

from pi.agent import AgentTool, AgentToolResult, AgentToolUpdateCallback
from simple_agent.tool.common_tools import create_all_coding_tools

if TYPE_CHECKING:
    from simple_agent.db.db import Database
    from simple_agent.process.agent_process import AgentProcess
    from simple_agent.task_manager import TaskManager
    from pi.agent import AgentEvent
    from pi.agent.types import AgentMessage

_log = logging.getLogger(__name__)

RunnerPhase = Literal["idle", "running", "done", "error"]

SYSTEM_PROMPT = """You are a helpful coding agent.

Use create_todo before starting a coherent unit of work.
Call finish_todo when the active todo is complete.
Call error_todo if the active todo cannot be completed.
Keep responses concise and use available tools to do the work.
"""


def _tool_result_payload(result: AgentToolResult) -> dict[str, Any]:
    return {
        "content": [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item.__dict__)
            for item in result.content
        ],
        "details": result.details,
    }


class SessionRunner:
    """Persisted runner for one Session.run invocation at a time."""

    _session_id: str
    _db: Database
    _task_manager: TaskManager
    _agent_process: AgentProcess
    _cancel_event: asyncio.Event
    _phase: RunnerPhase
    _active_user_task_id: int | None
    _messages: list[AgentMessage]

    def __init__(
        self,
        *,
        session_id: str,
        db: Database,
        task_manager: TaskManager,
        agent_process: AgentProcess,
        cancel_event: asyncio.Event,
    ):
        self._session_id = session_id
        self._db = db
        self._task_manager = task_manager
        self._agent_process = agent_process
        self._cancel_event = cancel_event
        self._phase = "idle"
        self._active_user_task_id = None
        self._messages = []

    def subscribe(self, callback: Callable) -> None:
        self._agent_process.subscribe(callback)

    def unsubscribe(self, callback: Callable) -> None:
        self._agent_process.unsubscribe(callback)

    def pause(self) -> None:
        self._cancel_event.set()

    def load(self) -> None:
        metadata = self._db.get_runner_state_metadata(self._session_id)
        self._messages = self._db.list_runner_messages(self._session_id)
        if metadata is None:
            self._phase = "idle"
            self._active_user_task_id = None
            self._task_manager.load(None)
            return
        self._phase = metadata.phase
        self._active_user_task_id = metadata.active_user_task_id
        self._task_manager.load(metadata.active_user_task_id)

    def save_metadata(self, *, status: str | None = None, last_error: str | None = None, session=None) -> None:
        self._db.upsert_runner_state_metadata(
            self._session_id,
            phase=self._phase,
            status=status or self._phase,
            active_user_task_id=self._active_user_task_id,
            last_error=last_error,
            session=session,
        )

    def save_current_data(
        self,
        messages: list[AgentMessage] | None = None,
        *,
        status: str | None = None,
        last_error: str | None = None,
        save_tasks: bool = True,
    ) -> None:
        messages = messages or []
        with self._db.create_session() as session:
            if messages:
                self._db.append_runner_messages(self._session_id, messages, session=session)
            if save_tasks:
                self._task_manager.save(session=session)
            self.save_metadata(status=status or self._phase, last_error=last_error, session=session)
            session.commit()
        self._messages.extend(messages)

    def _create_tools(self):
        tools = [
            self._task_manager.create_create_todo_tool(),
            self._task_manager.create_finish_todo_tool(),
            self._task_manager.create_error_todo_tool(),
            *create_all_coding_tools("."),
        ]
        return self.wrap_tools(tools)

    def wrap_tool(self, tool: AgentTool) -> AgentTool:
        original = tool.execute

        async def execute(
            tool_call_id: str,
            params: dict[str, Any],
            cancel_event: asyncio.Event | None = None,
            on_update: AgentToolUpdateCallback | None = None,
        ) -> AgentToolResult:
            started_at = time.time()
            try:
                result = await original(tool_call_id, params, cancel_event, on_update)
            except Exception as exc:
                self._db.insert_runner_tool_call(
                    session_id=self._session_id,
                    tool_call_id=tool_call_id,
                    tool_name=tool.name,
                    params=params,
                    result=None,
                    status="error",
                    started_at=started_at,
                    finished_at=time.time(),
                    error=str(exc),
                )
                raise

            log_id = self._db.insert_runner_tool_call(
                session_id=self._session_id,
                tool_call_id=tool_call_id,
                tool_name=tool.name,
                params=params,
                result=_tool_result_payload(result),
                status="success",
                started_at=started_at,
                finished_at=time.time(),
                error=None,
            )
            self._task_manager.record_tool_call(log_id)
            return result

        tool.execute = execute
        return tool

    def wrap_tools(self, tools: list[AgentTool]) -> list[AgentTool]:
        return [self.wrap_tool(tool) for tool in tools]

    async def run(self, user_input: str):
        self.load()
        try:
            while self._phase != "done":
                if self._phase in ("idle", "done", "error"):
                    await self.handle_idle(user_input)
                    continue
                if self._phase == "running":
                    await self.handle_running(user_input)
                    continue
                raise RuntimeError(f"Unknown runner phase: {self._phase}")
        except Exception as exc:
            self.handle_error(exc)
            raise

        if self._active_user_task_id is None:
            return None
        return self._db.get_managed_task(self._active_user_task_id)

    async def handle_idle(self, user_input: str) -> None:
        user_task = self._task_manager.create_user_task(user_input)
        self._active_user_task_id = user_task.id
        self._phase = "running"
        self.save_current_data(status="running")

    async def handle_running(self, user_input: str) -> None:
        def save_current_data(event: AgentEvent) -> None:
            messages = [event.message, *event.tool_results]
            self.save_current_data(messages, status=self._phase)

        hooks = {
            "turn_end": [save_current_data],
        }

        await self._agent_process.run(
            system_prompt=SYSTEM_PROMPT,
            messages=list(self._messages),
            tools=self._create_tools(),
            user_prompt=user_input,
            cancel_event=self._cancel_event,
            hooks=hooks,
        )
        if self._task_manager.active_todo_id is None:
            self._task_manager.finish_user_task()
        self._phase = "done"
        self.save_current_data(status="done")

    def handle_error(self, exc: Exception) -> None:
        _log.exception("session runner failed: session=%s", self._session_id)
        self._phase = "error"
        self.save_current_data(status="error", last_error=str(exc), save_tasks=False)
