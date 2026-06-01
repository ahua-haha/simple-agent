"""SessionRunner owns the persisted Session.run workflow."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Callable, Literal

from simple_agent.tool.common_tools import create_all_coding_tools

if TYPE_CHECKING:
    from simple_agent.db.db import Database
    from simple_agent.process.agent_process import AgentProcess
    from simple_agent.task_manager import TaskManager
    from simple_agent.tool.execution_logger import ToolExecutionLogger
    from pi.agent.types import AgentMessage

_log = logging.getLogger(__name__)

RunnerPhase = Literal["idle", "running", "done", "error"]

SYSTEM_PROMPT = """You are a helpful coding agent.

Use create_todo before starting a coherent unit of work.
Call finish_todo when the active todo is complete.
Call error_todo if the active todo cannot be completed.
Keep responses concise and use available tools to do the work.
"""


class SessionRunner:
    """Persisted runner for one Session.run invocation at a time."""

    _session_id: str
    _db: Database
    _task_manager: TaskManager
    _execution_logger: ToolExecutionLogger
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
        execution_logger: ToolExecutionLogger,
        agent_process: AgentProcess,
        cancel_event: asyncio.Event,
    ):
        self._session_id = session_id
        self._db = db
        self._task_manager = task_manager
        self._execution_logger = execution_logger
        self._agent_process = agent_process
        self._cancel_event = cancel_event
        self._phase = "idle"
        self._active_user_task_id = None
        self._messages = []

    def subscribe(self, callback: Callable) -> None:
        self._agent_process.subscribe(callback)

    def pause(self) -> None:
        self._cancel_event.set()

    def load(self) -> None:
        metadata = self._db.get_runner_state_metadata(self._session_id)
        self._messages = self._db.list_runner_messages(self._session_id)
        if metadata is None:
            self._phase = "idle"
            self._active_user_task_id = None
            return
        self._phase = metadata.phase
        self._active_user_task_id = metadata.active_user_task_id
        self._task_manager.active_user_task_id = metadata.active_user_task_id

    def checkpoint(self, *, status: str | None = None, last_error: str | None = None) -> None:
        self._db.upsert_runner_state_metadata(
            self._session_id,
            phase=self._phase,
            status=status or self._phase,
            active_user_task_id=self._active_user_task_id,
            last_error=last_error,
        )

    def _create_tools(self):
        tools = [
            self._task_manager.create_create_todo_tool(),
            self._task_manager.create_finish_todo_tool(),
            self._task_manager.create_error_todo_tool(),
            *create_all_coding_tools("."),
        ]
        return self._execution_logger.wrap_tools(tools)

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
        self.checkpoint(status="running")

    async def handle_running(self, user_input: str) -> None:
        new_messages = await self._agent_process.run(
            system_prompt=SYSTEM_PROMPT,
            messages=list(self._messages),
            tools=self._create_tools(),
            user_prompt=user_input,
            cancel_event=self._cancel_event,
        )
        self._db.append_runner_messages(self._session_id, new_messages)
        self._messages.extend(new_messages)
        if self._task_manager.active_todo_id is None:
            self._task_manager.finish_user_task()
        self._phase = "done"
        self.checkpoint(status="done")

    def handle_error(self, exc: Exception) -> None:
        _log.exception("session runner failed: session=%s", self._session_id)
        self._phase = "error"
        self.checkpoint(status="error", last_error=str(exc))
