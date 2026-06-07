"""SessionRunner owns the persisted Session.run workflow."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Callable, Literal

from pi.ai.types import AssistantMessage

if TYPE_CHECKING:
    from simple_agent.db.db import Database
    from simple_agent.process.agent_process import AgentProcess
    from simple_agent.task_manager import TaskManager
    from sqlmodel import Session

_log = logging.getLogger(__name__)

RunnerAction = Literal["normal_run", "compact", "handle_error", "wait_user_input"]

DEFAULT_CONTEXT_TOKEN_THRESHOLD = 120_000
DEFAULT_TOOL_CALL_THRESHOLD = 200

class SessionRunner:
    """Persisted runner for one Session.run invocation at a time."""

    _session_id: str
    _db: Database
    _task_manager: TaskManager
    _agent_process: AgentProcess
    _cancel_event: asyncio.Event
    _next_action: RunnerAction
    _active_user_task_id: int | None
    _last_error: str | None
    _context_token_threshold: int
    _tool_call_threshold: int
    _user_paused: bool

    def __init__(
        self,
        *,
        session_id: str,
        db: Database,
        task_manager: TaskManager,
        agent_process: AgentProcess,
        cancel_event: asyncio.Event,
        context_token_threshold: int = DEFAULT_CONTEXT_TOKEN_THRESHOLD,
        tool_call_threshold: int = DEFAULT_TOOL_CALL_THRESHOLD,
    ):
        self._session_id = session_id
        self._db = db
        self._task_manager = task_manager
        self._agent_process = agent_process
        self._cancel_event = cancel_event
        self._next_action = "wait_user_input"
        self._active_user_task_id = None
        self._last_error = None
        self._context_token_threshold = context_token_threshold
        self._tool_call_threshold = tool_call_threshold
        self._user_paused = False

    def subscribe(self, callback: Callable) -> None:
        self._agent_process.subscribe(callback)

    def unsubscribe(self, callback: Callable) -> None:
        self._agent_process.unsubscribe(callback)

    def pause(self) -> None:
        self._user_paused = True
        self._cancel_event.set()

    def load(self) -> None:
        with self._db.create_session() as session:
            metadata = self._db.get_runner_state_metadata(self._session_id, session=session)
            if metadata is None:
                self._next_action = "wait_user_input"
                self._active_user_task_id = None
                self._last_error = None
                self._task_manager.load(None, session=session)
                return
            self._next_action = metadata.next_action
            self._active_user_task_id = metadata.active_user_task_id
            self._last_error = metadata.last_error
            self._task_manager.load(metadata.active_user_task_id, session=session)

    def sync_metadata(self, *, session: Session) -> None:
        self._db.upsert_runner_state_metadata(
            self._session_id,
            next_action=self._next_action,
            active_user_task_id=self._active_user_task_id,
            last_error=self._last_error,
            session=session,
        )

    async def run(self, user_input: str | None):
        self._user_paused = False
        self._cancel_event.clear()
        self.load()
        try:
            next_action = self.handle_input(user_input)
        except Exception as exc:
            next_action = self.handle_error(exc)

        while next_action != "wait_user_input":
            if self._user_paused:
                break
            try:
                next_action = await self.route_next_action(next_action, user_input)
            except Exception as exc:
                next_action = self.handle_error(exc)

        if self._active_user_task_id is None:
            return None
        with self._db.create_session() as session:
            return self._db.get_managed_task(self._active_user_task_id, session=session)

    async def route_next_action(self, next_action: RunnerAction, user_input: str | None) -> RunnerAction:
        if next_action == "normal_run":
            return await self.handle_running(user_input)
        if next_action == "compact":
            return await self.handle_compact(user_input)
        if next_action == "handle_error":
            return self.handle_error()
        if next_action == "wait_user_input":
            return next_action
        raise RuntimeError(f"Unknown runner action: {next_action}")

    def handle_input(self, user_input: str | None) -> RunnerAction:
        if user_input is None:
            return self._next_action

        self.finish_previous_user_task()
        with self._db.create_session() as session:
            self._task_manager.load(None, session=session)
        user_task = self._task_manager.start_user_task(
            session_id=self._session_id,
            user_input=user_input,
        )
        self._active_user_task_id = user_task.id
        self._next_action = "normal_run"
        self._last_error = None
        self._cancel_event.clear()
        with self._db.create_session() as session:
            self.sync_metadata(session=session)
            session.commit()
        return self._next_action

    def finish_previous_user_task(self) -> None:
        previous_user_task = self._task_manager.user_task
        if previous_user_task is None:
            return

        if previous_user_task.status != "done":
            active_task = self._task_manager.active_lifecycle_for_tools().task
            if active_task.kind == "todo":
                self._task_manager.active_lifecycle_for_tools().error_task(error="Interrupted by new user input")
                self._task_manager.refresh_active_task()
            self._task_manager.finish_user_task()
            self._next_action = "wait_user_input"
            with self._db.create_session() as session:
                self._task_manager.save(session=session)
                self.sync_metadata(session=session)
                session.commit()
        self._active_user_task_id = None

    async def handle_running(self, user_input: str | None) -> RunnerAction:
        if self._next_action != "normal_run":
            return self._next_action

        result = await self._task_manager.run(
            session_id=self._session_id,
            agent_process=self._agent_process,
            cancel_event=self._cancel_event,
            context_token_threshold=self._context_token_threshold,
            tool_call_threshold=self._tool_call_threshold,
        )
        if result.error is not None:
            return self.exit(next_action=result.next_action, error=result.error)
        self._next_action = result.next_action
        self._active_user_task_id = self._task_manager.user_task.id if self._task_manager.user_task else None
        with self._db.create_session() as session:
            self.sync_metadata(session=session)
            session.commit()
        return self._next_action

    async def handle_compact(self, user_input: str | None) -> RunnerAction:
        result = await self._task_manager.run(
            session_id=self._session_id,
            agent_process=self._agent_process,
            cancel_event=self._cancel_event,
            context_token_threshold=self._context_token_threshold,
            tool_call_threshold=self._tool_call_threshold,
        )
        if result.error is not None:
            return self.exit(next_action=result.next_action, error=result.error)
        self._next_action = result.next_action
        self._active_user_task_id = self._task_manager.user_task.id if self._task_manager.user_task else None
        with self._db.create_session() as session:
            self.sync_metadata(session=session)
            session.commit()
        return self._next_action

    def exit(
        self,
        *,
        next_action: RunnerAction,
        error: Exception | AssistantMessage | str | None = None,
        clear_cancel: bool = False,
    ) -> RunnerAction:
        if error is not None:
            self._last_error = self._error_message(error)
        self._next_action = next_action
        if clear_cancel:
            self._cancel_event.clear()
        with self._db.create_session() as session:
            self.sync_metadata(session=session)
            session.commit()
        return self._next_action

    def handle_error(self, error: Exception | AssistantMessage | str | None = None) -> RunnerAction:
        if error is not None:
            self._last_error = self._error_message(error)
            if isinstance(error, Exception):
                _log.error(
                    "session runner failed: session=%s",
                    self._session_id,
                    exc_info=(type(error), error, error.__traceback__),
                )
            self._next_action = "handle_error"
            return self._next_action

        self._next_action = "wait_user_input"
        self._cancel_event.clear()
        with self._db.create_session() as session:
            self.sync_metadata(session=session)
            session.commit()
        return self._next_action

    def _error_message(self, error: Exception | AssistantMessage | str) -> str:
        if isinstance(error, AssistantMessage):
            return error.error_message or "assistant response stopped with error"
        return str(error)
