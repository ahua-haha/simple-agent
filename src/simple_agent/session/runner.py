"""SessionRunner owns the persisted Session.run workflow."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Callable, Literal, cast

from pi.ai.types import TextContent, UserMessage

from simple_agent.message_store import MessageEntry
from simple_agent.task_manager.lifecycle import (
    BaseTaskLifecycle,
    TaskLifecycleRuntime,
    TodoTaskLifecycle,
    UserTaskLifecycle,
)
from simple_agent.task_manager.models import ManagedTask, TodoTask, UserTask

if TYPE_CHECKING:
    from simple_agent.db.db import Database
    from simple_agent.process.agent_process import AgentProcess
    from sqlmodel import Session

RunnerAction = Literal["normal_run", "compact", "wait_user_input"]

DEFAULT_CONTEXT_TOKEN_THRESHOLD = 120_000
DEFAULT_TOOL_CALL_THRESHOLD = 200

class SessionRunner:
    """Persisted runner for one Session.run invocation at a time."""

    _session_id: str
    _db: Database
    _agent_process: AgentProcess
    _cancel_event: asyncio.Event
    _next_action: RunnerAction
    _active_user_task_id: int | None
    _last_error: str | None
    _user_task: UserTask | None
    _active_lifecycle: BaseTaskLifecycle | None
    _pending_lifecycles: list[BaseTaskLifecycle]
    _runtime: TaskLifecycleRuntime
    _next_task_id: int
    _context_token_threshold: int
    _tool_call_threshold: int
    _user_paused: bool

    def __init__(
        self,
        *,
        session_id: str,
        db: Database,
        agent_process: AgentProcess,
        cancel_event: asyncio.Event,
        context_token_threshold: int = DEFAULT_CONTEXT_TOKEN_THRESHOLD,
        tool_call_threshold: int = DEFAULT_TOOL_CALL_THRESHOLD,
    ):
        self._session_id = session_id
        self._db = db
        self._agent_process = agent_process
        self._cancel_event = cancel_event
        self._next_action = "wait_user_input"
        self._active_user_task_id = None
        self._last_error = None
        self._user_task = None
        self._active_lifecycle = None
        self._pending_lifecycles = []
        self._runtime = TaskLifecycleRuntime(messages=[])
        self._next_task_id = 1
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
            self._load_runtime(session=session)
            self._user_task = None
            self._active_lifecycle = None
            self._pending_lifecycles = []
            metadata = self._db.get_runner_state_metadata(self._session_id, session=session)
            if metadata is None:
                self._next_action = "wait_user_input"
                self._active_user_task_id = None
                self._last_error = None
                return
            self._next_action = metadata.next_action
            self._active_user_task_id = metadata.active_user_task_id
            self._last_error = metadata.last_error
            # TODO: reconstruct the task tree, pending lifecycles, and active
            # lifecycle from the stored runner state.

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
        self.handle_input(user_input)

        while self._active_lifecycle is not None:
            if self._user_paused:
                break
            result = await self.run_active_lifecycle()

            if result.error is not None:
                self._raise_lifecycle_error(result.error)

            self._next_action = result.next_action
            if self._active_lifecycle is None:
                self._next_action = "wait_user_input"
            self._active_user_task_id = (
                self._user_task.id
                if self._user_task is not None and self._user_task.status == "active"
                else None
            )
            with self._db.create_session() as session:
                self.sync_metadata(session=session)
                session.commit()

        return self._current_user_task_from_database()

    def handle_input(self, user_input: str | None) -> RunnerAction:
        if user_input is None:
            return self._next_action

        self.finish_previous_user_task()
        user_task = self.start_user_task(user_input)
        self._active_user_task_id = user_task.id
        self._next_action = "normal_run"
        self._last_error = None
        self._cancel_event.clear()
        with self._db.create_session() as session:
            self.sync_metadata(session=session)
            session.commit()
        return self._next_action

    def finish_previous_user_task(self) -> None:
        # TODO: finish or interrupt the previous user task through its lifecycle
        # before starting a new user task.
        self._active_user_task_id = None
        self._user_task = None
        self._active_lifecycle = None
        self._pending_lifecycles = []

    def _raise_lifecycle_error(self, error: Exception | object | str) -> None:
        if isinstance(error, Exception):
            raise error
        error_message = getattr(error, "error_message", None)
        if error_message:
            raise RuntimeError(error_message)
        raise RuntimeError(str(error))

    def _current_user_task_from_database(self) -> ManagedTask | None:
        if self._user_task is None:
            return None
        with self._db.create_session() as session:
            return self._db.get_managed_task(self._user_task.id, session=session)

    def start_user_task(self, user_input: str) -> UserTask:
        task = self.create_user_task(user_input)
        lifecycle = self._require_user_task_lifecycle()
        user_message = UserMessage(
            content=[TextContent(text=user_input)],
            timestamp=int(time.time() * 1000),
        )
        message_entry = lifecycle.append_message(user_message)
        task.start_message_id = message_entry.id
        with self._db.create_session() as session:
            lifecycle.sync_messages(self._session_id, self._db, [message_entry], session=session)
            lifecycle.sync(self._db, session)
            session.commit()
        return task

    def create_user_task(self, user_input: str, start_message_id: int | None = None) -> UserTask:
        if self._user_task is not None and self._user_task.status == "active":
            raise RuntimeError("Cannot create a second active user task")
        task = UserTask(id=self.allocate_task_id(), title=user_input, start_message_id=start_message_id)
        self._user_task = task
        self._active_user_task_id = task.id
        self._active_lifecycle = UserTaskLifecycle(
            task,
            allocate_task_id=self.allocate_task_id,
            runtime=self._runtime,
        )
        return task

    async def run_active_lifecycle(self):
        lifecycle = self._require_active_lifecycle()
        result = await lifecycle.run(
            agent_process=self._agent_process,
            cancel_event=self._cancel_event,
            context_token_threshold=self._context_token_threshold,
            tool_call_threshold=self._tool_call_threshold,
        )
        current_lifecycle = self._active_lifecycle
        next_task = result.next_task
        if next_task is None:
            self._active_lifecycle = None
            return result
        if current_lifecycle is not None and next_task.id == current_lifecycle.task.id:
            self._active_lifecycle = current_lifecycle
            return result
        if current_lifecycle is not None:
            self._pending_lifecycles.append(current_lifecycle)
        self._active_lifecycle = self._create_lifecycle(next_task)
        return result

    def _create_lifecycle(self, task: ManagedTask) -> BaseTaskLifecycle:
        if self._active_lifecycle is not None and task.id == self._active_lifecycle.task.id:
            return self._active_lifecycle
        for lifecycle in self._pending_lifecycles:
            if task.id == lifecycle.task.id:
                self._pending_lifecycles.remove(lifecycle)
                return lifecycle
        if task.kind == "user_task":
            return UserTaskLifecycle(
                cast(UserTask, task),
                allocate_task_id=self.allocate_task_id,
                runtime=self._runtime,
            )
        if task.kind == "todo":
            return TodoTaskLifecycle(
                cast(TodoTask, task),
                allocate_task_id=self.allocate_task_id,
                user_task=self._user_task,
                runtime=self._runtime,
            )
        raise RuntimeError("Tool-call tasks do not have a lifecycle")

    def _load_runtime(self, *, session: Session) -> None:
        self._runtime = TaskLifecycleRuntime(
            messages=[
                MessageEntry(id=message_id, message=message)
                for message_id, message in self._db.list_runner_message_entries(self._session_id, session=session)
            ],
            next_message_id=self._db.next_runner_message_id(session=session),
            next_tool_call_log_id=self._db.next_runner_tool_call_id(self._session_id, session=session),
        )
        self._next_task_id = self._db.next_managed_task_id(session=session)

    def allocate_task_id(self) -> int:
        task_id = self._next_task_id
        self._next_task_id += 1
        return task_id

    def _require_active_lifecycle(self) -> BaseTaskLifecycle:
        if self._active_lifecycle is None:
            raise RuntimeError("No active task lifecycle")
        return self._active_lifecycle

    @property
    def user_task(self) -> UserTask | None:
        return self._user_task

    def _require_user_task_lifecycle(self) -> UserTaskLifecycle:
        lifecycle = self._require_active_lifecycle()
        if not isinstance(lifecycle, UserTaskLifecycle):
            raise RuntimeError("Active lifecycle is not a user task lifecycle")
        return lifecycle
