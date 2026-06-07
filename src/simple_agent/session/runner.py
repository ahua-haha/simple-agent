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

class SessionRunner:
    """Persisted runner for one Session.run invocation at a time."""

    _session_id: str
    _db: Database
    _agent_process: AgentProcess
    _cancel_event: asyncio.Event
    _next_action: RunnerAction
    _last_error: str | None
    _user_task: UserTask | None
    _lifecycles: dict[str, BaseTaskLifecycle]
    _runtime: TaskLifecycleRuntime
    _user_paused: bool

    def __init__(
        self,
        *,
        session_id: str,
        db: Database,
        agent_process: AgentProcess,
        cancel_event: asyncio.Event,
    ):
        self._session_id = session_id
        self._db = db
        self._agent_process = agent_process
        self._cancel_event = cancel_event
        self._next_action = "wait_user_input"
        self._last_error = None
        self._user_task = None
        self._lifecycles = {}
        self._runtime = TaskLifecycleRuntime(messages=[])
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
            self._lifecycles = {}
            metadata = self._db.get_runner_state_metadata(self._session_id, session=session)
            if metadata is None:
                self._next_action = "wait_user_input"
                self._last_error = None
                return
            self._next_action = metadata.next_action
            self._last_error = metadata.last_error
            # TODO: reconstruct the task tree and active lifecycle from the
            # stored runner state.

    def sync_metadata(self, *, session: Session) -> None:
        self._db.upsert_runner_state_metadata(
            self._session_id,
            next_action=self._next_action,
            active_user_task_id=self._current_active_user_task_id(),
            last_error=self._last_error,
            session=session,
        )

    async def run(self, user_input: str | None):
        self._user_paused = False
        self._cancel_event.clear()
        self.load()
        self.handle_input(user_input)

        while self._runtime.next_task is not None:
            if self._user_paused:
                break
            result = await self.run_active_lifecycle()

            if result.error is not None:
                self._raise_lifecycle_error(result.error)

            self._next_action = result.next_action
            if self._runtime.next_task is None:
                self._next_action = "wait_user_input"
            with self._db.create_session() as session:
                self.sync_metadata(session=session)
                session.commit()

        return self._current_user_task_from_database()

    def handle_input(self, user_input: str | None) -> RunnerAction:
        if user_input is None:
            return self._next_action

        self.finish_previous_user_task()
        user_task = self.start_user_task(user_input)
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
        self._user_task = None

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
        lifecycle = self._lifecycle_for_task(task)
        self._runtime.next_task_id_to_run = task.id
        self._runtime.next_task = task
        lifecycle.set_data(self._runtime)
        user_message = UserMessage(
            content=[TextContent(text=user_input)],
            timestamp=int(time.time() * 1000),
        )
        try:
            message_entry = lifecycle.append_message(user_message)
            task.start_message_id = message_entry.id
            with self._db.create_session() as session:
                lifecycle.sync_messages(self._session_id, self._db, [message_entry], session=session)
                lifecycle.sync(self._db, session)
                session.commit()
        finally:
            lifecycle.clear_data()
        return task

    def create_user_task(self, user_input: str, start_message_id: int | None = None) -> UserTask:
        if self._user_task is not None and self._user_task.status == "active":
            raise RuntimeError("Cannot create a second active user task")
        task = UserTask(id=self.allocate_task_id(), title=user_input, start_message_id=start_message_id)
        self._user_task = task
        self._runtime.next_task_id_to_run = task.id
        self._runtime.next_task = task
        return task

    async def run_active_lifecycle(self):
        task = self._resolve_next_task()
        if task is None:
            raise RuntimeError("No active task")
        lifecycle = self._lifecycle_for_task(task)
        lifecycle.set_data(self._runtime)
        try:
            result = await lifecycle.run(
                agent_process=self._agent_process,
                cancel_event=self._cancel_event,
            )
        finally:
            lifecycle.clear_data()
        self._resolve_next_task()
        return result

    def _resolve_next_task(self) -> ManagedTask | None:
        next_task_id = self._runtime.next_task_id_to_run
        if next_task_id is None:
            self._runtime.next_task = None
            return None
        task = self._runtime.next_task
        if task is None or task.id != next_task_id:
            task = self.build_tree(next_task_id)
        if task is None:
            raise RuntimeError(f"Next task {next_task_id} is missing")
        self._runtime.next_task = task
        return task

    def _lifecycle_for_task(self, task: ManagedTask) -> BaseTaskLifecycle:
        if task.kind == "user_task":
            self._user_task = cast(UserTask, task)
            lifecycle = self._lifecycles.get("user_task")
            if lifecycle is None:
                lifecycle = UserTaskLifecycle()
                self._lifecycles["user_task"] = lifecycle
            return lifecycle
        if task.kind == "todo":
            lifecycle = self._lifecycles.get("todo")
            if lifecycle is None:
                lifecycle = TodoTaskLifecycle()
                self._lifecycles["todo"] = lifecycle
            return lifecycle
        raise RuntimeError("Tool-call tasks do not have a lifecycle")

    def build_tree(self, task_id: int) -> ManagedTask | None:
        with self._db.create_session() as session:
            root = self._db.get_managed_task(task_id, session=session)
            if root is None or root.id is None:
                return None

            def attach_children(task: ManagedTask) -> None:
                task.children = []
                for child in self._db.list_managed_task_children(task.id, session=session):
                    if child.id is not None:
                        attach_children(child)
                        task.children.append(child)

            attach_children(root)
            return root

    def _load_runtime(self, *, session: Session) -> None:
        self._runtime = TaskLifecycleRuntime(
            messages=[
                MessageEntry(id=message_id, message=message)
                for message_id, message in self._db.list_runner_message_entries(self._session_id, session=session)
            ],
            next_message_id=self._db.next_runner_message_id(session=session),
            next_tool_call_log_id=self._db.next_runner_tool_call_id(self._session_id, session=session),
            next_task_id_to_allocate=self._db.next_managed_task_id(session=session),
        )

    def allocate_task_id(self) -> int:
        if self._runtime.next_task_id_to_allocate is None:
            raise RuntimeError("Task lifecycle runtime is missing allocation state")
        task_id = self._runtime.next_task_id_to_allocate
        self._runtime.next_task_id_to_allocate += 1
        return task_id

    @property
    def user_task(self) -> UserTask | None:
        return self._user_task

    def _current_active_user_task_id(self) -> int | None:
        if self._user_task is not None and self._user_task.status == "active":
            return self._user_task.id
        return None
