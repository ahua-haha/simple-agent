"""Stateful task manager for agent-defined todos."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from pi.ai.types import AssistantMessage, TextContent, UserMessage

from simple_agent.message_store import MessageEntry
from simple_agent.task_manager.lifecycle import (
    BaseTaskLifecycle,
    TaskLifecycleRuntime,
    TodoTaskLifecycle,
    UserTaskLifecycle,
)
from simple_agent.task_manager.models import ManagedTask, TodoTask, UserTask

if TYPE_CHECKING:
    import asyncio

    from simple_agent.db.db import Database
    from simple_agent.process.agent_process import AgentProcess
    from sqlmodel import Session

TaskManagerAction = Literal["normal_run", "compact", "handle_error", "wait_user_input"]


@dataclass(frozen=True)
class TaskManagerRunResult:
    next_action: TaskManagerAction
    error: Exception | AssistantMessage | str | None = None


class TaskManagerError(RuntimeError):
    """Raised when task-manager lifecycle rules are violated."""


class TaskManager:
    """Manage one user task and one active todo at a time."""

    _db: Database
    active_task_id: int | None
    _user_task: UserTask | None
    _active_lifecycle: UserTaskLifecycle | TodoTaskLifecycle | None
    _runtime: TaskLifecycleRuntime

    def __init__(self, db: Database):
        self._db = db
        self.active_task_id: int | None = None
        self._user_task: UserTask | None = None
        self._active_lifecycle: UserTaskLifecycle | TodoTaskLifecycle | None = None
        self._runtime = TaskLifecycleRuntime(messages=[])

    def load(self, user_task_id: int | None, *, session: Session) -> None:
        self._user_task = None
        self._active_lifecycle = None
        self.active_task_id = None
        self._runtime = TaskLifecycleRuntime(
            messages=[],
            next_task_id_to_allocate=self._db.next_managed_task_id(session=session),
        )
        if user_task_id is None:
            return
        self._user_task = self.build_task_tree(user_task_id, session=session)
        if self._user_task is None:
            raise TaskManagerError("Active user task is missing")
        if self._user_task.kind != "user_task":
            raise TaskManagerError("Active task is not a user task")
        self._user_task = cast(UserTask, self._user_task)
        self.active_task_id = self._user_task.id
        self._active_lifecycle = self._create_user_task_lifecycle(self._user_task)
        self._set_active_task(self._loaded_active_task())

    def save(self, *, session: Session) -> None:
        if self._user_task is not None:
            self._lifecycle_for_task(self._user_task).sync(self._db, session)
            for task in self._user_task.children:
                if task.kind == "todo":
                    self._lifecycle_for_task(cast(TodoTask, task)).sync(self._db, session)

    # ------------------------------------------------------------------
    # Normal running phase
    # ------------------------------------------------------------------

    def create_user_task(self, input: str, start_message_id: int | None = None) -> UserTask:
        if self._user_task is not None and self._user_task.status == "active":
            raise TaskManagerError("Cannot create a second active user task")
        task = UserTask(title=input, start_message_id=start_message_id)
        task.id = self._allocate_task_id()
        self._user_task = task
        self.active_task_id = task.id
        self._active_lifecycle = self._create_user_task_lifecycle(task)
        return task

    def start_user_task(self, *, session_id: str, user_input: str) -> UserTask:
        user_task = self.create_user_task(user_input)
        lifecycle = self._require_user_task_lifecycle()
        self._load_lifecycle_runtime(lifecycle, session_id)
        user_message = UserMessage(
            content=[TextContent(text=user_input)],
            timestamp=int(time.time() * 1000),
        )
        message_entry = lifecycle.append_message(user_message)
        user_task.start_message_id = message_entry.id
        with self._db.create_session() as session:
            lifecycle.sync_messages(session_id, self._db, [message_entry], session=session)
            self.save(session=session)
            session.commit()
        return user_task

    @property
    def user_task(self) -> UserTask | None:
        return self._user_task

    def active_lifecycle_for_tools(self) -> BaseTaskLifecycle:
        if self._active_lifecycle is None:
            raise TaskManagerError("No active task lifecycle")
        return self._active_lifecycle

    def allocate_task_id(self) -> int:
        return self._allocate_task_id()

    async def run(
        self,
        *,
        session_id: str,
        agent_process: AgentProcess,
        cancel_event: asyncio.Event,
        context_token_threshold: int,
        tool_call_threshold: int,
    ) -> TaskManagerRunResult:
        lifecycle = self._run_lifecycle()
        if not lifecycle.messages:
            self._load_lifecycle_runtime(lifecycle, session_id)
        result = await lifecycle.run(
            agent_process=agent_process,
            cancel_event=cancel_event,
            context_token_threshold=context_token_threshold,
            tool_call_threshold=tool_call_threshold,
        )
        self._set_active_task(result.next_task)
        return TaskManagerRunResult(result.next_action, result.error)

    def finish_user_task(self, result: str | None = None, end_message_id: int | None = None) -> ManagedTask:
        user_task = self._require_user_task()
        if isinstance(self._active_lifecycle, TodoTaskLifecycle):
            raise TaskManagerError("Cannot finish user task while a todo is active")
        lifecycle = self._lifecycle_for_task(user_task)
        lifecycle.current_assistant_message_id = end_message_id
        try:
            lifecycle.finish_task(result=result)
        finally:
            lifecycle.current_assistant_message_id = None
        self.refresh_active_task()
        return user_task

    def refresh_active_task(self) -> None:
        current_lifecycle = self._active_lifecycle
        current_task = current_lifecycle.task if current_lifecycle is not None else None
        next_task = (
            self._find_task(current_lifecycle._runtime.next_task_id_to_run)
            if current_lifecycle is not None
            else None
        )
        if next_task is not None and next_task.status != "active":
            next_task = None
        if next_task is None:
            next_task = self._current_or_root_task(current_task)
        self._set_active_task(next_task)

    def _set_active_task(self, task: ManagedTask | None) -> None:
        self.active_task_id = task.id if task is not None else None
        self._active_lifecycle = self._lifecycle_for_task(task) if task is not None else None

    def _run_lifecycle(self) -> BaseTaskLifecycle:
        if self._active_lifecycle is not None:
            return self._active_lifecycle
        return self._lifecycle_for_task(self._require_loaded_user_task())

    def _require_user_task_lifecycle(self) -> UserTaskLifecycle:
        lifecycle = self.active_lifecycle_for_tools()
        if not isinstance(lifecycle, UserTaskLifecycle):
            raise TaskManagerError("Active lifecycle is not a user task lifecycle")
        return lifecycle

    # ------------------------------------------------------------------
    # Task tree and helper utilities
    # ------------------------------------------------------------------

    def build_task_tree(self, root_task_id: int, *, session: Session) -> ManagedTask | None:
        root = self._db.get_managed_task(root_task_id, session=session)
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

    def _create_user_task_lifecycle(self, task: UserTask) -> UserTaskLifecycle:
        lifecycle = UserTaskLifecycle()
        self._runtime.next_task = task
        self._runtime.next_task_id_to_run = task.id
        lifecycle.set_data(self._runtime)
        return lifecycle

    def _create_todo_task_lifecycle(self, task: TodoTask) -> TodoTaskLifecycle:
        lifecycle = TodoTaskLifecycle()
        self._runtime.next_task = task
        self._runtime.next_task_id_to_run = task.id
        lifecycle.set_data(self._runtime)
        lifecycle.user_task = self._user_task
        return lifecycle

    def _lifecycle_for_task(self, task: ManagedTask) -> UserTaskLifecycle | TodoTaskLifecycle:
        if self._active_lifecycle is not None and task.id == self._active_lifecycle.task.id:
            return self._active_lifecycle
        if task.kind == "user_task":
            return self._create_user_task_lifecycle(cast(UserTask, task))
        if task.kind == "todo":
            return self._create_todo_task_lifecycle(cast(TodoTask, task))
        raise TaskManagerError("Tool-call tasks do not have a lifecycle")

    def _current_or_root_task(self, task: ManagedTask | None) -> ManagedTask | None:
        if self._user_task is None or self._user_task.status != "active":
            return None
        if task is None:
            return self._user_task
        if task.status != "active":
            return None
        return task

    def _loaded_active_task(self) -> ManagedTask | None:
        if self._user_task is None or self._user_task.status != "active":
            return None
        for task in self._walk_tasks():
            if task.kind == "todo" and task.status == "active":
                return task
        return self._user_task

    def _allocate_task_id(self) -> int:
        if self._runtime.next_task_id_to_allocate is None:
            raise TaskManagerError("Task manager must be loaded before creating tasks")
        task_id = self._runtime.next_task_id_to_allocate
        self._runtime.next_task_id_to_allocate += 1
        return task_id

    def _require_user_task(self) -> UserTask:
        if self._user_task is None or self._user_task.status != "active":
            raise TaskManagerError("No active user task")
        return self._user_task

    def _require_loaded_user_task(self) -> UserTask:
        if self._user_task is None:
            raise TaskManagerError("No loaded user task")
        return self._user_task

    def _walk_tasks(self) -> list[ManagedTask]:
        if self._user_task is None:
            return []
        return self._flatten_task_tree(self._user_task)

    def _find_task(self, task_id: int | None) -> ManagedTask | None:
        if task_id is None or self._user_task is None:
            return None
        for task in self._walk_tasks():
            if task.id == task_id:
                return task
        return None

    def _flatten_task_tree(self, task: ManagedTask) -> list[ManagedTask]:
        tasks: list[ManagedTask] = []
        stack = [task]
        while stack:
            task = stack.pop()
            tasks.append(task)
            stack.extend(reversed(task.children))
        return tasks

    def _load_lifecycle_runtime(self, lifecycle: BaseTaskLifecycle, session_id: str) -> None:
        with self._db.create_session() as session:
            lifecycle.load_messages(
                [
                    MessageEntry(id=message_id, message=message)
                    for message_id, message in self._db.list_runner_message_entries(session_id, session=session)
                ],
                next_message_id=self._db.next_runner_message_id(session=session),
            )
            lifecycle.load_tool_call_log_id(
                self._db.next_runner_tool_call_id(session_id, session=session)
            )
