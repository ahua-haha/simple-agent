"""Stateful task manager for agent-defined todos."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Mapping, cast

from pi.agent import AgentTool

from simple_agent.task_manager.lifecycle import BaseTaskLifecycle, TodoTaskLifecycle, UserTaskLifecycle, todo_status_text
from simple_agent.task_manager.models import ManagedTask, TodoTask, ToolCallTask, UserTask
from simple_agent.task_manager.review import (
    TaskTreeReview,
    TaskTreeReviewFormat,
    TaskTreeReviewRenderer,
    ToolCallReview,
)

if TYPE_CHECKING:
    from simple_agent.db.db import Database
    from pi.agent.types import AgentMessage
    from sqlmodel import Session


class TaskManagerError(RuntimeError):
    """Raised when task-manager lifecycle rules are violated."""


class TaskManager:
    """Manage one user task and one active todo at a time."""

    _db: Database
    active_task_id: int | None
    _user_task: UserTask | None
    _active_lifecycle: UserTaskLifecycle | TodoTaskLifecycle | None
    _next_task_id: int | None

    def __init__(self, db: Database):
        self._db = db
        self.active_task_id: int | None = None
        self._user_task: UserTask | None = None
        self._active_lifecycle: UserTaskLifecycle | TodoTaskLifecycle | None = None
        self._next_task_id: int | None = None

    def load(self, user_task_id: int | None, *, session: Session) -> None:
        self._user_task = None
        self._active_lifecycle = None
        self.active_task_id = None
        self._next_task_id = self._db.next_managed_task_id(session=session)
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

    @property
    def user_task(self) -> UserTask | None:
        return self._user_task

    def active_task_for_tools(self) -> ManagedTask:
        return self.active_lifecycle_for_tools().task

    def active_lifecycle_for_tools(self) -> BaseTaskLifecycle:
        if self._active_lifecycle is None:
            raise TaskManagerError("No active task lifecycle")
        return self._active_lifecycle

    def allocate_task_id(self) -> int:
        return self._allocate_task_id()

    def create_tools(self) -> list[AgentTool]:
        return self.active_lifecycle_for_tools().create_tools()

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
        next_task = current_lifecycle.consume_next_task() if current_lifecycle is not None else None
        if next_task is None:
            next_task = self._current_or_root_task(current_task)
        self._set_active_task(next_task)

    def _set_active_task(self, task: ManagedTask | None) -> None:
        self.active_task_id = task.id if task is not None else None
        self._active_lifecycle = self._lifecycle_for_task(task) if task is not None else None

    def todo_status_text(self) -> str:
        return todo_status_text(self._require_user_task())

    def user_instruction_text(self) -> str:
        if self._user_task is None:
            return (
                "Runtime instruction for this turn:\n"
                "- Wait for the user to provide a task before creating todos or doing tool work."
            )

        return self.active_lifecycle_for_tools().instruction_text()

    # ------------------------------------------------------------------
    # Compact phase
    # ------------------------------------------------------------------

    def begin_compact(self, *, run_done: bool) -> bool:
        if not run_done:
            return False
        lifecycle = self._lifecycle_for_user_task_compaction()
        if not lifecycle.begin_compaction():
            return False
        self.active_task_id = lifecycle.task.id
        self._active_lifecycle = lifecycle
        return True

    def compact_instruction_text(
        self,
        *,
        session_id: str,
    ) -> str:
        return self._lifecycle_for_user_task_compaction().compaction_instruction_text(
            tool_calls=self._load_tool_call_reviews(session_id),
        )

    def compacted_messages(self) -> tuple[int, int, list["AgentMessage"]]:
        return self._lifecycle_for_user_task_compaction().compaction_result()

    def create_compact_tools(self) -> list[AgentTool]:
        return self._lifecycle_for_user_task_compaction().create_compact_tools()

    def sync_compaction(self, *, session: Session) -> ManagedTask:
        return self._lifecycle_for_user_task_compaction().sync_compaction(self._db, session)

    # ------------------------------------------------------------------
    # Task tree and helper utilities
    # ------------------------------------------------------------------

    def review_task_tree(
        self,
        *,
        format: TaskTreeReviewFormat = "tree",
        depth: int | None = None,
        tool_calls: Mapping[int, ToolCallReview] | None = None,
    ) -> TaskTreeReview:
        user_task = self._require_user_task()
        renderer = TaskTreeReviewRenderer(format=format, depth=depth, tool_calls=tool_calls or {})
        return renderer.render(user_task)

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
        return UserTaskLifecycle(task, allocate_task_id=self.allocate_task_id)

    def _create_todo_task_lifecycle(self, task: TodoTask) -> TodoTaskLifecycle:
        return TodoTaskLifecycle(task, allocate_task_id=self.allocate_task_id, user_task=self._user_task)

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
        if self._next_task_id is None:
            raise TaskManagerError("Task manager must be loaded before creating tasks")
        task_id = self._next_task_id
        self._next_task_id += 1
        return task_id

    def _require_user_task(self) -> UserTask:
        if self._user_task is None or self._user_task.status != "active":
            raise TaskManagerError("No active user task")
        return self._user_task

    def _require_loaded_user_task(self) -> UserTask:
        if self._user_task is None:
            raise TaskManagerError("No loaded user task")
        return self._user_task

    def _lifecycle_for_user_task_compaction(self) -> UserTaskLifecycle:
        user_task = self._require_loaded_user_task()
        lifecycle = self._lifecycle_for_task(user_task)
        if not isinstance(lifecycle, UserTaskLifecycle):
            raise TaskManagerError("User task compaction requires a user task lifecycle")
        return lifecycle

    def _walk_tasks(self) -> list[ManagedTask]:
        if self._user_task is None:
            return []
        return self._flatten_task_tree(self._user_task)

    def _flatten_task_tree(self, task: ManagedTask) -> list[ManagedTask]:
        tasks: list[ManagedTask] = []
        stack = [task]
        while stack:
            task = stack.pop()
            tasks.append(task)
            stack.extend(reversed(task.children))
        return tasks

    def _load_tool_call_reviews(self, session_id: str) -> dict[int, ToolCallReview]:
        records = self._db.list_runner_tool_calls(session_id)
        return {
            record.id: ToolCallReview(
                name=record.tool_name,
                arguments=self._tool_call_arguments(record.tool_call_json),
            )
            for record in records
            if record.id is not None
        }

    def _tool_call_arguments(self, tool_call_json: str) -> object | None:
        try:
            payload = json.loads(tool_call_json)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        return payload.get("arguments")
