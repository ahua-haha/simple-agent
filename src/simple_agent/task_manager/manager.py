"""Stateful task manager for agent-defined todos."""

from __future__ import annotations

from typing import TYPE_CHECKING

from simple_agent.task_manager.models import ManagedTask, TaskItem

if TYPE_CHECKING:
    from simple_agent.db.db import Database


class TaskManagerError(RuntimeError):
    """Raised when task-manager lifecycle rules are violated."""


class TaskManager:
    """Manage one user task and one active todo at a time."""

    def __init__(self, db: Database):
        self._db = db
        self.active_user_task_id: int | None = None
        self.active_todo_id: int | None = None

    def create_user_task(self, input: str) -> ManagedTask:
        if self.active_user_task_id is not None:
            raise TaskManagerError("Cannot create a second active user task")
        task = ManagedTask(kind="user_task", title=input)
        task.id = self._db.upsert_managed_task(task)
        self.active_user_task_id = task.id
        return task

    def create_todo(self, title: str) -> ManagedTask:
        user_task = self._require_user_task()
        if self.active_todo_id is not None:
            raise TaskManagerError("Cannot create todo while another active todo exists")

        todo = ManagedTask(kind="todo", title=title, parent_id=user_task.id)
        todo.id = self._db.upsert_managed_task(todo)

        user_task.items.append(TaskItem(kind="task", ref_id=todo.id))
        user_task.touch()
        self._db.upsert_managed_task(user_task)
        self.active_todo_id = todo.id
        return todo

    def finish_todo(self, result: str | None = None) -> ManagedTask:
        todo = self._require_active_todo()
        todo.status = "done"
        todo.result = result
        todo.touch()
        self._db.upsert_managed_task(todo)
        self.active_todo_id = None
        return todo

    def error_todo(self, error: str) -> ManagedTask:
        todo = self._require_active_todo()
        todo.status = "error"
        todo.error = error
        todo.touch()
        self._db.upsert_managed_task(todo)
        self.active_todo_id = None
        return todo

    def finish_user_task(self, result: str | None = None) -> ManagedTask:
        user_task = self._require_user_task()
        if self.active_todo_id is not None:
            raise TaskManagerError("Cannot finish user task while a todo is active")
        user_task.status = "done"
        user_task.result = result
        user_task.touch()
        self._db.upsert_managed_task(user_task)
        self.active_user_task_id = None
        return user_task

    def _require_user_task(self) -> ManagedTask:
        if self.active_user_task_id is None:
            raise TaskManagerError("No active user task")
        task = self._db.get_managed_task(self.active_user_task_id)
        if task is None:
            raise TaskManagerError("Active user task is missing")
        return task

    def _require_active_todo(self) -> ManagedTask:
        if self.active_todo_id is None:
            raise TaskManagerError("No active todo")
        task = self._db.get_managed_task(self.active_todo_id)
        if task is None:
            raise TaskManagerError("Active todo is missing")
        return task
