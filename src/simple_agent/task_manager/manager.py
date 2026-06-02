"""Stateful task manager for agent-defined todos."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pi.agent import AgentTool, AgentToolResult
from pi.ai.types import TextContent

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
        self._tasks: dict[int, ManagedTask] = {}
        self._next_task_id: int | None = None

    def load(self, active_user_task_id: int | None) -> None:
        self._tasks = {}
        self.active_user_task_id = active_user_task_id
        self.active_todo_id = None
        self._next_task_id = self._db.next_managed_task_id()
        if active_user_task_id is None:
            return
        user_task = self._load_task_tree(active_user_task_id)
        if user_task is None:
            raise TaskManagerError("Active user task is missing")
        for task in self._tasks.values():
            if task.kind == "todo" and task.status == "active":
                self.active_todo_id = task.id
                break

    def save(self, session=None) -> None:
        if session is None:
            with self._db.create_session() as session:
                self.save(session=session)
                session.commit()
            return

        for task_id in sorted(self._tasks):
            task = self._tasks[task_id]
            self._db.upsert_managed_task(task, session=session)

    def create_user_task(self, input: str) -> ManagedTask:
        if self.active_user_task_id is not None:
            raise TaskManagerError("Cannot create a second active user task")
        task = ManagedTask(kind="user_task", title=input)
        task.id = self._allocate_task_id()
        self._tasks[task.id] = task
        self.active_user_task_id = task.id
        return task

    def create_create_todo_tool(self) -> AgentTool:
        tool = AgentTool(
            name="create_todo",
            description="Create a todo for the next coherent unit of work.",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short title for the todo"},
                },
                "required": ["title"],
            },
        )

        async def execute(tool_call_id, params, cancel_event=None, on_update=None):
            todo = self.create_todo(params["title"])
            return AgentToolResult(content=[TextContent(text=f"created todo {todo.id}")])

        tool.execute = execute
        return tool

    def create_finish_todo_tool(self) -> AgentTool:
        tool = AgentTool(
            name="finish_todo",
            description="Mark the active todo as done.",
            parameters={
                "type": "object",
                "properties": {
                    "result": {"type": "string", "description": "Optional concise result for this todo"},
                },
                "required": [],
            },
        )

        async def execute(tool_call_id, params, cancel_event=None, on_update=None):
            todo = self.finish_task(params.get("result"))
            return AgentToolResult(content=[TextContent(text=f"finished todo {todo.id}")])

        tool.execute = execute
        return tool

    def create_error_todo_tool(self) -> AgentTool:
        tool = AgentTool(
            name="error_todo",
            description="Mark the active todo as failed.",
            parameters={
                "type": "object",
                "properties": {
                    "error": {"type": "string", "description": "Error details for the active todo"},
                },
                "required": ["error"],
            },
        )

        async def execute(tool_call_id, params, cancel_event=None, on_update=None):
            todo = self.error_task(params["error"])
            return AgentToolResult(content=[TextContent(text=f"errored todo {todo.id}")])

        tool.execute = execute
        return tool

    def create_todo(self, title: str) -> ManagedTask:
        if self.active_user_task_id is None:
            raise TaskManagerError("No active user task")
        user_task = self._get_task(self.active_user_task_id)
        if user_task is None:
            raise TaskManagerError("Active user task is missing")
        if self.active_todo_id is not None:
            raise TaskManagerError("Cannot create todo while another active todo exists")

        todo = ManagedTask(kind="todo", title=title, parent_id=user_task.id)
        todo.id = self._allocate_task_id()
        self._tasks[todo.id] = todo

        user_task.items.append(TaskItem(kind="task", ref_id=todo.id))
        user_task.touch()
        self.active_todo_id = todo.id
        return todo

    def finish_task(self, result: str | None = None) -> ManagedTask:
        todo = self._require_active_todo()
        todo.status = "done"
        todo.result = result
        todo.touch()
        self.active_todo_id = None
        return todo

    def error_task(self, error: str) -> ManagedTask:
        todo = self._require_active_todo()
        todo.status = "error"
        todo.error = error
        todo.touch()
        self.active_todo_id = None
        return todo

    def record_tool_call(self, tool_call_id: int, task_id: int | None = None) -> None:
        if task_id is not None:
            target = self._get_task(task_id)
            if target is None:
                raise TaskManagerError("Cannot record tool call for missing task")
        else:
            if self.active_todo_id is not None:
                target = self._require_active_todo()
            elif self.active_user_task_id is not None:
                target = self._get_task(self.active_user_task_id)
                if target is None:
                    raise TaskManagerError("Active user task is missing")
            else:
                raise TaskManagerError("No active user task")
        target.items.append(TaskItem(kind="tool_call", ref_id=tool_call_id))
        target.touch()

    def finish_user_task(self, result: str | None = None) -> ManagedTask:
        if self.active_user_task_id is None:
            raise TaskManagerError("No active user task")
        user_task = self._get_task(self.active_user_task_id)
        if user_task is None:
            raise TaskManagerError("Active user task is missing")
        if self.active_todo_id is not None:
            raise TaskManagerError("Cannot finish user task while a todo is active")
        user_task.status = "done"
        user_task.result = result
        user_task.touch()
        self.active_user_task_id = None
        return user_task

    def _require_active_todo(self) -> ManagedTask:
        if self.active_todo_id is None:
            raise TaskManagerError("No active todo")
        task = self._get_task(self.active_todo_id)
        if task is None:
            raise TaskManagerError("Active todo is missing")
        return task

    def _get_task(self, task_id: int) -> ManagedTask | None:
        return self._tasks.get(task_id)

    def _allocate_task_id(self) -> int:
        if self._next_task_id is None:
            raise TaskManagerError("Task manager must be loaded before creating tasks")
        task_id = self._next_task_id
        self._next_task_id += 1
        return task_id

    def _load_task_tree(self, task_id: int) -> ManagedTask | None:
        task = self._db.get_managed_task(task_id)
        if task is None or task.id is None:
            return None
        self._tasks[task.id] = task
        for item in task.items:
            if item.kind == "task":
                self._load_task_tree(item.ref_id)
        return task
