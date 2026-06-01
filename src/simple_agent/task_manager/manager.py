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

    def create_user_task(self, input: str) -> ManagedTask:
        if self.active_user_task_id is not None:
            raise TaskManagerError("Cannot create a second active user task")
        task = ManagedTask(kind="user_task", title=input)
        task.id = self._db.upsert_managed_task(task)
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
            todo = self.finish_todo(params.get("result"))
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
            todo = self.error_todo(params["error"])
            return AgentToolResult(content=[TextContent(text=f"errored todo {todo.id}")])

        tool.execute = execute
        return tool

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

    def record_tool_call(self, tool_call_id: int) -> None:
        target = self._require_active_todo() if self.active_todo_id is not None else self._require_user_task()
        target.items.append(TaskItem(kind="tool_call", ref_id=tool_call_id))
        target.touch()
        self._db.upsert_managed_task(target)

    def compact_items(
        self,
        parent_task_id: int,
        item_refs: list[TaskItem],
        title: str,
        result: str,
        items: list[TaskItem],
    ) -> ManagedTask:
        if not item_refs:
            raise TaskManagerError("Cannot compact an empty item list")

        seen = {(item.kind, item.ref_id) for item in item_refs}
        if len(seen) != len(item_refs):
            raise TaskManagerError("Cannot compact duplicate refs")

        parent = self._db.get_managed_task(parent_task_id)
        if parent is None:
            raise TaskManagerError("Parent task is missing")

        visible = [(item.kind, item.ref_id) for item in parent.items]
        selected = [(item.kind, item.ref_id) for item in item_refs]
        if not all(ref in visible for ref in selected):
            raise TaskManagerError("Cannot compact refs outside parent visible items")

        for item in item_refs:
            if item.kind != "task":
                continue
            task = self._db.get_managed_task(item.ref_id)
            if task is None:
                raise TaskManagerError("Cannot compact missing task")
            if task.status == "active":
                raise TaskManagerError("Cannot compact active task")

        aggregate = ManagedTask(
            kind="aggregate",
            parent_id=parent.id,
            title=title,
            status="done",
            result=result,
            items=list(items),
        )
        aggregate.id = self._db.upsert_managed_task(aggregate)

        selected_set = set(selected)
        new_items: list[TaskItem] = []
        inserted = False
        for item in parent.items:
            ref = (item.kind, item.ref_id)
            if ref in selected_set:
                if not inserted:
                    new_items.append(TaskItem(kind="task", ref_id=aggregate.id))
                    inserted = True
                continue
            new_items.append(item)

        parent.items = new_items
        parent.touch()
        self._db.upsert_managed_task(parent)
        return aggregate

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
