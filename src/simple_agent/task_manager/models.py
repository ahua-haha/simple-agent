"""Typed task-manager models."""

from __future__ import annotations

import json
import time
from typing import Any, Literal

from pi.agent import AgentTool, AgentToolResult
from pi.ai.types import TextContent
from pydantic import BaseModel, Field

TaskKind = Literal["user_task", "todo", "tool_call"]
TaskStatus = Literal["active", "done", "error"]


class TaskRuntimeContext(BaseModel):
    """Transient runtime data used by task lifecycle decisions."""

    session_id: str
    context_tokens: int
    total_tool_calls: int
    active_task_tool_calls: int
    current_assistant_message_id: int | None = None
    run_done: bool = False


class BaseTask(BaseModel):
    """Common in-memory task fields."""

    id: int | None = None
    parent_id: int | None = None
    kind: TaskKind
    status: TaskStatus = "active"
    children: list["ManagedTask"] = Field(default_factory=list, exclude=True)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def touch(self) -> None:
        self.updated_at = time.time()

    def metadata_json(self) -> str:
        return self.model_dump_json(exclude={"id", "parent_id", "kind", "status", "children"})


class UserTask(BaseTask):
    kind: Literal["user_task"] = "user_task"
    title: str
    result: str | None = None
    error: str | None = None
    start_message_id: int | None = None
    end_message_id: int | None = None

    def instruction_text(self, context: TaskRuntimeContext) -> str:
        if context.active_task_tool_calls > 5:
            return (
                "Runtime instruction for this turn:\n"
                "- More than 5 tool calls have run since the previous todo.\n"
                "- Stop and create a small atomic todo before doing more work.\n"
                "- The todo should describe only the next coherent unit of work."
            )
        return (
            "Runtime instruction for this turn:\n"
            "- Determine whether the user task is complex before doing more work.\n"
            "- If it is complex or long-running, create the next small atomic todo first.\n"
            "- If it is simple, answer directly or use the needed tools."
        )

    def create_todo_task(
        self,
        *,
        task_id: int,
        title: str,
        start_message_id: int | None = None,
    ) -> "TodoTask":
        todo = TodoTask(
            id=task_id,
            title=title,
            parent_id=self.id,
            start_message_id=start_message_id,
        )
        self.children.append(todo)
        self.touch()
        return todo

    def finish_task(self, *, result: str | None = None, end_message_id: int | None = None) -> "UserTask":
        self.status = "done"
        self.result = result
        self.end_message_id = end_message_id
        self.touch()
        return self

    def append_tool_call_task(
        self,
        *,
        task_id: int,
        tool_call_log_id: int,
        assistant_message: Any | None = None,
        tool_result_message: Any | None = None,
    ) -> "ToolCallTask":
        tool_call_task = ToolCallTask(
            id=task_id,
            title=f"Tool call {tool_call_log_id}",
            status="done",
            parent_id=self.id,
            tool_call_log_id=tool_call_log_id,
        )
        self.children.append(tool_call_task)
        self.touch()
        return tool_call_task

    def create_tools(self, manager: Any) -> list[AgentTool]:
        return [
            self.create_create_todo_tool(manager),
            self.create_finish_user_task_tool(manager),
        ]

    def sync(self, database: Any, session: Any) -> None:
        database.upsert_managed_task(self, session=session)
        for child in sorted(self.children, key=lambda item: item.id or 0):
            database.upsert_managed_task(child, session=session)

    def create_create_todo_tool(self, manager: Any) -> AgentTool:
        tool = AgentTool(
            name="create_todo",
            description=(
                "Create the next todo item for the current session task list. "
                "Use for complex tasks with 3+ steps or when the user provides "
                "multiple tasks. Create items in priority order. Only one todo "
                "may be active at a time."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short content for the next coherent unit of work.",
                    },
                },
                "required": ["title"],
            },
        )

        async def execute(tool_call_id, params, cancel_event=None, on_update=None):
            manager.create_todo(params["title"])
            return AgentToolResult(content=[TextContent(text=manager.todo_status_text())])

        tool.execute = execute
        return tool

    def create_finish_user_task_tool(self, manager: Any) -> AgentTool:
        tool = AgentTool(
            name="finish_user_task",
            description=(
                "Mark the current user task as completed. Call when the user's "
                "request is fully satisfied and no todo is active."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "result": {"type": "string", "description": "Optional concise result for this user task"},
                },
                "required": [],
            },
        )

        async def execute(tool_call_id, params, cancel_event=None, on_update=None):
            task = manager.finish_user_task(params.get("result"))
            return AgentToolResult(content=[TextContent(text=f"User task finished: {task.result or task.title}")])

        tool.execute = execute
        return tool

    @classmethod
    def from_metadata(
        cls,
        *,
        id: int | None,
        parent_id: int | None,
        status: str,
        metadata: str,
    ) -> "UserTask":
        return cls(id=id, parent_id=parent_id, status=status, **_metadata_dict(metadata))


class TodoTask(BaseTask):
    kind: Literal["todo"] = "todo"
    title: str
    result: str | None = None
    error: str | None = None
    start_message_id: int | None = None
    end_message_id: int | None = None

    def instruction_text(self, context: TaskRuntimeContext) -> str:
        if context.active_task_tool_calls > 10:
            return (
                "Runtime instruction for this turn:\n"
                "- More than 10 tool calls have run for the active todo.\n"
                "- Determine whether the active todo is finished.\n"
                "- If it is finished, call finish_todo now with a concise result.\n"
                "- If it is not finished, do only the next action needed to complete it."
            )
        return (
            "Runtime instruction for this turn:\n"
            f"- Focus on the active todo: {self.title}\n"
            "- Use tools only for work needed by this todo.\n"
            "- Call finish_todo immediately when it is complete."
        )

    def finish_task(self, *, result: str | None = None, end_message_id: int | None = None) -> "TodoTask":
        self.status = "done"
        self.result = result
        self.end_message_id = end_message_id
        self.touch()
        return self

    def error_task(self, *, error: str, end_message_id: int | None = None) -> "TodoTask":
        self.status = "error"
        self.error = error
        self.end_message_id = end_message_id
        self.touch()
        return self

    def append_tool_call_task(
        self,
        *,
        task_id: int,
        tool_call_log_id: int,
        assistant_message: Any | None = None,
        tool_result_message: Any | None = None,
    ) -> "ToolCallTask":
        tool_call_task = ToolCallTask(
            id=task_id,
            title=f"Tool call {tool_call_log_id}",
            status="done",
            parent_id=self.id,
            tool_call_log_id=tool_call_log_id,
        )
        self.children.append(tool_call_task)
        self.touch()
        return tool_call_task

    def create_tools(self, manager: Any) -> list[AgentTool]:
        return [
            self.create_finish_todo_tool(manager),
            self.create_error_todo_tool(manager),
        ]

    def sync(self, database: Any, session: Any) -> None:
        database.upsert_managed_task(self, session=session)
        for child in sorted(self.children, key=lambda item: item.id or 0):
            database.upsert_managed_task(child, session=session)

    def create_finish_todo_tool(self, manager: Any) -> AgentTool:
        tool = AgentTool(
            name="finish_todo",
            description=(
                "Mark the active todo as completed. Call immediately when the "
                "todo is done before moving to the next item."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "result": {"type": "string", "description": "Optional concise result for this todo"},
                },
                "required": [],
            },
        )

        async def execute(tool_call_id, params, cancel_event=None, on_update=None):
            manager.finish_task(params.get("result"))
            return AgentToolResult(content=[TextContent(text=manager.todo_status_text())])

        tool.execute = execute
        return tool

    def create_error_todo_tool(self, manager: Any) -> AgentTool:
        tool = AgentTool(
            name="error_todo",
            description=(
                "Cancel the active todo because it cannot be completed. If "
                "there is a clear next step, create a revised todo after this."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "error": {"type": "string", "description": "Error details for the active todo"},
                },
                "required": ["error"],
            },
        )

        async def execute(tool_call_id, params, cancel_event=None, on_update=None):
            manager.error_task(params["error"])
            return AgentToolResult(content=[TextContent(text=manager.todo_status_text())])

        tool.execute = execute
        return tool

    @classmethod
    def from_metadata(
        cls,
        *,
        id: int | None,
        parent_id: int | None,
        status: str,
        metadata: str,
    ) -> "TodoTask":
        return cls(id=id, parent_id=parent_id, status=status, **_metadata_dict(metadata))


class ToolCallTask(BaseTask):
    kind: Literal["tool_call"] = "tool_call"
    title: str
    tool_call_log_id: int | None = None

    def sync(self, database: Any, session: Any) -> None:
        database.upsert_managed_task(self, session=session)

    @classmethod
    def from_metadata(
        cls,
        *,
        id: int | None,
        parent_id: int | None,
        status: str,
        metadata: str,
    ) -> "ToolCallTask":
        return cls(id=id, parent_id=parent_id, status=status, **_metadata_dict(metadata))


ManagedTask = UserTask | TodoTask | ToolCallTask


def task_from_metadata(
    *,
    id: int | None,
    parent_id: int | None,
    kind: str,
    status: str,
    metadata: str,
) -> ManagedTask:
    if kind == "user_task":
        return UserTask.from_metadata(id=id, parent_id=parent_id, status=status, metadata=metadata)
    if kind == "todo":
        return TodoTask.from_metadata(id=id, parent_id=parent_id, status=status, metadata=metadata)
    if kind == "tool_call":
        return ToolCallTask.from_metadata(id=id, parent_id=parent_id, status=status, metadata=metadata)
    raise ValueError(f"Unknown task kind: {kind}")


def _metadata_dict(metadata: str) -> dict:
    payload = json.loads(metadata or "{}")
    if not isinstance(payload, dict):
        raise ValueError("Task metadata must be a JSON object")
    return payload
