"""Typed task-manager models."""

from __future__ import annotations

import json
import time
from typing import Literal

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
