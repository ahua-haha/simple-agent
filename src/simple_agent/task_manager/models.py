"""Typed task-manager models."""

from __future__ import annotations

import json
import time
from typing import Any, Literal

from pydantic import BaseModel, Field, PrivateAttr

from simple_agent.index.indexer import AgentIndex

TaskKind = Literal["user_task", "todo", "tool_call", "repo_memory"]
TaskStatus = Literal["active", "done", "error"]


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

    def format_for_render(self, *, tool_call: Any | None = None, sequence: int | None = None) -> str:
        return f"{self.kind} [{self.status}] {_task_title(self)}"


class UserTask(BaseTask):
    kind: Literal["user_task"] = "user_task"
    title: str
    result: str | None = None
    error: str | None = None
    start_message_id: int | None = None
    end_message_id: int | None = None

    def format_for_render(self, *, tool_call: Any | None = None, sequence: int | None = None) -> str:
        return f"user_task [{self.status}] {self.title}"

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

    def format_for_render(self, *, tool_call: Any | None = None, sequence: int | None = None) -> str:
        return f"todo [{self.status}] {self.title}"

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
    tool_call_log_id: int | None = None
    tool_call_name: str | None = None
    tool_call_args: Any | None = None

    def format_for_render(self, *, tool_call: Any | None = None, sequence: int | None = None) -> str:
        seq = sequence if sequence is not None else "?"
        tool_name = self.tool_call_name or "unknown_tool"
        line = f"tool_call {seq}. {tool_name}"
        if self.tool_call_args is not None:
            line += f" args: {_truncate_text(_format_tool_call_args(self.tool_call_args), limit=120)}"
        return line

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


class RepoMemoryTask(BaseTask):
    kind: Literal["repo_memory"] = "repo_memory"
    title: str
    repo_path: str = "."
    index_db_path: str
    result: str | None = None
    error: str | None = None
    _agent_index: AgentIndex | None = PrivateAttr(default=None)
    _current_assistant_message_id: int | None = PrivateAttr(default=None)

    def format_for_render(self, *, tool_call: Any | None = None, sequence: int | None = None) -> str:
        return f"repo_memory [{self.status}] {self.title}"

    def agent_index(self) -> AgentIndex:
        if self._agent_index is None:
            self._agent_index = AgentIndex(
                db_path=self.index_db_path,
                base_dir=self.repo_path,
            )
        return self._agent_index

    @property
    def current_assistant_message_id(self) -> int | None:
        return self._current_assistant_message_id

    @current_assistant_message_id.setter
    def current_assistant_message_id(self, message_id: int | None) -> None:
        self._current_assistant_message_id = message_id

    @classmethod
    def from_metadata(
        cls,
        *,
        id: int | None,
        parent_id: int | None,
        status: str,
        metadata: str,
    ) -> "RepoMemoryTask":
        return cls(id=id, parent_id=parent_id, status=status, **_metadata_dict(metadata))


ManagedTask = UserTask | TodoTask | ToolCallTask | RepoMemoryTask


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
    if kind == "repo_memory":
        return RepoMemoryTask.from_metadata(id=id, parent_id=parent_id, status=status, metadata=metadata)
    raise ValueError(f"Unknown task kind: {kind}")


def _metadata_dict(metadata: str) -> dict:
    payload = json.loads(metadata or "{}")
    if not isinstance(payload, dict):
        raise ValueError("Task metadata must be a JSON object")
    return payload


def _task_title(task: BaseTask) -> str:
    title = getattr(task, "title", None)
    return str(title) if title is not None else ""


def _format_tool_call_args(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    if hasattr(arguments, "model_dump_json"):
        return arguments.model_dump_json()
    return json.dumps(arguments, separators=(",", ":"))


def _truncate_text(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
