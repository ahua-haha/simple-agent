"""Unified task-manager models."""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field

TaskKind = Literal["user_task", "todo", "tool_call", "aggregate"]
TaskStatus = Literal["active", "done", "error"]
class ManagedTask(BaseModel):
    """Unified task model for user tasks, todos, and aggregate tasks."""

    id: int | None = None
    parent_id: int | None = None
    kind: TaskKind
    title: str
    status: TaskStatus = "active"
    seq: str = ""
    result: str | None = None
    error: str | None = None
    create_tool_call_id: str | None = None
    end_tool_call_id: str | None = None
    tool_call_log_id: int | None = None
    children: list["ManagedTask"] = Field(default_factory=list, exclude=True)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def touch(self) -> None:
        self.updated_at = time.time()
