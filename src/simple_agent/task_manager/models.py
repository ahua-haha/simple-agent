"""Unified task-manager models."""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field

TaskKind = Literal["user_task", "todo", "aggregate"]
TaskStatus = Literal["active", "done", "error"]
TaskItemKind = Literal["task", "tool_call"]


class TaskItem(BaseModel):
    """A visible ordered reference owned by a managed task."""

    kind: TaskItemKind
    ref_id: int


class ManagedTask(BaseModel):
    """Unified task model for user tasks, todos, and aggregate tasks."""

    id: int | None = None
    parent_id: int | None = None
    kind: TaskKind
    title: str
    status: TaskStatus = "active"
    items: list[TaskItem] = Field(default_factory=list)
    result: str | None = None
    error: str | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def touch(self) -> None:
        self.updated_at = time.time()
