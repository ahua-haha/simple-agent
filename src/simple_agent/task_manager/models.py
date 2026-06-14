"""Typed task-manager models."""

from __future__ import annotations

import time
from typing import Any, Literal

from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel, Field

_TASK_INFO_ENV = Environment(undefined=StrictUndefined)

TASK_INFO_TEMPLATE = """\
## Current Task
{{ title }} [{{ status }}]

## Task Plan
{{ task_plan }}

## Latest Instruction and Response
Instruction: {{ instruction }}
Response: {{ response }}"""

TaskKind = Literal["user_task"]
TaskStatus = Literal["active", "done", "error", "index_memory_upsert", "compact_finished"]


class UserTask(BaseModel):
    """Single user task that holds all metadata during an agent run.

    Standalone model — does not inherit from BaseTask. The task_plan
    markdown tracks sub-goals. Tool calls are tracked via tool_call_log_ids.
    """

    id: int | None = None
    kind: Literal["user_task"] = "user_task"
    status: TaskStatus = "active"
    title: str
    result: str | None = None
    error: str | None = None
    start_message_id: int | None = None
    end_message_id: int | None = None
    task_plan: str | None = None
    instruction: str | None = None
    response: str | None = None
    tool_call_log_ids: list[int] = Field(default_factory=list)
    compacted_tool_call_log_ids: list[int] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def touch(self) -> None:
        self.updated_at = time.time()

    def format_for_render(self, *, tool_call: Any | None = None, sequence: int | None = None) -> str:
        return f"user_task [{self.status}] {self.title}"

    def task_info(self) -> str:
        """Format the task's current state as a markdown string for the orchestrator."""
        return _TASK_INFO_ENV.from_string(TASK_INFO_TEMPLATE).render(
            title=self.title,
            status=self.status,
            task_plan=self.task_plan or "(no plan yet)",
            instruction=self.instruction or "(none)",
            response=self.response or "(none)",
        )

    def metadata_json(self) -> str:
        return self.model_dump_json(exclude={"id", "kind", "status"})
