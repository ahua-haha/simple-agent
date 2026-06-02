"""State module — domain models, DB record classes, and serialization."""

from __future__ import annotations

import json
import time

from pydantic import BaseModel, TypeAdapter
from sqlmodel import SQLModel, Field

from pi.agent.types import AgentMessage
from simple_agent.task_manager.models import ManagedTask, TaskItem


# ── DB record classes ────────────────────────────────────────────────


class SessionRecord(SQLModel, table=True):
    """SQLite model for session metadata."""

    id: str = Field(primary_key=True)
    name: str = Field(default="")
    cursor_id: int | None = Field(default=None)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class TaskRecord(SQLModel, table=True):
    """SQLite model for task tree persistence.

    Each task is a flat row.  ``parent_id`` and ``running_task_id`` are
    plain ints (no FK) resolved to object refs in memory on load.
    """

    id: int | None = Field(default=None, primary_key=True)
    parent_id: int | None = Field(default=None, index=True)
    running_task_id: int | None = Field(default=None)
    finished_task_ids: str | None = None  # JSON: list[int]
    type: str = Field(default="single_run")
    state: str = Field(default="PENDING")
    input: str = ""
    messages: str | None = None  # JSON: list[AgentMessage]
    result: str | None = None    # JSON: list[TextResult]
    result_msg: str | None = None  # JSON: list[AgentMessage]
    repo_path: str = "."
    start_snapshot: str | None = None
    end_snapshot: str | None = None


class ManagedTaskRecord(SQLModel, table=True):
    """SQLite model for the replacement task manager."""

    id: int | None = Field(default=None, primary_key=True)
    parent_id: int | None = Field(default=None, index=True)
    kind: str = Field(index=True)
    title: str
    status: str = Field(default="active", index=True)
    items: str | None = None
    result: str | None = None
    error: str | None = None
    create_tool_call_id: str | None = Field(default=None, index=True)
    end_tool_call_id: str | None = Field(default=None, index=True)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class RunnerStateMetadataRecord(SQLModel, table=True):
    """SQLite model for session-runner lifecycle metadata."""

    session_id: str = Field(primary_key=True)
    phase: str = Field(default="idle", index=True)
    status: str = Field(default="idle", index=True)
    active_user_task_id: int | None = Field(default=None, index=True)
    last_error: str | None = None
    version: int = Field(default=1)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class RunnerMessageRecord(SQLModel, table=True):
    """SQLite model for ordered session-runner messages."""

    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    seq: int = Field(index=True)
    role: str = Field(index=True)
    content_json: str
    timestamp_ms: int | None = Field(default=None)
    created_at: float = Field(default_factory=time.time)


class RunnerToolCallRecord(SQLModel, table=True):
    """SQLite model for structured session-runner tool execution logs."""

    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    tool_call_id: str = Field(index=True)
    tool_name: str = Field(index=True)
    params_json: str
    result_json: str | None = None
    status: str = Field(index=True)
    started_at: float
    finished_at: float | None = None
    error: str | None = None


# ── Domain models ────────────────────────────────────────────────────


class TextResult(BaseModel):
    desc: str
    toolCallLogID: list[int]


TEXT_RESULT_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "desc": {"type": "string", "description": "Description of the result"},
        "toolCallLogID": {"type": "array", "items": {"type": "integer"}, "description": "List of tool call log IDs"},
    },
    "required": ["desc", "toolCallLogID"],
}

_message_adapter = TypeAdapter(list[AgentMessage])
_single_message_adapter = TypeAdapter(AgentMessage)
_result_adapter = TypeAdapter(list[TextResult])
_task_item_adapter = TypeAdapter(list[TaskItem])


def agent_message_to_json(message: AgentMessage) -> str:
    return _single_message_adapter.dump_json(message).decode("utf-8")


def agent_message_from_json(payload: str) -> AgentMessage:
    return _single_message_adapter.validate_json(payload)


def managed_task_to_record(task: ManagedTask) -> ManagedTaskRecord:
    return ManagedTaskRecord(
        id=task.id,
        parent_id=task.parent_id,
        kind=task.kind,
        title=task.title,
        status=task.status,
        items=_task_item_adapter.dump_json(task.items).decode("utf-8"),
        result=task.result,
        error=task.error,
        create_tool_call_id=task.create_tool_call_id,
        end_tool_call_id=task.end_tool_call_id,
        created_at=task.created_at,
        updated_at=task.updated_at,
    )


def managed_task_from_record(record: ManagedTaskRecord) -> ManagedTask:
    return ManagedTask(
        id=record.id,
        parent_id=record.parent_id,
        kind=record.kind,
        title=record.title,
        status=record.status,
        items=_task_item_adapter.validate_json(record.items or "[]"),
        result=record.result,
        error=record.error,
        create_tool_call_id=record.create_tool_call_id,
        end_tool_call_id=record.end_tool_call_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


class Task(BaseModel):
    """A node in the task tree.

    Persisted as a flat row in SQLite.  Relationships use IDs (no
    object refs) — ``parent_id``, ``running_task_id``, and
    ``finished_task_ids`` encode the tree structure.

    ``metadata`` is a runtime-only dict for objects like RepoWatcher
    and pre-built context messages.  It is NOT persisted.
    """

    type: str = "single_run"
    state: str = "PENDING"
    input: str
    result: list[TextResult] = None
    messages: list[AgentMessage] = None
    result_msg: list[AgentMessage] = []
    repo_path: str = "."
    start_snapshot: str | None = None
    end_snapshot: str | None = None
    # tree structure — ID-based, flat
    id: int | None = None
    parent_id: int | None = None
    running_task_id: int | None = None
    finished_task_ids: list[int] = []
    # in-memory object ref (wired by from_db_rows)
    running_task: "Task | None" = None

    model_config = {"arbitrary_types_allowed": True}

    def to_db_row(self) -> TaskRecord:
        """Return a new ``TaskRecord`` with serialized fields."""
        return TaskRecord(
            id=self.id,
            parent_id=self.parent_id,
            running_task_id=self.running_task_id,
            finished_task_ids=json.dumps(self.finished_task_ids or []),
            type=self.type,
            state=self.state,
            input=self.input,
            messages=_message_adapter.dump_json(self.messages or []).decode("utf-8"),
            result=_result_adapter.dump_json(self.result or []).decode("utf-8"),
            result_msg=_message_adapter.dump_json(self.result_msg or []).decode("utf-8"),
            repo_path=self.repo_path,
            start_snapshot=self.start_snapshot,
            end_snapshot=self.end_snapshot,
        )

    def __init__(self, **data):
        super().__init__(**data)
        if self.result is None:
            self.result = []
        if self.messages is None:
            self.messages = []
        if self.result_msg is None:
            self.result_msg = []
        # runtime cache, not persisted
        object.__setattr__(self, "metadata", {})

    def context(self, tasks_by_id: dict[int, "Task"] | None = None) -> list[AgentMessage]:
        """Return the ancestor message chain.

        If *tasks_by_id* is provided, walks ancestors via ``parent_id``
        lookups.  Otherwise only returns ``self.messages``.
        """
        msgs: list[AgentMessage] = []
        if tasks_by_id is not None and self.parent_id is not None:
            parent = tasks_by_id.get(self.parent_id)
            if parent is not None:
                msgs.extend(parent.context(tasks_by_id))
        msgs.extend(self.messages)
        return msgs

    def find_active(self) -> "Task":
        """Walk ``running_task`` chain to find the single active node."""
        if self.state != "FINISHED" and self.running_task is None:
            return self
        if self.running_task is not None:
            return self.running_task.find_active()
        return self

    @staticmethod
    def from_db_rows(records: list[TaskRecord]) -> dict[int, "Task"]:
        """Build Task objects from ``TaskRecord`` rows.

        Returns a dict mapping ``id → Task`` with ``running_task``
        object refs wired.  Callers can find the root via
        ``parent_id is None``.
        """
        tasks_by_id: dict[int, Task] = {}
        for r in records:
            task = Task(
                id=r.id,
                parent_id=r.parent_id,
                running_task_id=r.running_task_id,
                finished_task_ids=json.loads(r.finished_task_ids or "[]"),
                type=r.type,
                state=r.state,
                input=r.input,
                messages=_message_adapter.validate_json(r.messages or "[]"),
                result=_result_adapter.validate_json(r.result or "[]"),
                result_msg=_message_adapter.validate_json(r.result_msg or "[]"),
                repo_path=r.repo_path or ".",
                start_snapshot=r.start_snapshot,
                end_snapshot=r.end_snapshot,
            )
            tasks_by_id[task.id] = task

        for task in tasks_by_id.values():
            if task.running_task_id is not None:
                task.running_task = tasks_by_id.get(task.running_task_id)

        return tasks_by_id

class StateClarification(BaseModel):
    state: str
    reason: str
