"""Database storage and query module for tool calls and tasks."""

from __future__ import annotations

import json
import time
from typing import Any

from pi.ai import ToolCall, AssistantMessage, ToolResultMessage
from pi.agent import AgentToolResult, AgentMessage
from pi.ai.types import TextContent
from sqlmodel import SQLModel, Field, Session, create_engine, select
import sqlite3
from pydantic import TypeAdapter


from simple_agent.state.state import TextResult, ToolExecMessage


class ToolCallRecord(SQLModel, table=True):
    """SQLite model for tool call executions.

    Stores the full ToolExecMessage serialized as JSON for complete fidelity.
    """
    id: int | None = Field(default=None, primary_key=True)
    tool: str = Field(index=True)
    content: str | None = Field(default=None)  # JSON serialized ToolExecMessage
    created_at: int = Field(default_factory=lambda: int(time.time()), index=True)


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


class Database:
    """SQLite database for tool call storage and task history."""

    def __init__(self, db_path: str = "./data/tool_log.db"):
        self._db_path = db_path
        self._engine = None
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database with sqlmodel."""
        self._engine = create_engine(
            f"sqlite:///{self._db_path}",
            connect_args={"check_same_thread": False}
        )
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
        SQLModel.metadata.create_all(self._engine)

    def _get_session(self) -> Session:
        """Get a new sqlmodel Session."""
        return Session(self._engine)

    # --- ToolCall operations ---

    def next_tool_call_id(self) -> int:
        with self._get_session() as session:
            max_record = session.exec(select(ToolCallRecord).order_by(ToolCallRecord.id.desc())).first()
            return (max_record.id + 1) if max_record else 0

    def insert_tool_call(self, tool_exec: ToolExecMessage) -> int:
        """Insert a tool call record and return its ID.

        Args:
            tool_exec: ToolExecMessage to store

        Returns:
            The auto-generated ID of the inserted record
        """
        with self._get_session() as session:
            # Get next ID
            next_id = self.next_tool_call_id()

            record = ToolCallRecord(
                id=next_id,
                tool=tool_exec.tool_call.name,
                content=json.dumps(tool_exec.model_dump()),
            )
            session.add(record)
            session.commit()
            return next_id

    def get_tool_call(self, id: int) -> ToolExecMessage | None:
        """Get a tool call record by ID."""
        with self._get_session() as session:
            record = session.exec(select(ToolCallRecord).where(ToolCallRecord.id == id)).first()
            if not record:
                return None
            return ToolExecMessage.model_validate_json(record.content)

    def get_tool_calls_by_ids(self, ids: list[int]) -> list[ToolExecMessage]:
        """Get multiple tool call records by IDs, sorted by ID."""
        if not ids:
            return []
        with self._get_session() as session:
            records = session.exec(select(ToolCallRecord).where(ToolCallRecord.id.in_(ids))).all()
            records.sort(key=lambda r: r.id)
            return [ToolExecMessage.model_validate_json(r.content) for r in records]

    def list_tool_calls(self, limit: int = 10) -> list[ToolCallRecord]:
        """List recent tool call records."""
        with self._get_session() as session:
            records = session.exec(select(ToolCallRecord).order_by(ToolCallRecord.id.desc()).limit(limit)).all()
            return list(records)

    # --- Task operations ---

    def upsert_task(self, task) -> int:
        """INSERT or UPDATE a task row.  Returns the task ``id``."""
        message_adapter = TypeAdapter(list[AgentMessage])
        result_adapter = TypeAdapter(list[TextResult])

        import json as _json

        row_id = task.id
        with self._get_session() as session:
            if row_id is not None:
                record = session.get(TaskRecord, row_id)
            else:
                record = None

            if record is None:
                record = TaskRecord()
                session.add(record)

            record.parent_id = task.parent_id
            record.running_task_id = task.running_task_id
            record.finished_task_ids = _json.dumps(task.finished_task_ids or [])
            record.type = task.type
            record.state = task.state
            record.input = task.input
            record.messages = message_adapter.dump_json(task.messages or []).decode("utf-8")
            record.result = result_adapter.dump_json(task.result or []).decode("utf-8")
            record.result_msg = message_adapter.dump_json(task.result_msg or []).decode("utf-8")
            record.repo_path = task.repo_path
            record.start_snapshot = task.start_snapshot
            record.end_snapshot = task.end_snapshot

            session.commit()
            session.refresh(record)
            return record.id

    def get_task(self, task_id: int) -> dict | None:
        """Return a single task row as dict, or None."""
        import json as _json
        message_adapter = TypeAdapter(list[AgentMessage])
        result_adapter = TypeAdapter(list[TextResult])
        with self._get_session() as session:
            record = session.get(TaskRecord, task_id)
            if record is None:
                return None
            return {
                "id": record.id,
                "parent_id": record.parent_id,
                "running_task_id": record.running_task_id,
                "finished_task_ids": _json.loads(record.finished_task_ids or "[]"),
                "type": record.type,
                "state": record.state,
                "input": record.input,
                "messages": message_adapter.validate_json(record.messages or "[]"),
                "result": result_adapter.validate_json(record.result or "[]"),
                "result_msg": message_adapter.validate_json(record.result_msg or "[]"),
                "repo_path": record.repo_path or ".",
                "start_snapshot": record.start_snapshot,
                "end_snapshot": record.end_snapshot,
            }

    def load_all_tasks(self) -> list[dict]:
        """Return all task rows as dicts, ordered by id."""
        import json as _json

        message_adapter = TypeAdapter(list[AgentMessage])
        result_adapter = TypeAdapter(list[TextResult])
        with self._get_session() as session:
            records = session.exec(select(TaskRecord).order_by(TaskRecord.id)).all()
            return [
                {
                    "id": r.id,
                    "parent_id": r.parent_id,
                    "running_task_id": r.running_task_id,
                    "finished_task_ids": _json.loads(r.finished_task_ids or "[]"),
                    "type": r.type,
                    "state": r.state,
                    "input": r.input,
                    "messages": message_adapter.validate_json(r.messages or "[]"),
                    "result": result_adapter.validate_json(r.result or "[]"),
                    "result_msg": message_adapter.validate_json(r.result_msg or "[]"),
                    "repo_path": r.repo_path or ".",
                    "start_snapshot": r.start_snapshot,
                    "end_snapshot": r.end_snapshot,
                }
                for r in records
            ]

    def delete_task(self, task_id: int) -> None:
        """Delete a task row by id."""
        with self._get_session() as session:
            record = session.get(TaskRecord, task_id)
            if record is not None:
                session.delete(record)
                session.commit()
