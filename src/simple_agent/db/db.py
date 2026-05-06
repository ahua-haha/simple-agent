"""Database storage and query module for tool calls and tasks."""

from __future__ import annotations

import json
import time
from typing import Any

from pi.ai import ToolCall, AssistantMessage, ToolResultMessage
from pi.agent import AgentToolResult, AgentMessage
from pi.ai.types import TextContent
from sqlmodel import SQLModel, Field, Session, create_engine
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
    """SQLite model for task history storage."""
    id: int | None = Field(default=None, primary_key=True)
    type: str = Field(index=True)
    input: str | None = None
    messages: str | None = None  # JSON serialized list of AgentMessage
    results: str | None = None   # JSON serialized list of TextResult
    status: str | None = None
    created_at: int = Field(default_factory=lambda: int(time.time()))


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

    def insert_tool_call(self, tool_exec: ToolExecMessage) -> int:
        """Insert a tool call record and return its ID.

        Args:
            tool_exec: ToolExecMessage to store

        Returns:
            The auto-generated ID of the inserted record
        """
        with self._get_session() as session:
            # Get next ID
            max_record = session.query(ToolCallRecord).order_by(ToolCallRecord.id.desc()).first()
            next_id = (max_record.id + 1) if max_record else 0

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
            record = session.query(ToolCallRecord).filter(ToolCallRecord.id == id).first()
            if not record:
                return None
            return ToolExecMessage.model_validate_json(record.content)

    def get_tool_calls_by_ids(self, ids: list[int]) -> list[ToolExecMessage]:
        """Get multiple tool call records by IDs, sorted by ID."""
        if not ids:
            return []
        with self._get_session() as session:
            records = session.query(ToolCallRecord).filter(ToolCallRecord.id.in_(ids)).all()
            records.sort(key=lambda r: r.id)
            return [ToolExecMessage.model_validate_json(r.content) for r in records]

    # --- Task operations ---

    def save_task(self, task_type: str, task_input: str, messages: list, results: list, status: str):
        """Save a completed task to SQLite.

        Args:
            task_type: Type of task ('explore' or 'single_run')
            task_input: The task input description
            messages: List of AgentMessage objects (serialized to JSON)
            results: List of TextResult objects (serialized to JSON)
            status: Task status ('finished', 'error', etc.)
        """
        message_adapter = TypeAdapter(list[AgentMessage])
        result_adapter = TypeAdapter(list[TextResult])
        with self._get_session() as session:
            record = TaskRecord(
                type=task_type,
                input=task_input,
                messages=message_adapter.dump_json(messages or []).decode("utf-8"),
                result=result_adapter.dump_json(results or []).decode("utf-8"),
                status=status,
            )
            session.add(record)
            session.commit()

    def get_task(self, task_id: int) -> dict | None:
        """Retrieve a task by ID from SQLite.

        Args:
            task_id: The task ID to retrieve

        Returns:
            Dict with task data or None if not found
        """
        message_adapter = TypeAdapter(list[AgentMessage])
        result_adapter = TypeAdapter(list[TextResult])
        with self._get_session() as session:
            record = session.query(TaskRecord).filter(TaskRecord.id == task_id).first()
            if not record:
                return None
            return {
                "id": record.id,
                "type": record.type,
                "input": record.input,
                "messages": message_adapter.validate_json(record.messages or "[]"),
                "results": result_adapter.validate_json(record.results or "[]"),
                "status": record.status,
                "created_at": record.created_at,
            }

    def list_tasks(self, limit: int = 10, type_filter: str | None = None) -> list[dict]:
        """List recent tasks from SQLite.

        Args:
            limit: Maximum number of tasks to return
            type_filter: Optional filter by task type ('explore' or 'single_run')

        Returns:
            List of dicts with task data
        """
        with self._get_session() as session:
            query = session.query(TaskRecord).order_by(TaskRecord.id.desc()).limit(limit)
            if type_filter:
                query = query.filter(TaskRecord.type == type_filter)
            records = query.all()
            return [
                {
                    "id": r.id,
                    "type": r.type,
                    "input": r.input,
                    "status": r.status,
                    "created_at": r.created_at,
                }
                for r in records
            ]
