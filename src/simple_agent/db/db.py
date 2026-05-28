"""Database storage and query module for tool calls and tasks."""

from __future__ import annotations

import json
import time
from functools import wraps

from sqlmodel import SQLModel, Session, create_engine, select
import sqlite3

from simple_agent.state.state import (
    SessionRecord,
    TaskRecord,
    ToolCallRecord,
    ToolExecMessage,
)


def provide_session(func):
    """Decorate a Database write method to inject a session if not provided.

    - ``session=None`` (default): creates a session, commits, closes.
    - ``session=s``: uses *s* — caller controls commit.
    """
    @wraps(func)
    def wrapper(self: Database, *args, **kwargs):
        session = kwargs.get("session")
        if session is not None:
            return func(self, *args, **kwargs)

        with self._get_session() as s:
            kwargs["session"] = s
            result = func(self, *args, **kwargs)
            s.commit()
            return result

    return wrapper


class Database:
    """SQLite database for tool call storage and task history.

    Write methods accept an optional *session* parameter:

    - ``session=None``: opens a new session, commits, and closes.
    - ``session=s``: uses *s* without committing — the caller controls
      the transaction boundary.
    """

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


    # ------------------------------------------------------------------
    # ToolCall operations
    # ------------------------------------------------------------------

    def next_tool_call_id(self) -> int:
        with self._get_session() as s:
            max_record = s.exec(select(ToolCallRecord).order_by(ToolCallRecord.id.desc())).first()
            return (max_record.id + 1) if max_record else 0

    @provide_session
    def insert_tool_call(self, tool_exec: ToolExecMessage, *,
                         session: Session | None = None) -> int:
        """Insert a tool call record and return its ID."""
        next_id = self.next_tool_call_id()
        record = ToolCallRecord(
            id=next_id,
            tool=tool_exec.tool_call.name,
            content=json.dumps(tool_exec.model_dump()),
        )
        session.add(record)
        return next_id

    def get_tool_call(self, id: int) -> ToolExecMessage | None:
        """Get a tool call record by ID."""
        with self._get_session() as s:
            record = s.exec(select(ToolCallRecord).where(ToolCallRecord.id == id)).first()
            if not record:
                return None
            return ToolExecMessage.model_validate_json(record.content)

    def get_tool_calls_by_ids(self, ids: list[int]) -> list[ToolExecMessage]:
        """Get multiple tool call records by IDs, sorted by ID."""
        if not ids:
            return []
        with self._get_session() as s:
            records = s.exec(select(ToolCallRecord).where(ToolCallRecord.id.in_(ids))).all()
            records.sort(key=lambda r: r.id)
            return [ToolExecMessage.model_validate_json(r.content) for r in records]

    def list_tool_calls(self, limit: int = 10) -> list[ToolCallRecord]:
        """List recent tool call records."""
        with self._get_session() as s:
            records = s.exec(select(ToolCallRecord).order_by(ToolCallRecord.id.desc()).limit(limit)).all()
            return list(records)

    # ------------------------------------------------------------------
    # Task operations
    # ------------------------------------------------------------------

    @provide_session
    def upsert_task(self, task, *, session: Session | None = None) -> int:
        """INSERT or UPDATE a task row.  Returns the task ``id``."""
        record = session.merge(task.to_db_row())
        session.flush()
        task.id = record.id
        return record.id

    def get_task(self, task_id: int) -> TaskRecord | None:
        """Return a single task record, or None."""
        with self._get_session() as s:
            return s.get(TaskRecord, task_id)

    def load_all_tasks(self) -> list[TaskRecord]:
        """Return all task records, ordered by id."""
        with self._get_session() as s:
            return list(s.exec(select(TaskRecord).order_by(TaskRecord.id)).all())

    @provide_session
    def delete_task(self, task_id: int, *, session: Session | None = None) -> None:
        """Delete a task row by id."""
        record = session.get(TaskRecord, task_id)
        if record is not None:
            session.delete(record)

    # ------------------------------------------------------------------
    # Session metadata operations
    # ------------------------------------------------------------------

    @provide_session
    def upsert_session(self, session_id: str, name: str = "",
                       cursor_id: int | None = None, *,
                       session: Session | None = None) -> None:
        """Insert or update session metadata."""
        record = session.get(SessionRecord, session_id)
        if record is None:
            record = SessionRecord(id=session_id)
            session.add(record)
        record.name = name
        record.cursor_id = cursor_id
        record.updated_at = time.time()

    def get_session(self, session_id: str) -> dict | None:
        """Return session metadata by ID, or None."""
        with self._get_session() as s:
            record = s.get(SessionRecord, session_id)
            if record is None:
                return None
            return {
                "id": record.id,
                "name": record.name,
                "cursor_id": record.cursor_id,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
            }

    @provide_session
    def delete_session(self, session_id: str, *,
                       session: Session | None = None) -> None:
        """Delete a session metadata row."""
        record = session.get(SessionRecord, session_id)
        if record is not None:
            session.delete(record)

    # ------------------------------------------------------------------
    # atomic checkpoint
    # ------------------------------------------------------------------

    def checkpoint(
        self,
        session_id: str,
        cursor_id: int | None,
        updates: list | None = None,
        inserts: list | None = None,
    ) -> None:
        """Atomically persist task changes and session metadata."""
        with self._get_session() as s:
            all_tasks: list = []
            if updates:
                all_tasks.extend(updates)
            if inserts:
                all_tasks.extend(inserts)

            for task in all_tasks:
                self.upsert_task(task, session=s)

            self.upsert_session(session_id, "", cursor_id, session=s)
            s.commit()
