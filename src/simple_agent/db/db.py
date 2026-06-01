"""Database storage and query module for tool calls and tasks."""

from __future__ import annotations

import json
import time
from functools import wraps

from sqlmodel import SQLModel, Session, create_engine, select
import sqlite3

from pi.agent.types import AgentMessage

from simple_agent.state.state import (
    ManagedTaskRecord,
    RunnerMessageRecord,
    RunnerStateMetadataRecord,
    RunnerToolCallRecord,
    SessionRecord,
    TaskRecord,
    agent_message_from_json,
    agent_message_to_json,
    managed_task_from_record,
    managed_task_to_record,
)


def standalone_or_compose(func):
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

    def __init__(self, db_path: str):
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
    # Task operations
    # ------------------------------------------------------------------

    @standalone_or_compose
    def upsert_task(self, task, *, session: Session | None = None) -> int:
        """INSERT or UPDATE a task row.  Returns the task ``id``."""
        record = session.merge(task.to_db_row())
        session.flush()
        task.id = record.id
        return record.id

    @standalone_or_compose
    def get_task(self, task_id: int, *, session: Session | None = None) -> TaskRecord | None:
        """Return a single task record, or None."""
        record = session.get(TaskRecord, task_id)
        if record is not None:
            session.expunge(record)
        return record

    @standalone_or_compose
    def load_all_tasks(self, *, session: Session | None = None) -> list[TaskRecord]:
        """Return all task records, ordered by id."""
        records = list(session.exec(select(TaskRecord).order_by(TaskRecord.id)).all())
        for r in records:
            session.expunge(r)
        return records

    @standalone_or_compose
    def delete_task(self, task_id: int, *, session: Session | None = None) -> None:
        """Delete a task row by id."""
        record = session.get(TaskRecord, task_id)
        if record is not None:
            session.delete(record)

    # ------------------------------------------------------------------
    # ManagedTask operations
    # ------------------------------------------------------------------

    @standalone_or_compose
    def upsert_managed_task(self, task, *, session: Session | None = None) -> int:
        """Insert or update a replacement task-manager row."""
        record = session.merge(managed_task_to_record(task))
        session.flush()
        task.id = record.id
        return record.id

    @standalone_or_compose
    def get_managed_task(self, task_id: int, *, session: Session | None = None):
        """Return a managed task by ID, or None."""
        record = session.get(ManagedTaskRecord, task_id)
        if record is None:
            return None
        session.expunge(record)
        return managed_task_from_record(record)

    @standalone_or_compose
    def list_managed_tasks(self, *, session: Session | None = None):
        """Return all managed tasks ordered by ID."""
        records = list(session.exec(select(ManagedTaskRecord).order_by(ManagedTaskRecord.id)).all())
        for record in records:
            session.expunge(record)
        return [managed_task_from_record(record) for record in records]

    # ------------------------------------------------------------------
    # Runner state operations
    # ------------------------------------------------------------------

    @standalone_or_compose
    def upsert_runner_state_metadata(
        self,
        session_id: str,
        *,
        phase: str,
        status: str,
        active_user_task_id: int | None = None,
        last_error: str | None = None,
        session: Session | None = None,
    ) -> None:
        record = session.get(RunnerStateMetadataRecord, session_id)
        now = time.time()
        if record is None:
            record = RunnerStateMetadataRecord(session_id=session_id, created_at=now)
            session.add(record)
        record.phase = phase
        record.status = status
        record.active_user_task_id = active_user_task_id
        record.last_error = last_error
        record.updated_at = now

    @standalone_or_compose
    def get_runner_state_metadata(
        self,
        session_id: str,
        *,
        session: Session | None = None,
    ) -> RunnerStateMetadataRecord | None:
        record = session.get(RunnerStateMetadataRecord, session_id)
        if record is not None:
            session.expunge(record)
        return record

    @standalone_or_compose
    def next_runner_message_seq(
        self,
        session_id: str,
        *,
        session: Session | None = None,
    ) -> int:
        record = session.exec(
            select(RunnerMessageRecord)
            .where(RunnerMessageRecord.session_id == session_id)
            .order_by(RunnerMessageRecord.seq.desc())
        ).first()
        return (record.seq + 1) if record else 0

    @standalone_or_compose
    def append_runner_messages(
        self,
        session_id: str,
        messages: list[AgentMessage],
        *,
        session: Session | None = None,
    ) -> None:
        seq = self.next_runner_message_seq(session_id, session=session)
        for message in messages:
            record = RunnerMessageRecord(
                session_id=session_id,
                seq=seq,
                role=message.role,
                content_json=agent_message_to_json(message),
                timestamp_ms=getattr(message, "timestamp", None),
            )
            session.add(record)
            seq += 1

    @standalone_or_compose
    def list_runner_messages(
        self,
        session_id: str,
        *,
        session: Session | None = None,
    ) -> list[AgentMessage]:
        records = list(
            session.exec(
                select(RunnerMessageRecord)
                .where(RunnerMessageRecord.session_id == session_id)
                .order_by(RunnerMessageRecord.seq)
            ).all()
        )
        return [agent_message_from_json(record.content_json) for record in records]

    @standalone_or_compose
    def next_runner_tool_call_id(
        self,
        session_id: str,
        *,
        session: Session | None = None,
    ) -> int:
        record = session.exec(
            select(RunnerToolCallRecord)
            .where(RunnerToolCallRecord.session_id == session_id)
            .order_by(RunnerToolCallRecord.id.desc())
        ).first()
        return (record.id + 1) if record else 0

    @standalone_or_compose
    def insert_runner_tool_call(
        self,
        *,
        session_id: str,
        tool_call_id: str,
        tool_name: str,
        params: dict,
        result: dict | None,
        status: str,
        started_at: float,
        finished_at: float | None,
        error: str | None,
        session: Session | None = None,
    ) -> int:
        next_id = self.next_runner_tool_call_id(session_id, session=session)
        record = RunnerToolCallRecord(
            id=next_id,
            session_id=session_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            params_json=json.dumps(params, sort_keys=True),
            result_json=json.dumps(result, sort_keys=True) if result is not None else None,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            error=error,
        )
        session.add(record)
        return next_id

    @standalone_or_compose
    def list_runner_tool_calls(
        self,
        session_id: str,
        *,
        session: Session | None = None,
    ) -> list[RunnerToolCallRecord]:
        records = list(
            session.exec(
                select(RunnerToolCallRecord)
                .where(RunnerToolCallRecord.session_id == session_id)
                .order_by(RunnerToolCallRecord.id)
            ).all()
        )
        for record in records:
            session.expunge(record)
        return records

    # ------------------------------------------------------------------
    # Session metadata operations
    # ------------------------------------------------------------------

    @standalone_or_compose
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

    @standalone_or_compose
    def get_session(self, session_id: str, *,
                    session: Session | None = None) -> dict | None:
        """Return session metadata by ID, or None."""
        record = session.get(SessionRecord, session_id)
        if record is None:
            return None
        return {
            "id": record.id,
            "name": record.name,
            "cursor_id": record.cursor_id,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }

    @standalone_or_compose
    def delete_session(self, session_id: str, *,
                       session: Session | None = None) -> None:
        """Delete a session metadata row."""
        record = session.get(SessionRecord, session_id)
        if record is not None:
            session.delete(record)
