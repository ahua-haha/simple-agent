"""Database storage and query module for tool calls and tasks."""

from __future__ import annotations

import time
from functools import wraps

from sqlmodel import SQLModel, Session, create_engine, select, delete
import sqlite3

from pi.agent.types import AgentMessage

from simple_agent.state.state import (
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

    def create_session(self) -> Session:
        """Create a database session for composing multiple writes."""
        return self._get_session()

    # ------------------------------------------------------------------
    # ManagedTask operations
    # ------------------------------------------------------------------

    @standalone_or_compose
    def upsert_managed_task(self, task, *, session: Session | None = None) -> int:
        """Insert or update a replacement task-manager row."""
        if task.id is None:
            task.id = self.next_managed_task_id(session=session)
        record = session.merge(managed_task_to_record(task))
        task.id = record.id
        return record.id

    @standalone_or_compose
    def get_managed_task(self, task_id: int, *, session: Session | None = None):
        """Return a managed task by ID, or None."""
        record = session.get(TaskRecord, task_id)
        if record is None:
            return None
        session.expunge(record)
        return managed_task_from_record(record)

    @standalone_or_compose
    def list_managed_tasks(self, *, session: Session | None = None):
        """Return all managed tasks ordered by ID."""
        records = list(session.exec(select(TaskRecord).order_by(TaskRecord.id)).all())
        for record in records:
            session.expunge(record)
        return [managed_task_from_record(record) for record in records]

    @standalone_or_compose
    def list_managed_task_children(self, parent_id: int, *, session: Session | None = None):
        """Return direct managed-task children in append order."""
        records = list(
            session.exec(
                select(TaskRecord)
                .where(TaskRecord.parent_id == parent_id)
                .order_by(TaskRecord.id)
            ).all()
        )
        for record in records:
            session.expunge(record)
        return [managed_task_from_record(record) for record in records]

    @standalone_or_compose
    def next_managed_task_id(self, *, session: Session | None = None) -> int:
        record = session.exec(select(TaskRecord).order_by(TaskRecord.id.desc())).first()
        return (record.id + 1) if record and record.id is not None else 1

    @standalone_or_compose
    def delete_managed_tasks(self, task_ids: list[int], *, session: Session | None = None) -> None:
        for task_id in task_ids:
            record = session.get(TaskRecord, task_id)
            if record is not None:
                session.delete(record)

    @standalone_or_compose
    def replace_managed_task_tree(self, root_task, *, session: Session | None = None) -> None:
        if root_task.id is None:
            raise ValueError("Cannot replace a managed task tree without a root task id")
        session.exec(
            delete(TaskRecord)
            .where(TaskRecord.id > root_task.id)
        )
        for task in _flatten_managed_task_tree(root_task):
            self.upsert_managed_task(task, session=session)

    # ------------------------------------------------------------------
    # Runner state operations
    # ------------------------------------------------------------------

    @standalone_or_compose
    def upsert_runner_state_metadata(
        self,
        session_id: str,
        *,
        next_action: str,
        active_user_task_id: int | None = None,
        last_error: str | None = None,
        session: Session | None = None,
    ) -> None:
        record = session.get(RunnerStateMetadataRecord, session_id)
        now = time.time()
        if record is None:
            record = RunnerStateMetadataRecord(session_id=session_id, created_at=now)
            session.add(record)
        record.next_action = next_action
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
    def insert_runner_message(
        self,
        session_id: str,
        message: AgentMessage,
        *,
        id: int | None = None,
        session: Session | None = None,
    ) -> int:
        stable_id = id if id is not None else self.next_runner_message_id(session=session)
        record = RunnerMessageRecord(
            id=stable_id,
            session_id=session_id,
            role=message.role,
            content_json=agent_message_to_json(message),
            timestamp_ms=getattr(message, "timestamp", None),
        )
        session.add(record)
        return stable_id

    @standalone_or_compose
    def next_runner_message_id(
        self,
        *,
        session: Session | None = None,
    ) -> int:
        record = session.exec(select(RunnerMessageRecord).order_by(RunnerMessageRecord.id.desc())).first()
        return (record.id + 1) if record and record.id is not None else 1

    @standalone_or_compose
    def replace_runner_messages(
        self,
        session_id: str,
        messages: list[AgentMessage],
        *,
        ids: list[int] | None = None,
        session: Session | None = None,
    ) -> None:
        if ids is not None and len(ids) != len(messages):
            raise ValueError("Message ids must match messages length")
        session.exec(
            delete(RunnerMessageRecord)
            .where(RunnerMessageRecord.session_id == session_id)
        )
        for index, message in enumerate(messages):
            self.insert_runner_message(
                session_id,
                message,
                id=ids[index] if ids is not None else None,
                session=session,
            )

    @standalone_or_compose
    def delete_runner_message_seq_range(
        self,
        session_id: str,
        *,
        start_seq: int,
        end_seq: int,
        session: Session | None = None,
    ) -> None:
        if end_seq < start_seq:
            raise ValueError("End message seq is before start message seq")
        session.exec(
            delete(RunnerMessageRecord)
            .where(RunnerMessageRecord.session_id == session_id)
            .where(RunnerMessageRecord.seq >= start_seq)
            .where(RunnerMessageRecord.seq <= end_seq)
        )

    @standalone_or_compose
    def get_runner_message_seq(
        self,
        session_id: str,
        message_id: int,
        *,
        session: Session | None = None,
    ) -> int:
        record = session.exec(
            select(RunnerMessageRecord)
            .where(RunnerMessageRecord.session_id == session_id)
            .where(RunnerMessageRecord.id == message_id)
        ).first()
        if record is None:
            raise ValueError(f"Runner message id does not exist: {message_id}")
        if record.seq is None:
            raise ValueError(f"Runner message id has no sequence: {message_id}")
        return record.seq

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
    def list_runner_message_entries(
        self,
        session_id: str,
        *,
        session: Session | None = None,
    ) -> list[tuple[int, AgentMessage]]:
        records = list(
            session.exec(
                select(RunnerMessageRecord)
                .where(RunnerMessageRecord.session_id == session_id)
                .order_by(RunnerMessageRecord.seq)
            ).all()
        )
        entries: list[tuple[int, AgentMessage]] = []
        for record in records:
            if record.id is None:
                raise RuntimeError("Runner message record is missing stable id")
            entries.append((record.id, agent_message_from_json(record.content_json)))
        return entries

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
        id: int | None = None,
        session_id: str,
        tool_call_id: str,
        tool_name: str,
        tool_call_json: str,
        tool_result_json: str,
        session: Session | None = None,
    ) -> int:
        next_id = id if id is not None else self.next_runner_tool_call_id(session_id, session=session)
        record = RunnerToolCallRecord(
            id=next_id,
            session_id=session_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_call_json=tool_call_json,
            tool_result_json=tool_result_json,
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


def _flatten_managed_task_tree(task) -> list:
    tasks = []
    stack = [task]
    while stack:
        current = stack.pop()
        tasks.append(current)
        stack.extend(reversed(current.children))
    return tasks
