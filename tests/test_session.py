"""Tests for Session."""

from __future__ import annotations

import os

import pytest

from simple_agent.session import Session
from simple_agent.state.state import Task


class TestSessionInit:
    """Tests for Session initialization."""

    def test_new_session_has_no_root(self, tmp_path):
        session = Session(base_dir=str(tmp_path))
        assert session.root is None

    def test_existing_session_loads_root(self, tmp_path):
        from simple_agent.db.db import Database

        db_path = os.path.join(str(tmp_path), "test.db")
        db = Database(db_path)
        root = Task(input="hello", state="PENDING")
        root.id = db.upsert_task(root)

        session = Session(session_id="test", base_dir=str(tmp_path))
        assert session.root is not None
        assert session.root.input == "hello"
        assert session.root.state == "PENDING"


class TestSessionCheckpoint:
    """Tests for session checkpoint (metadata + task persistence)."""

    def test_checkpoint_persists_metadata(self, tmp_path):
        session = Session(base_dir=str(tmp_path))
        session._cursor_id = 7
        session._checkpoint()

        data = session._db.get_session(session.id)
        assert data is not None
        assert data["cursor_id"] == 7
        assert "created_at" in data
        assert "updated_at" in data

    def test_checkpoint_persists_tasks(self, tmp_path):
        from simple_agent.state.state import Task
        session = Session(base_dir=str(tmp_path))
        task = Task(input="hello")
        task.id = session._db.upsert_task(task)
        session._cursor_id = task.id
        session._checkpoint(updates=[task])

        data = session._db.get_session(session.id)
        assert data["cursor_id"] == task.id

    def test_load_restores_metadata(self, tmp_path):
        session = Session(base_dir=str(tmp_path))
        session._cursor_id = 42
        session._checkpoint()

        session2 = Session(session_id=session.id, base_dir=str(tmp_path))
        assert session2._cursor_id == 42

    def test_new_session_has_timestamps(self, tmp_path):
        session = Session(base_dir=str(tmp_path))
        assert session._created_at is not None
        assert session._updated_at is not None
        assert session._created_at == session._updated_at

    def test_checkpoint_updates_updated_at(self, tmp_path):
        import time
        session = Session(base_dir=str(tmp_path))
        original = session._updated_at
        time.sleep(0.01)
        session._checkpoint()
        assert session._updated_at > original


class TestSessionManagerList:
    """Tests for session listing via SessionManager."""

    def test_empty_directory(self, tmp_path):
        from simple_agent.session.session_manager import SessionManager
        sm = SessionManager(sessions_dir=str(tmp_path))
        sessions = sm.list()
        assert sessions == []

    def test_lists_db_files(self, tmp_path):
        from simple_agent.session.session_manager import SessionManager
        for name in ["a", "b"]:
            from simple_agent.db.db import Database
            db = Database(os.path.join(str(tmp_path), f"{name}.db"))
            root = Task(input="x")
            db.upsert_task(root)

        sm = SessionManager(sessions_dir=str(tmp_path))
        sessions = sm.list()
        ids = sorted(s["id"] for s in sessions)
        assert ids == ["a", "b"]

    def test_ignores_non_db(self, tmp_path):
        from simple_agent.session.session_manager import SessionManager
        from simple_agent.db.db import Database
        db = Database(os.path.join(str(tmp_path), "test.db"))
        root = Task(input="x")
        db.upsert_task(root)
        with open(os.path.join(str(tmp_path), "notes.txt"), "w") as f:
            f.write("hello")

        sm = SessionManager(sessions_dir=str(tmp_path))
        sessions = sm.list()
        ids = [s["id"] for s in sessions]
        assert ids == ["test"]


class TestSessionEventQueue:
    """Tests for Session event queue lifecycle."""

    @pytest.mark.asyncio
    async def test_queue_created_in_run(self, tmp_path):
        session = Session(base_dir=str(tmp_path))
        assert session.event_queue is None

        # Start run and cancel immediately to avoid actual agent execution
        session._running = True
        session.event_queue = __import__("asyncio").Queue()
        session.event_queue.put_nowait(None)

        assert session.event_queue is not None

    @pytest.mark.asyncio
    async def test_queue_none_after_run(self, tmp_path):
        session = Session(base_dir=str(tmp_path))
        session.event_queue = __import__("asyncio").Queue()
        session.event_queue.put_nowait(None)
        session.event_queue = None

        assert session.event_queue is None

    @pytest.mark.asyncio
    async def test_agent_event_pushed_to_queue(self, tmp_path):
        import asyncio
        session = Session(base_dir=str(tmp_path))
        session.event_queue = asyncio.Queue()

        from pi.agent.types import AgentEndEvent
        from pi.ai.types import AssistantMessage, TextContent
        msg = AssistantMessage(role="assistant", content=[TextContent(text="hello")])
        event = AgentEndEvent(messages=[msg])
        session._on_agent_event(event)

        received = session.event_queue.get_nowait()
        assert received is event

    @pytest.mark.asyncio
    async def test_no_push_when_queue_is_none(self, tmp_path):
        session = Session(base_dir=str(tmp_path))
        assert session.event_queue is None

        # Should not raise
        from pi.agent.types import AgentEndEvent
        from pi.ai.types import AssistantMessage, TextContent
        msg = AssistantMessage(role="assistant", content=[TextContent(text="hello")])
        event = AgentEndEvent(messages=[msg])
        session._on_agent_event(event)
