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


class TestSessionSave:
    """Tests for session metadata save/load."""

    def test_save_persists_to_db(self, tmp_path):
        session = Session(base_dir=str(tmp_path))
        session._cursor_id = 7
        session.save()

        data = session._db.get_session(session.id)
        assert data is not None
        assert data["cursor_id"] == 7
        assert "created_at" in data
        assert "updated_at" in data

    def test_save_includes_all_metadata(self, tmp_path):
        session = Session(base_dir=str(tmp_path))
        session._cursor_id = 3
        session.save()

        data = session._db.get_session(session.id)
        assert data["cursor_id"] == 3
        assert "created_at" in data
        assert "updated_at" in data

    def test_load_restores_metadata(self, tmp_path):
        session = Session(base_dir=str(tmp_path))
        session._cursor_id = 42
        session.save()

        session2 = Session(session_id=session.id, base_dir=str(tmp_path))
        assert session2._cursor_id == 42

    def test_new_session_has_timestamps(self, tmp_path):
        session = Session(base_dir=str(tmp_path))
        assert session._created_at is not None
        assert session._updated_at is not None
        assert session._created_at == session._updated_at

    def test_save_updates_updated_at(self, tmp_path):
        import time
        session = Session(base_dir=str(tmp_path))
        original = session._updated_at
        time.sleep(0.01)
        session.save()
        assert session._updated_at > original


class TestSessionListSessions:
    """Tests for list_sessions."""

    def test_empty_directory(self, tmp_path):
        sessions = Session.list_sessions(str(tmp_path))
        assert sessions == []

    def test_lists_db_files(self, tmp_path):
        for name in ["a", "b"]:
            from simple_agent.db.db import Database
            db = Database(os.path.join(str(tmp_path), f"{name}.db"))
            root = Task(input="x")
            db.upsert_task(root)

        sessions = Session.list_sessions(str(tmp_path))
        assert sessions == ["a", "b"]

    def test_ignores_non_db(self, tmp_path):
        from simple_agent.db.db import Database
        db = Database(os.path.join(str(tmp_path), "test.db"))
        root = Task(input="x")
        db.upsert_task(root)
        with open(os.path.join(str(tmp_path), "notes.txt"), "w") as f:
            f.write("hello")

        sessions = Session.list_sessions(str(tmp_path))
        assert sessions == ["test"]
