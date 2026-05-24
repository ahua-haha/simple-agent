"""Tests for Session."""

from __future__ import annotations

import os

import pytest

from simple_agent.session import Session
from simple_agent.state.state import Task


class TestSessionInit:
    """Tests for Session initialization."""

    def test_new_session_has_no_root(self, tmp_path):
        session = Session("test", base_dir=str(tmp_path))
        assert session.root is None

    def test_existing_session_loads_root(self, tmp_path):
        from simple_agent.db.db import Database

        db_path = os.path.join(str(tmp_path), "test.db")
        db = Database(db_path)
        root = Task(input="hello", state="PENDING")
        root.id = db.upsert_task(root)

        session = Session("test", base_dir=str(tmp_path))
        assert session.root is not None
        assert session.root.input == "hello"
        assert session.root.state == "PENDING"


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
