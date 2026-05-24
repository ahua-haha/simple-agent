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
        root = Task(input="hello", state="PENDING")
        filepath = os.path.join(str(tmp_path), "test.json")
        os.makedirs(str(tmp_path), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(root.to_checkpoint())

        session = Session("test", base_dir=str(tmp_path))
        assert session.root is not None
        assert session.root.input == "hello"
        assert session.root.state == "PENDING"


class TestSessionCheckpoint:
    """Tests for checkpoint behavior."""

    def test_checkpoint_writes_file(self, tmp_path):
        session = Session("test", base_dir=str(tmp_path))
        session._root = Task(input="hello", state="RUNNING")
        filepath = session.checkpoint()
        assert os.path.exists(filepath)

    def test_checkpoint_loads_back(self, tmp_path):
        session = Session("test", base_dir=str(tmp_path))
        session._root = Task(input="hello", state="RUNNING")
        session.checkpoint()

        session2 = Session("test", base_dir=str(tmp_path))
        assert session2.root is not None
        assert session2.root.input == "hello"


class TestSessionListSessions:
    """Tests for list_sessions."""

    def test_empty_directory(self, tmp_path):
        sessions = Session.list_sessions(str(tmp_path))
        assert sessions == []

    def test_lists_json_files(self, tmp_path):
        for name in ["a", "b"]:
            filepath = os.path.join(str(tmp_path), f"{name}.json")
            Task(input="x", state="FINISHED").to_checkpoint()
            with open(filepath, "w") as f:
                f.write(Task(input="x", state="FINISHED").to_checkpoint())

        sessions = Session.list_sessions(str(tmp_path))
        assert sessions == ["a", "b"]

    def test_ignores_non_json(self, tmp_path):
        filepath = os.path.join(str(tmp_path), "test.json")
        with open(filepath, "w") as f:
            f.write(Task(input="x").to_checkpoint())
        with open(os.path.join(str(tmp_path), "notes.txt"), "w") as f:
            f.write("hello")

        sessions = Session.list_sessions(str(tmp_path))
        assert sessions == ["test"]
