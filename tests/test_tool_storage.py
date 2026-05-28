"""Tests for Database storage module."""

from __future__ import annotations

import os
import tempfile

import pytest
from sqlmodel import SQLModel

from simple_agent.db.db import Database
from simple_agent.state.state import ToolCallRecord, TaskRecord
from simple_agent.state.state import Task, ToolExecMessage
from pi.ai import ToolCall


class TestDatabaseInit:
    """Tests for Database._init_db() method."""

    def test_init_db_creates_tables(self):
        """_init_db() should create tables in SQLite database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            db = Database(db_path)

            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()

            assert "toolcallrecord" in tables
            assert "taskrecord" in tables
        finally:
            os.unlink(db_path)

    def test_init_db_enables_wal_mode(self):
        """_init_db() should enable WAL mode for concurrent reads."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            db = Database(db_path)

            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]
            conn.close()

            assert mode == "wal"
        finally:
            os.unlink(db_path)


class TestDatabaseToolCalls:
    """Tests for Database tool call operations."""

    def test_insert_tool_call(self):
        """insert_tool_call() should insert and return ID."""
        from pi.ai.types import TextContent
        from pi.agent import AgentToolResult

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            db = Database(db_path)

            tool_exec = ToolExecMessage(
                tool_call=ToolCall(id="call_1", arguments={"arg": "value"}, name="test_tool"),
                raw_output="test output",
                tool_result=AgentToolResult(content=[TextContent(text="test output")])
            )
            id = db.insert_tool_call(tool_exec)

            assert id == 0
            record = db.get_tool_call(0)
            assert record is not None
            assert record.tool_call.name == "test_tool"
        finally:
            os.unlink(db_path)

    def test_get_tool_calls_by_ids(self):
        """get_tool_calls_by_ids() should return records sorted by ID."""
        from pi.ai.types import TextContent
        from pi.agent import AgentToolResult

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            db = Database(db_path)

            for i in range(3):
                tool_exec = ToolExecMessage(
                    tool_call=ToolCall(id=f"call_{i}", arguments={"index": i}, name=f"tool_{i}"),
                    raw_output=f"output_{i}",
                    tool_result=AgentToolResult(content=[TextContent(text=f"output_{i}")])
                )
                db.insert_tool_call(tool_exec)

            records = db.get_tool_calls_by_ids([2, 0])

            assert len(records) == 2
            assert records[0].tool_call.name == "tool_0"
            assert records[1].tool_call.name == "tool_2"
        finally:
            os.unlink(db_path)

    def test_insert_tool_call_with_shared_session(self):
        """insert_tool_call with explicit session — caller controls commit."""
        from pi.ai.types import TextContent
        from pi.agent import AgentToolResult

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            db = Database(db_path)

            tool_exec = ToolExecMessage(
                tool_call=ToolCall(id="call_1", arguments={"arg": "value"}, name="test_tool"),
                raw_output="test output",
                tool_result=AgentToolResult(content=[TextContent(text="test output")])
            )
            with db._get_session() as s:
                id = db.insert_tool_call(tool_exec, session=s)
                s.commit()

            assert id == 0
            record = db.get_tool_call(0)
            assert record is not None
            assert record.tool_call.name == "test_tool"
        finally:
            os.unlink(db_path)


class TestDatabaseTaskOperations:
    """Tests for task upsert, get, and delete."""

    def test_upsert_and_get_task_roundtrip(self):
        """upsert_task() and get_task() round-trip."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            db = Database(db_path)

            task = Task(input="test input", type="explore", state="finished")
            task.id = db.upsert_task(task)

            row = db.get_task(task.id)
            assert row is not None
            assert row.type == "explore"
            assert row.input == "test input"
            assert row.state == "finished"
        finally:
            os.unlink(db_path)

    def test_load_all_tasks(self):
        """load_all_tasks() returns all tasks."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            db = Database(db_path)

            for i in range(5):
                task = Task(input=f"input_{i}", type="single_run", state="finished")
                db.upsert_task(task)

            rows = db.load_all_tasks()
            assert len(rows) == 5
        finally:
            os.unlink(db_path)

    def test_delete_task(self):
        """delete_task() removes a task row."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            db = Database(db_path)

            task = Task(input="test")
            task.id = db.upsert_task(task)
            assert db.get_task(task.id) is not None

            db.delete_task(task.id)
            assert db.get_task(task.id) is None
        finally:
            os.unlink(db_path)

    def test_upsert_with_shared_session(self):
        """upsert_task with explicit session — caller controls commit."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            db = Database(db_path)

            with db._get_session() as s:
                task = Task(input="batch")
                task.id = db.upsert_task(task, session=s)
                s.commit()

            row = db.get_task(task.id)
            assert row is not None
            assert row.input == "batch"
        finally:
            os.unlink(db_path)
