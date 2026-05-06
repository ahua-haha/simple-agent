"""Tests for Database storage module."""

from __future__ import annotations

import os
import tempfile

import pytest
from sqlmodel import SQLModel

from simple_agent.db.db import Database, ToolCallRecord, TaskRecord
from simple_agent.state.state import ToolExecMessage
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


class TestDatabaseGetAllMessages:
    """Tests for Database.get_all_messages() method."""

    def test_get_all_messages_queries_sqlite(self):
        """get_all_messages() should return AgentMessage list from tool call records."""
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

            messages = db.get_all_messages([0, 2])

            assert len(messages) == 4
        finally:
            os.unlink(db_path)


class TestDatabaseTaskHistory:
    """Tests for task history storage methods."""

    def test_save_task_and_get_task_roundtrip(self):
        """save_task() and get_task() should provide round-trip storage."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            db = Database(db_path)

            db.save_task(
                task_type="explore",
                task_input="test input",
                messages=[],
                results=[],
                status="finished",
            )

            task = db.get_task(1)

            assert task is not None
            assert task["type"] == "explore"
            assert task["input"] == "test input"
            assert task["status"] == "finished"
        finally:
            os.unlink(db_path)

    def test_list_tasks_returns_recent_tasks(self):
        """list_tasks() should return recent tasks ordered by ID desc."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            db = Database(db_path)

            for i in range(5):
                db.save_task(
                    task_type="single_run",
                    task_input=f"input_{i}",
                    messages=[],
                    results=[],
                    status="finished",
                )

            tasks = db.list_tasks(limit=3)

            assert len(tasks) == 3
            assert tasks[0]["input"] == "input_4"
        finally:
            os.unlink(db_path)
