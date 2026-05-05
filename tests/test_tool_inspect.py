"""Tests for tool-inspect CLI with SQLite."""

from __future__ import annotations

import os
import subprocess
import tempfile

import pytest
from sqlmodel import Session, create_engine

from simple_agent.tool.db import ToolCallRecord


class TestToolInspect:
    """Tests for tool-inspect CLI with SQLite backend."""

    def test_tool_inspect_prints_content(self):
        """tool-inspect should print content for given ID."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            from sqlmodel import SQLModel
            engine = create_engine(f"sqlite:///{db_path}")
            SQLModel.metadata.create_all(engine)

            with Session(engine) as session:
                record = ToolCallRecord(
                    id=0,
                    tool="test",
                    params="{}",
                    content="hello world",
                )
                session.add(record)
                session.commit()

            result = subprocess.run(
                ["python", "-m", "simple_agent.tool.tool_inspect", "0", "--path", db_path],
                capture_output=True,
                text=True
            )

            assert result.returncode == 0
            assert result.stdout == "hello world"
        finally:
            os.unlink(db_path)

    def test_tool_inspect_missing_id(self):
        """tool-inspect should exit 1 for missing ID."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            from sqlmodel import SQLModel
            engine = create_engine(f"sqlite:///{db_path}")
            SQLModel.metadata.create_all(engine)

            result = subprocess.run(
                ["python", "-m", "simple_agent.tool.tool_inspect", "999", "--path", db_path],
                capture_output=True,
                text=True
            )

            assert result.returncode == 1
        finally:
            os.unlink(db_path)

    def test_tool_inspect_list_recent(self):
        """tool-inspect --list should show recent tool calls."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            from sqlmodel import SQLModel
            engine = create_engine(f"sqlite:///{db_path}")
            SQLModel.metadata.create_all(engine)

            with Session(engine) as session:
                for i in range(3):
                    record = ToolCallRecord(
                        id=i,
                        tool=f"tool_{i}",
                        params="{}",
                        content=f"result_{i}",
                    )
                    session.add(record)
                session.commit()

            result = subprocess.run(
                ["python", "-m", "simple_agent.tool.tool_inspect", "--list", "--limit", "2", "--path", db_path],
                capture_output=True,
                text=True
            )

            assert result.returncode == 0
            assert "tool_2" in result.stdout
            assert "tool_1" in result.stdout
        finally:
            os.unlink(db_path)
