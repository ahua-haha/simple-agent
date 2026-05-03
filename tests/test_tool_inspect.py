"""Tests for tool-inspect CLI and ToolMgr flush."""

from __future__ import annotations

import os
import subprocess
import tempfile

import pytest

from simple_agent.tool.tool_mgr import ToolMgr
from simple_agent.state.state import ToolExecMessage
from pi.agent import AgentToolResult
from pi.ai.types import TextContent, ToolCall


class TestToolMgrFlush:
    """Tests for ToolMgr.flush() method."""

    def test_flush_writes_json_lines(self):
        """flush() should write records as JSON Lines."""
        mgr = ToolMgr()
        mgr.records.append(ToolExecMessage(
            input=ToolCall(id="1", arguments={"a": 1}, name="tool_a"),
            output=AgentToolResult(content=[TextContent(text="output_a")])
        ))

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            mgr.flush(path)
            with open(path, "r") as f:
                lines = f.readlines()

            assert len(lines) == 1
            import json
            entry = json.loads(lines[0])
            assert entry["id"] == 0
            assert entry["tool"] == "tool_a"
            assert entry["content"] == "output_a"
        finally:
            os.unlink(path)

    def test_flush_clears_records(self):
        """flush() should clear records after writing."""
        mgr = ToolMgr()
        mgr.records.append(ToolExecMessage(
            input=ToolCall(id="1", arguments={}, name="test"),
            output=AgentToolResult(content=[TextContent(text="test")])
        ))

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            mgr.flush(path)
            assert len(mgr.records) == 0
        finally:
            os.unlink(path)

    def test_flush_increments_next_id(self):
        """flush() should update _next_id counter."""
        mgr = ToolMgr()
        mgr.records.append(ToolExecMessage(
            input=ToolCall(id="1", arguments={}, name="test"),
            output=AgentToolResult(content=[TextContent(text="test")])
        ))

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            initial_id = mgr._next_id
            mgr.flush(path)
            assert mgr._next_id == initial_id + 1
        finally:
            os.unlink(path)

    def test_flush_creates_file_if_not_exists(self):
        """flush() should create file if it doesn't exist and there are records."""
        mgr = ToolMgr()
        path = "/tmp/test_flush_nonexistent.jsonl"
        if os.path.exists(path):
            os.unlink(path)

        # Add a record first
        mgr.records.append(ToolExecMessage(
            input=ToolCall(id="1", arguments={}, name="test"),
            output=AgentToolResult(content=[TextContent(text="test")])
        ))

        try:
            mgr.flush(path)
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_flush_appends_to_existing_file(self):
        """flush() should append to existing file."""
        mgr = ToolMgr()
        mgr.records.append(ToolExecMessage(
            input=ToolCall(id="1", arguments={}, name="tool1"),
            output=AgentToolResult(content=[TextContent(text="result1")])
        ))

        path = "/tmp/test_flush_append.jsonl"
        if os.path.exists(path):
            os.unlink(path)

        try:
            # First flush
            mgr.flush(path)
            # Add more records
            mgr.records.append(ToolExecMessage(
                input=ToolCall(id="2", arguments={}, name="tool2"),
                output=AgentToolResult(content=[TextContent(text="result2")])
            ))
            # Second flush
            mgr.flush(path)

            with open(path, "r") as f:
                lines = f.readlines()
            assert len(lines) == 2
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestToolInspect:
    """Tests for tool-inspect CLI."""

    def test_tool_inspect_prints_content(self):
        """tool-inspect should print content for given ID."""
        path = "/tmp/test_tool_inspect.jsonl"
        with open(path, "w") as f:
            f.write('{"id":0,"tool":"test","params":{},"content":"hello world"}\n')

        result = subprocess.run(
            ["python", "-m", "simple_agent.tool.tool_inspect", "0", "--path", path],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert result.stdout == "hello world"
        os.unlink(path)

    def test_tool_inspect_missing_id(self):
        """tool-inspect should exit 1 for missing ID."""
        path = "/tmp/test_tool_inspect_missing.jsonl"
        with open(path, "w") as f:
            f.write('{"id":0,"tool":"test","params":{},"content":"hello"}\n')

        result = subprocess.run(
            ["python", "-m", "simple_agent.tool.tool_inspect", "999", "--path", path],
            capture_output=True,
            text=True
        )

        assert result.returncode == 1
        os.unlink(path)

    def test_tool_inspect_output_pipe_friendly(self):
        """tool-inspect output should work with shell pipes."""
        path = "/tmp/test_tool_inspect_pipe.jsonl"
        with open(path, "w") as f:
            f.write('{"id":0,"tool":"test","params":{},"content":"error: file not found\\nerror: permission denied"}\n')

        result = subprocess.run(
            ["python", "-m", "simple_agent.tool.tool_inspect", "0", "--path", path],
            capture_output=True,
            text=True
        )

        # grep should find the error
        assert "error" in result.stdout
        os.unlink(path)