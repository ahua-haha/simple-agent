"""Tests for SessionRunner tool execution wrapping."""

from __future__ import annotations

import asyncio

import pytest

from pi.agent import AgentTool, AgentToolResult
from pi.ai.types import TextContent

from simple_agent.db.db import Database
from simple_agent.session.runner import SessionRunner
from simple_agent.task_manager import TaskManager


class _FakeAgentProcess:
    def subscribe(self, callback):
        pass

    def unsubscribe(self, callback):
        pass


def _make_runner(db: Database) -> SessionRunner:
    manager = TaskManager(db)
    manager.load(None)
    manager.create_user_task("Build feature")
    return SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=manager,
        agent_process=_FakeAgentProcess(),
        cancel_event=asyncio.Event(),
    )


@pytest.mark.asyncio
async def test_wrap_tool_records_runner_tool_call_success(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    tool = AgentTool(name="example", description="Example", parameters={"type": "object", "properties": {}})

    async def execute(tool_call_id, params, cancel_event=None, on_update=None):
        return AgentToolResult(content=[TextContent(text="hello")])

    tool.execute = execute
    runner = _make_runner(db)
    wrapped = runner.wrap_tool(tool)

    await wrapped.execute("call_1", {"name": "Ada"})

    records = db.list_runner_tool_calls("session_a")
    assert len(records) == 1
    assert records[0].tool_call_id == "call_1"
    assert records[0].tool_name == "example"
    assert records[0].params_json == '{"name": "Ada"}'
    assert records[0].status == "success"
    assert records[0].error is None


@pytest.mark.asyncio
async def test_wrap_tool_records_runner_tool_call_error_and_reraises(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    tool = AgentTool(name="explode", description="Explode", parameters={"type": "object", "properties": {}})

    async def execute(tool_call_id, params, cancel_event=None, on_update=None):
        raise RuntimeError("boom")

    tool.execute = execute
    runner = _make_runner(db)
    wrapped = runner.wrap_tool(tool)

    with pytest.raises(RuntimeError, match="boom"):
        await wrapped.execute("call_2", {"x": 1})

    records = db.list_runner_tool_calls("session_a")
    assert len(records) == 1
    assert records[0].tool_call_id == "call_2"
    assert records[0].tool_name == "explode"
    assert records[0].status == "error"
    assert records[0].error == "boom"
