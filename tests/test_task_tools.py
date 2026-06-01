"""Tests for task-manager-backed tools."""

from __future__ import annotations

import asyncio
import tempfile

from simple_agent.db.db import Database
from simple_agent.task_manager import TaskManager
from simple_agent.tool.tool_mgr import ToolMgr


def _make_db() -> Database:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        return Database(f.name)


def test_create_todo_tool_creates_active_todo():
    db = _make_db()
    manager = TaskManager(db)
    manager.create_user_task("Build feature")
    tools = ToolMgr(db, task_manager=manager)
    tool = tools.create_create_todo_tool()

    async def run():
        return await tool.execute("call_1", {"title": "Inspect files"})

    result = asyncio.run(run())

    assert "created todo" in result.content[0].text.lower()
    assert manager.active_todo_id is not None


def test_finish_todo_tool_finishes_active_todo():
    db = _make_db()
    manager = TaskManager(db)
    manager.create_user_task("Build feature")
    todo = manager.create_todo("Inspect files")
    tools = ToolMgr(db, task_manager=manager)
    tool = tools.create_finish_todo_tool()

    async def run():
        return await tool.execute("call_1", {"result": "Inspected files"})

    asyncio.run(run())
    loaded = db.get_managed_task(todo.id)

    assert loaded.status == "done"
    assert loaded.result == "Inspected files"
    assert manager.active_todo_id is None


def test_normal_tool_call_records_under_active_todo():
    db = _make_db()
    manager = TaskManager(db)
    manager.create_user_task("Build feature")
    todo = manager.create_todo("Inspect files")
    tools = ToolMgr(db, task_manager=manager)

    from pi.agent import AgentTool, AgentToolResult
    from pi.ai.types import TextContent

    async def execute(tool_call_id, params, cancel_event=None, on_update=None):
        return AgentToolResult(content=[TextContent(text="raw output")])

    tool = AgentTool(
        name="sample",
        description="Sample",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )
    wrapped = tools.wrap_tools(tool)

    async def run():
        return await wrapped.execute("call_1", {})

    asyncio.run(run())
    loaded_todo = db.get_managed_task(todo.id)

    assert len(loaded_todo.items) == 1
    assert loaded_todo.items[0].kind == "tool_call"
