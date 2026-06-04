"""Tests for task-manager-backed tools."""

from __future__ import annotations

import asyncio
import tempfile

from simple_agent.db.db import Database
from simple_agent.session.runner import SessionRunner
from simple_agent.task_manager import TaskManager


class _FakeAgentProcess:
    def subscribe(self, callback):
        pass

    def unsubscribe(self, callback):
        pass


def _make_db() -> Database:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        return Database(f.name)


def _make_runner(db: Database, manager: TaskManager) -> SessionRunner:
    return SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=manager,
        agent_process=_FakeAgentProcess(),
        cancel_event=asyncio.Event(),
    )


def test_create_todo_tool_creates_active_todo():
    db = _make_db()
    manager = TaskManager(db)
    manager.load(None)
    manager.create_user_task("Build feature")
    runner = _make_runner(db, manager)
    tool = runner.wrap_tool(manager.create_create_todo_tool())

    async def run():
        return await tool.execute("call_1", {"title": "Inspect files"})

    result = asyncio.run(run())

    assert "Todos:" in result.content[0].text
    assert f"- {manager.active_todo_id}: [active] Inspect files" in result.content[0].text
    assert manager.active_todo_id is not None


def test_finish_todo_tool_finishes_active_todo():
    db = _make_db()
    manager = TaskManager(db)
    manager.load(None)
    manager.create_user_task("Build feature")
    todo = manager.create_todo("Inspect files")
    runner = _make_runner(db, manager)
    tool = runner.wrap_tool(manager.create_finish_todo_tool())

    async def run():
        return await tool.execute("call_1", {"result": "Inspected files"})

    result = asyncio.run(run())
    manager.save()
    loaded = db.get_managed_task(todo.id)

    assert loaded.status == "done"
    assert loaded.result == "Inspected files"
    assert manager.active_todo_id is None
    assert f"- {todo.id}: [done] Inspect files" in result.content[0].text
    assert "result=Inspected files" in result.content[0].text


def test_error_todo_tool_returns_latest_todo_status():
    db = _make_db()
    manager = TaskManager(db)
    manager.load(None)
    manager.create_user_task("Build feature")
    todo = manager.create_todo("Inspect files")
    runner = _make_runner(db, manager)
    tool = runner.wrap_tool(manager.create_error_todo_tool())

    async def run():
        return await tool.execute("call_1", {"error": "Missing dependency"})

    result = asyncio.run(run())
    manager.save()
    loaded = db.get_managed_task(todo.id)

    assert loaded.status == "error"
    assert manager.active_todo_id is None
    assert f"- {todo.id}: [error] Inspect files" in result.content[0].text
    assert "error=Missing dependency" in result.content[0].text


def test_normal_tool_call_records_under_active_todo():
    db = _make_db()
    manager = TaskManager(db)
    manager.load(None)
    manager.create_user_task("Build feature")
    todo = manager.create_todo("Inspect files")
    runner = _make_runner(db, manager)

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
    wrapped = runner.wrap_tool(tool)

    async def run():
        return await wrapped.execute("call_1", {})

    asyncio.run(run())
    manager.save()
    loaded_manager = TaskManager(db)
    loaded_manager.load(todo.parent_id)
    loaded_children = loaded_manager.active_todo.children

    assert len(loaded_children) == 1
    assert loaded_children[0].kind == "tool_call"
