"""Tests for task-manager-backed tools."""

from __future__ import annotations

import asyncio
import tempfile

from simple_agent.db.db import Database
from simple_agent.task_manager import TaskManager


def _make_db() -> Database:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        return Database(f.name)


def _load(manager: TaskManager, active_user_task_id: int | None) -> None:
    with manager._db.create_session() as session:
        manager.load(active_user_task_id, session=session)


def _save(manager: TaskManager) -> None:
    with manager._db.create_session() as session:
        manager.save(session=session)
        session.commit()


def test_create_todo_tool_creates_active_todo():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    tool = manager.create_create_todo_tool()

    async def run():
        return await tool.execute("call_1", {"title": "Inspect files"})

    result = asyncio.run(run())

    assert "Todos:" in result.content[0].text
    assert "- [active] Inspect files" in result.content[0].text
    assert f"- {manager.active_todo_id}:" not in result.content[0].text
    assert manager.active_todo_id is not None


def test_finish_todo_tool_finishes_active_todo():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    todo = manager.create_todo("Inspect files")
    tool = manager.create_finish_todo_tool()

    async def run():
        return await tool.execute("call_1", {"result": "Inspected files"})

    result = asyncio.run(run())
    _save(manager)
    loaded = db.get_managed_task(todo.id)

    assert loaded.status == "done"
    assert loaded.result == "Inspected files"
    assert manager.active_todo_id is None
    assert "- [done] Inspect files" in result.content[0].text
    assert f"- {todo.id}:" not in result.content[0].text
    assert "result=Inspected files" in result.content[0].text


def test_error_todo_tool_returns_latest_todo_status():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    todo = manager.create_todo("Inspect files")
    tool = manager.create_error_todo_tool()

    async def run():
        return await tool.execute("call_1", {"error": "Missing dependency"})

    result = asyncio.run(run())
    _save(manager)
    loaded = db.get_managed_task(todo.id)

    assert loaded.status == "error"
    assert manager.active_todo_id is None
    assert "- [error] Inspect files" in result.content[0].text
    assert f"- {todo.id}:" not in result.content[0].text
    assert "error=Missing dependency" in result.content[0].text


def test_todo_tools_do_not_require_runner_wrapping():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    tool = manager.create_create_todo_tool()

    async def run():
        return await tool.execute("call_1", {"title": "Inspect files"})

    result = asyncio.run(run())

    assert "Inspect files" in result.content[0].text


def test_user_task_tools_include_create_todo_and_finish_user_task():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")

    tools = [tool.name for tool in manager.create_tools()]

    assert tools == ["create_todo", "finish_user_task"]


def test_active_todo_tools_include_todo_lifecycle_tools():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    manager.create_todo("Inspect files")

    tools = [tool.name for tool in manager.create_tools()]

    assert tools == ["finish_todo", "error_todo"]


def test_finish_user_task_tool_finishes_user_task():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    user_task = manager.create_user_task("Build feature")
    tool = manager.create_finish_user_task_tool()

    async def run():
        return await tool.execute("call_1", {"result": "Feature built"})

    result = asyncio.run(run())

    assert user_task.status == "done"
    assert user_task.result == "Feature built"
    assert manager.active_user_task_id is None
    assert "User task finished: Feature built" in result.content[0].text
