"""Tests for the replacement task manager."""

from __future__ import annotations

import tempfile

import pytest

from simple_agent.db.db import Database
from simple_agent.task_manager import TaskManager, TaskManagerError
from simple_agent.task_manager.models import ManagedTask, TaskItem


def test_task_item_defaults_to_task_ref():
    item = TaskItem(kind="task", ref_id=10)
    assert item.kind == "task"
    assert item.ref_id == 10


def test_managed_task_defaults():
    task = ManagedTask(kind="user_task", title="Build feature")
    assert task.kind == "user_task"
    assert task.status == "active"
    assert task.items == []
    assert task.result is None
    assert task.error is None


def test_managed_task_accepts_mixed_ordered_items():
    task = ManagedTask(
        kind="user_task",
        title="Build feature",
        items=[
            TaskItem(kind="tool_call", ref_id=1),
            TaskItem(kind="task", ref_id=2),
            TaskItem(kind="tool_call", ref_id=3),
        ],
    )
    assert [(item.kind, item.ref_id) for item in task.items] == [
        ("tool_call", 1),
        ("task", 2),
        ("tool_call", 3),
    ]


def _make_db() -> Database:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        return Database(f.name)


def test_managed_task_roundtrip_preserves_items():
    db = _make_db()
    task = ManagedTask(
        kind="user_task",
        title="Build feature",
        items=[TaskItem(kind="tool_call", ref_id=1), TaskItem(kind="task", ref_id=2)],
    )

    task.id = db.upsert_managed_task(task)
    loaded = db.get_managed_task(task.id)

    assert loaded is not None
    assert loaded.id == task.id
    assert loaded.kind == "user_task"
    assert loaded.title == "Build feature"
    assert [(item.kind, item.ref_id) for item in loaded.items] == [
        ("tool_call", 1),
        ("task", 2),
    ]


def test_create_user_task_sets_active_user_task():
    db = _make_db()
    manager = TaskManager(db)

    user_task = manager.create_user_task("Build feature")

    assert user_task.id is not None
    assert user_task.kind == "user_task"
    assert user_task.title == "Build feature"
    assert manager.active_user_task_id == user_task.id


def test_create_todo_appends_task_item_to_user_task():
    db = _make_db()
    manager = TaskManager(db)
    user_task = manager.create_user_task("Build feature")

    todo = manager.create_todo("Inspect files")
    loaded_user_task = db.get_managed_task(user_task.id)

    assert todo.parent_id == user_task.id
    assert manager.active_todo_id == todo.id
    assert [(item.kind, item.ref_id) for item in loaded_user_task.items] == [
        ("task", todo.id),
    ]


def test_create_todo_rejects_existing_active_todo():
    db = _make_db()
    manager = TaskManager(db)
    manager.create_user_task("Build feature")
    manager.create_todo("Inspect files")

    with pytest.raises(TaskManagerError, match="active todo"):
        manager.create_todo("Edit files")


def test_finish_todo_marks_done_and_clears_active_todo():
    db = _make_db()
    manager = TaskManager(db)
    manager.create_user_task("Build feature")
    todo = manager.create_todo("Inspect files")

    finished = manager.finish_todo("Found app.py")

    assert finished.id == todo.id
    assert finished.status == "done"
    assert finished.result == "Found app.py"
    assert manager.active_todo_id is None


def test_finish_todo_rejects_missing_active_todo():
    db = _make_db()
    manager = TaskManager(db)
    manager.create_user_task("Build feature")

    with pytest.raises(TaskManagerError, match="No active todo"):
        manager.finish_todo()


def test_record_tool_call_without_active_todo_attaches_to_user_task():
    db = _make_db()
    manager = TaskManager(db)
    user_task = manager.create_user_task("Build feature")

    manager.record_tool_call(7)
    loaded_user_task = db.get_managed_task(user_task.id)

    assert [(item.kind, item.ref_id) for item in loaded_user_task.items] == [
        ("tool_call", 7),
    ]


def test_record_tool_call_with_active_todo_attaches_to_todo():
    db = _make_db()
    manager = TaskManager(db)
    manager.create_user_task("Build feature")
    todo = manager.create_todo("Inspect files")

    manager.record_tool_call(8)
    loaded_todo = db.get_managed_task(todo.id)

    assert [(item.kind, item.ref_id) for item in loaded_todo.items] == [
        ("tool_call", 8),
    ]


def test_mixed_user_task_order_is_preserved():
    db = _make_db()
    manager = TaskManager(db)
    user_task = manager.create_user_task("Build feature")

    manager.record_tool_call(1)
    todo = manager.create_todo("Inspect files")
    manager.finish_todo()
    manager.record_tool_call(2)

    loaded_user_task = db.get_managed_task(user_task.id)
    assert [(item.kind, item.ref_id) for item in loaded_user_task.items] == [
        ("tool_call", 1),
        ("task", todo.id),
        ("tool_call", 2),
    ]
