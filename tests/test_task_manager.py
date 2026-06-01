"""Tests for the replacement task manager."""

from __future__ import annotations

import tempfile

from simple_agent.db.db import Database
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
