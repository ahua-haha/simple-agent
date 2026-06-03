"""Tests for the replacement task manager."""

from __future__ import annotations

import tempfile

import pytest

from simple_agent.db.db import Database
from simple_agent.task_manager import TaskManager, TaskManagerError
from simple_agent.task_manager.models import ManagedTask


def test_managed_task_defaults():
    task = ManagedTask(kind="user_task", title="Build feature")
    assert task.kind == "user_task"
    assert task.status == "active"
    assert task.seq == ""
    assert task.children == []
    assert task.result is None
    assert task.error is None


def _make_db() -> Database:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        return Database(f.name)


def test_managed_task_roundtrip_preserves_parent_and_seq():
    db = _make_db()
    child = ManagedTask(kind="todo", title="Runtime child", parent_id=10, seq="V")
    task = ManagedTask(
        kind="user_task",
        title="Build feature",
        parent_id=10,
        seq="U",
        children=[child],
    )

    task.id = db.upsert_managed_task(task)
    loaded = db.get_managed_task(task.id)

    assert loaded is not None
    assert loaded.id == task.id
    assert loaded.kind == "user_task"
    assert loaded.title == "Build feature"
    assert loaded.parent_id == 10
    assert loaded.seq == "U"
    assert loaded.children == []


def test_managed_task_roundtrip_preserves_create_tool_call_id():
    db = _make_db()
    task = ManagedTask(
        kind="todo",
        title="Inspect files",
        create_tool_call_id="call_create",
        end_tool_call_id="call_finish",
    )

    task.id = db.upsert_managed_task(task)
    loaded = db.get_managed_task(task.id)

    assert loaded.create_tool_call_id == "call_create"
    assert loaded.end_tool_call_id == "call_finish"


def test_create_user_task_sets_active_user_task():
    db = _make_db()
    manager = TaskManager(db)
    manager.load(None)

    user_task = manager.create_user_task("Build feature")

    assert user_task.id is not None
    assert user_task.kind == "user_task"
    assert user_task.title == "Build feature"
    assert user_task.seq != ""
    assert manager.active_user_task_id == user_task.id
    assert db.get_managed_task(user_task.id) is None


def test_create_todo_appends_child_task_to_user_task():
    db = _make_db()
    manager = TaskManager(db)
    manager.load(None)
    user_task = manager.create_user_task("Build feature")

    todo = manager.create_todo("Inspect files")

    assert todo.parent_id == user_task.id
    assert manager.active_todo_id == todo.id
    assert db.get_managed_task(user_task.id) is None

    manager.save()
    loaded_manager = TaskManager(db)
    loaded_manager.load(user_task.id)
    loaded_user_task = loaded_manager.active_user_task

    assert [task.id for task in loaded_user_task.children] == [todo.id]
    assert [task.id for task in loaded_user_task.children] == [todo.id]


def test_create_todo_rejects_existing_active_todo():
    db = _make_db()
    manager = TaskManager(db)
    manager.load(None)
    manager.create_user_task("Build feature")
    manager.create_todo("Inspect files")

    with pytest.raises(TaskManagerError, match="active todo"):
        manager.create_todo("Edit files")


def test_finish_todo_marks_done_and_clears_active_todo():
    db = _make_db()
    manager = TaskManager(db)
    manager.load(None)
    manager.create_user_task("Build feature")
    todo = manager.create_todo("Inspect files")

    finished = manager.finish_task("Found app.py", tool_call_id="call_finish")

    assert finished.id == todo.id
    assert finished.status == "done"
    assert finished.result == "Found app.py"
    assert finished.end_tool_call_id == "call_finish"
    assert manager.active_todo_id is None


def test_finish_todo_rejects_missing_active_todo():
    db = _make_db()
    manager = TaskManager(db)
    manager.load(None)
    manager.create_user_task("Build feature")

    with pytest.raises(TaskManagerError, match="No active todo"):
        manager.finish_task()


def test_create_todo_stores_create_tool_call_id():
    db = _make_db()
    manager = TaskManager(db)
    manager.load(None)
    manager.create_user_task("Build feature")

    todo = manager.create_todo("Inspect files", tool_call_id="call_create")

    assert todo.create_tool_call_id == "call_create"


def test_error_todo_stores_end_tool_call_id():
    db = _make_db()
    manager = TaskManager(db)
    manager.load(None)
    manager.create_user_task("Build feature")
    todo = manager.create_todo("Inspect files")

    errored = manager.error_task("failed", tool_call_id="call_error")

    assert errored.id == todo.id
    assert errored.status == "error"
    assert errored.end_tool_call_id == "call_error"


def test_record_tool_call_without_active_todo_attaches_to_user_task():
    db = _make_db()
    manager = TaskManager(db)
    manager.load(None)
    user_task = manager.create_user_task("Build feature")

    tool_call_task = manager.record_tool_call(7)
    manager.save()
    loaded_tool_call = db.get_managed_task(tool_call_task.id)

    assert loaded_tool_call.parent_id == user_task.id
    assert loaded_tool_call.kind == "tool_call"
    assert loaded_tool_call.tool_call_log_id == 7


def test_record_tool_call_with_active_todo_attaches_to_todo():
    db = _make_db()
    manager = TaskManager(db)
    manager.load(None)
    manager.create_user_task("Build feature")
    todo = manager.create_todo("Inspect files")

    tool_call_task = manager.record_tool_call(8)
    manager.save()
    loaded_tool_call = db.get_managed_task(tool_call_task.id)

    assert loaded_tool_call.parent_id == todo.id
    assert loaded_tool_call.kind == "tool_call"
    assert loaded_tool_call.tool_call_log_id == 8


def test_mixed_user_task_order_is_preserved():
    db = _make_db()
    manager = TaskManager(db)
    manager.load(None)
    user_task = manager.create_user_task("Build feature")

    manager.record_tool_call(1)
    todo = manager.create_todo("Inspect files")
    manager.finish_task()
    manager.record_tool_call(2)

    manager.save()
    loaded_manager = TaskManager(db)
    loaded_manager.load(user_task.id)
    loaded_children = loaded_manager.active_user_task.children
    assert [(task.kind, task.id if task.kind == "todo" else task.tool_call_log_id) for task in loaded_children] == [
        ("tool_call", 1),
        ("todo", todo.id),
        ("tool_call", 2),
    ]


def test_next_task_seq_advances_after_in_memory_assignment():
    db = _make_db()
    manager = TaskManager(db)
    manager.load(None)
    manager.create_user_task("Build feature")

    first = manager.record_tool_call(1)
    second = manager.record_tool_call(2)

    assert first.seq < second.seq


def test_next_task_seq_continues_from_database_next_seq():
    db = _make_db()
    manager = TaskManager(db)
    manager.load(None)
    first_user_task = manager.create_user_task("Build feature")
    manager.save()

    next_manager = TaskManager(db)
    next_manager.load(None)
    second_user_task = next_manager.create_user_task("Build another feature")

    assert first_user_task.seq < second_user_task.seq


def test_compact_scope_selects_first_todo_through_latest_finished_todo():
    db = _make_db()
    manager = TaskManager(db)
    manager.load(None)
    manager.create_user_task("Build feature")
    first = manager.create_todo("One")
    manager.finish_task("done")
    second = manager.create_todo("Two")
    manager.finish_task("done")
    manager.create_todo("Three")

    scope = manager.compact_scope()

    assert [task.id for task in scope.compact_todos] == [first.id, second.id]
    assert [task.title for task in scope.preserved_todos] == ["Three"]


def test_compact_scope_returns_none_without_finished_todo():
    db = _make_db()
    manager = TaskManager(db)
    manager.load(None)
    manager.create_user_task("Build feature")
    manager.create_todo("Still active")

    assert manager.compact_scope() is None


def test_compact_tools_create_one_finished_compacted_todo():
    db = _make_db()
    manager = TaskManager(db)
    manager.load(None)
    manager.create_user_task("Build feature")
    manager.begin_compact_buffer()

    compacted = manager.create_compacted_todo("Summary")
    manager.record_compacted_tool_call(5)
    manager.finish_compacted_todo()

    result = manager.consume_compact_buffer()

    assert result.id == compacted.id
    assert result.status == "done"
    assert result.result == "Summary"
    assert [child.tool_call_log_id for child in result.children] == [5]


def test_replace_compact_scope_persists_rebuilt_task_tree():
    db = _make_db()
    manager = TaskManager(db)
    manager.load(None)
    user_task = manager.create_user_task("Build feature")
    first = manager.create_todo("One")
    manager.finish_task("done", tool_call_id="call_finish_1")
    second = manager.create_todo("Two")
    manager.finish_task("done", tool_call_id="call_finish_2")
    active = manager.create_todo("Three")
    manager.save()
    manager.begin_compact_buffer()
    compacted = manager.create_compacted_todo("Summary")
    manager.finish_compacted_todo()

    with db.create_session() as session:
        compacted = manager.replace_compact_scope(session=session)
        session.commit()

    loaded_compacted = db.get_managed_task(compacted.id)
    loaded_active = db.get_managed_task(active.id)
    loaded_manager = TaskManager(db)
    loaded_manager.load(user_task.id)

    assert db.get_managed_task(first.id) is None
    assert db.get_managed_task(second.id) is None
    assert loaded_active.id == active.id
    assert [task.id for task in loaded_manager.active_user_task.children] == [compacted.id, active.id]
    assert loaded_compacted.parent_id == user_task.id
    assert loaded_compacted.seq < loaded_active.seq


def test_load_loads_children_and_active_todo():
    db = _make_db()
    manager = TaskManager(db)
    manager.load(None)
    user_task = manager.create_user_task("Build feature")
    todo = manager.create_todo("Inspect files")
    manager.record_tool_call(9)
    manager.save()

    loaded_manager = TaskManager(db)
    loaded_manager.load(user_task.id)
    loaded_user_task = loaded_manager.active_user_task
    loaded_todo = loaded_manager.active_todo

    assert loaded_manager.active_user_task_id == user_task.id
    assert loaded_manager.active_todo_id == todo.id
    assert loaded_user_task.children[0].id == todo.id
    assert loaded_todo.children[0].tool_call_log_id == 9

    loaded_manager.finish_task("Done")
    loaded_manager.save()

    persisted_todo = db.get_managed_task(todo.id)
    assert persisted_todo.status == "done"
    loaded_children = loaded_todo.children
    assert loaded_children[0].tool_call_log_id == 9
