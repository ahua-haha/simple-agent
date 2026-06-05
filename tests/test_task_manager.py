"""Tests for the replacement task manager."""

from __future__ import annotations

import tempfile

import pytest

from simple_agent.db.db import Database
from simple_agent.task_manager import TaskManager, TaskManagerError, ToolCallReview
from simple_agent.task_manager.models import ManagedTask


def test_managed_task_defaults():
    task = ManagedTask(kind="user_task", title="Build feature")
    assert task.kind == "user_task"
    assert task.status == "active"
    assert not hasattr(task, "seq")
    assert task.children == []
    assert task.result is None
    assert task.error is None


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


def test_task_manager_load_requires_session():
    db = _make_db()
    manager = TaskManager(db)

    with pytest.raises(TypeError):
        manager.load(None)


def test_task_manager_save_requires_session():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)

    with pytest.raises(TypeError):
        manager.save()


def test_managed_task_roundtrip_preserves_parent():
    db = _make_db()
    child = ManagedTask(kind="todo", title="Runtime child", parent_id=10)
    task = ManagedTask(
        kind="user_task",
        title="Build feature",
        parent_id=10,
        children=[child],
    )

    task.id = db.upsert_managed_task(task)
    loaded = db.get_managed_task(task.id)

    assert loaded is not None
    assert loaded.id == task.id
    assert loaded.kind == "user_task"
    assert loaded.title == "Build feature"
    assert loaded.parent_id == 10
    assert not hasattr(loaded, "seq")
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
    _load(manager, None)

    user_task = manager.create_user_task("Build feature")

    assert user_task.id is not None
    assert user_task.kind == "user_task"
    assert user_task.title == "Build feature"
    assert manager.active_user_task_id == user_task.id
    assert db.get_managed_task(user_task.id) is None


def test_create_todo_appends_child_task_to_user_task():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    user_task = manager.create_user_task("Build feature")

    todo = manager.create_todo("Inspect files")

    assert todo.parent_id == user_task.id
    assert manager.active_todo_id == todo.id
    assert db.get_managed_task(user_task.id) is None

    _save(manager)
    loaded_manager = TaskManager(db)
    _load(loaded_manager, user_task.id)
    loaded_user_task = loaded_manager.active_user_task

    assert [task.id for task in loaded_user_task.children] == [todo.id]
    assert [task.id for task in loaded_user_task.children] == [todo.id]


def test_create_todo_rejects_existing_active_todo():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    manager.create_todo("Inspect files")

    with pytest.raises(TaskManagerError, match="active todo"):
        manager.create_todo("Edit files")


def test_finish_todo_marks_done_and_clears_active_todo():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
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
    _load(manager, None)
    manager.create_user_task("Build feature")

    with pytest.raises(TaskManagerError, match="No active todo"):
        manager.finish_task()


def test_create_todo_stores_create_tool_call_id():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")

    todo = manager.create_todo("Inspect files", tool_call_id="call_create")

    assert todo.create_tool_call_id == "call_create"


def test_error_todo_stores_end_tool_call_id():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    todo = manager.create_todo("Inspect files")

    errored = manager.error_task("failed", tool_call_id="call_error")

    assert errored.id == todo.id
    assert errored.status == "error"
    assert errored.end_tool_call_id == "call_error"


def test_record_tool_call_without_active_todo_attaches_to_user_task():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    user_task = manager.create_user_task("Build feature")

    tool_call_task = manager.record_tool_call(7)
    _save(manager)
    loaded_tool_call = db.get_managed_task(tool_call_task.id)

    assert loaded_tool_call.parent_id == user_task.id
    assert loaded_tool_call.kind == "tool_call"
    assert loaded_tool_call.tool_call_log_id == 7


def test_record_tool_call_with_active_todo_attaches_to_todo():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    todo = manager.create_todo("Inspect files")

    tool_call_task = manager.record_tool_call(8)
    _save(manager)
    loaded_tool_call = db.get_managed_task(tool_call_task.id)

    assert loaded_tool_call.parent_id == todo.id
    assert loaded_tool_call.kind == "tool_call"
    assert loaded_tool_call.tool_call_log_id == 8


def test_user_instruction_without_active_todo_asks_for_complexity_check():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")

    instruction = manager.user_instruction_text()

    assert "determine whether the user task is complex" in instruction.lower()
    assert "create the next small atomic todo" in instruction


def test_user_instruction_without_active_todo_after_many_tools_requires_todo():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    todo = manager.create_todo("Inspect files")
    manager.finish_task("Done")
    for tool_call_id in range(6):
        manager.record_tool_call(tool_call_id)

    instruction = manager.user_instruction_text()

    assert todo.status == "done"
    assert "More than 5 tool calls have run since the previous todo" in instruction
    assert "create a small atomic todo before doing more work" in instruction


def test_user_instruction_with_active_todo_focuses_on_current_todo():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    manager.create_todo("Inspect files")

    instruction = manager.user_instruction_text()

    assert "Focus on the active todo" in instruction
    assert "finish_todo immediately when it is complete" in instruction


def test_user_instruction_with_active_todo_after_many_tools_asks_to_finish_if_done():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    manager.create_todo("Inspect files")
    for tool_call_id in range(11):
        manager.record_tool_call(tool_call_id)

    instruction = manager.user_instruction_text()

    assert "More than 10 tool calls have run for the active todo" in instruction
    assert "determine whether the active todo is finished" in instruction.lower()
    assert "call finish_todo now" in instruction


def test_review_task_tree_renders_tasks_and_tool_calls_with_temp_sequence():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    manager.record_tool_call(10)
    todo = manager.create_todo("Inspect files")
    manager.record_tool_call(11)
    manager.finish_task("Found manager.py")
    manager.record_tool_call(12)

    review = manager.review_task_tree(
        tool_calls={
            10: ToolCallReview(name="ls", arguments={"path": "."}),
            11: ToolCallReview(name="sed", arguments={"file": "manager.py"}),
            12: ToolCallReview(name="rg", arguments={"pattern": "TaskManager"}),
        }
    )

    assert review.text == "\n".join(
        [
            "Task tree:",
            "- user_task [active] Build feature",
            '  - tool_call 1. ls args: {"path":"."}',
            "  - todo [done] Inspect files",
            "    result: Found manager.py",
            '    - tool_call 2. sed args: {"file":"manager.py"}',
            '  - tool_call 3. rg args: {"pattern":"TaskManager"}',
        ]
    )
    assert review.tool_call_log_ids == {1: 10, 2: 11, 3: 12}
    assert "log_id" not in review.text
    assert "parent:" not in review.text


def test_review_task_tree_depth_limits_tree_and_keeps_direct_tool_calls():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    manager.record_tool_call(10)
    manager.create_todo("Inspect files")
    manager.record_tool_call(11)

    review = manager.review_task_tree(
        depth=1,
        tool_calls={
            10: ToolCallReview(name="ls", arguments={"path": "."}),
            11: ToolCallReview(name="sed", arguments={"file": "manager.py"}),
        },
    )

    assert review.text == "\n".join(
        [
            "Task tree:",
            "- user_task [active] Build feature",
            '  - tool_call 1. ls args: {"path":"."}',
            "  - todo [active] Inspect files",
        ]
    )
    assert review.tool_call_log_ids == {1: 10}
    assert "sed" not in review.text


def test_review_task_tree_flat_format_flattens_tool_calls_under_user_task():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    manager.record_tool_call(10)
    manager.create_todo("Inspect files")
    manager.record_tool_call(11)

    review = manager.review_task_tree(
        format="flat",
        tool_calls={
            10: ToolCallReview(name="ls", arguments={"path": "."}),
            11: ToolCallReview(name="sed", arguments={"file": "manager.py"}),
        },
    )

    assert review.text == "\n".join(
        [
            "Task tree:",
            "- user_task [active] Build feature",
            '  - tool_call 1. ls args: {"path":"."}',
            '  - tool_call 2. sed args: {"file":"manager.py"}',
        ]
    )
    assert review.tool_call_log_ids == {1: 10, 2: 11}
    assert "todo" not in review.text


def test_mixed_user_task_order_is_preserved():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    user_task = manager.create_user_task("Build feature")

    manager.record_tool_call(1)
    todo = manager.create_todo("Inspect files")
    manager.finish_task()
    manager.record_tool_call(2)

    _save(manager)
    loaded_manager = TaskManager(db)
    _load(loaded_manager, user_task.id)
    loaded_children = loaded_manager.active_user_task.children
    assert [(task.kind, task.id if task.kind == "todo" else task.tool_call_log_id) for task in loaded_children] == [
        ("tool_call", 1),
        ("todo", todo.id),
        ("tool_call", 2),
    ]


def test_compact_scope_selects_first_todo_through_latest_finished_todo():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
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
    _load(manager, None)
    manager.create_user_task("Build feature")
    manager.create_todo("Still active")

    assert manager.compact_scope() is None


def test_compact_tools_create_one_finished_compacted_todo():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
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
    _load(manager, None)
    user_task = manager.create_user_task("Build feature")
    first = manager.create_todo("One")
    manager.finish_task("done", tool_call_id="call_finish_1")
    second = manager.create_todo("Two")
    manager.finish_task("done", tool_call_id="call_finish_2")
    active = manager.create_todo("Three")
    _save(manager)
    manager.begin_compact_buffer()
    compacted = manager.create_compacted_todo("Summary")
    compacted_buffer_id = compacted.id
    manager.finish_compacted_todo()

    with db.create_session() as session:
        compacted = manager.replace_compact_scope(session=session)
        session.commit()

    loaded_compacted = db.get_managed_task(compacted.id)
    loaded_active = db.get_managed_task(active.id)
    loaded_manager = TaskManager(db)
    _load(loaded_manager, user_task.id)

    assert compacted.id == first.id
    assert db.get_managed_task(compacted_buffer_id) is None
    assert db.get_managed_task(second.id) is None
    assert loaded_active.id == active.id
    assert [task.id for task in loaded_manager.active_user_task.children] == [compacted.id, active.id]
    assert loaded_compacted.parent_id == user_task.id


def test_load_loads_children_and_active_todo():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    user_task = manager.create_user_task("Build feature")
    todo = manager.create_todo("Inspect files")
    manager.record_tool_call(9)
    _save(manager)

    loaded_manager = TaskManager(db)
    _load(loaded_manager, user_task.id)
    loaded_user_task = loaded_manager.active_user_task
    loaded_todo = loaded_manager.active_todo

    assert loaded_manager.active_user_task_id == user_task.id
    assert loaded_manager.active_todo_id == todo.id
    assert loaded_user_task.children[0].id == todo.id
    assert loaded_todo.children[0].tool_call_log_id == 9

    loaded_manager.finish_task("Done")
    _save(loaded_manager)

    persisted_todo = db.get_managed_task(todo.id)
    assert persisted_todo.status == "done"
    loaded_children = loaded_todo.children
    assert loaded_children[0].tool_call_log_id == 9
