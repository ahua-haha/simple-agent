"""Tests for the replacement task manager."""

from __future__ import annotations

import tempfile

import pytest
from pi.ai.types import AssistantMessage, TextContent, ToolResultMessage

from simple_agent.db.db import Database
from simple_agent.task_manager import TaskManager
from simple_agent.task_manager.lifecycle import TaskLifecycleRuntime, TodoTaskLifecycle, UserTaskLifecycle
from simple_agent.task_manager.models import TodoTask, ToolCallTask, UserTask


def test_managed_task_defaults():
    task = UserTask(title="Build feature")
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


def _user_lifecycle(user_task):
    runtime = TaskLifecycleRuntime(messages=[], next_task=user_task, next_task_id_to_run=user_task.id)
    lifecycle = UserTaskLifecycle()
    lifecycle.set_data(runtime)
    return lifecycle


def _todo_lifecycle(todo):
    runtime = TaskLifecycleRuntime(messages=[], next_task=todo, next_task_id_to_run=todo.id)
    lifecycle = TodoTaskLifecycle()
    lifecycle.set_data(runtime)
    return lifecycle


def _create_todo(
    manager: TaskManager,
    title: str,
    *,
    start_message_id: int | None = None,
) -> TodoTask:
    lifecycle = manager.active_lifecycle_for_tools()
    if start_message_id is not None:
        lifecycle.current_assistant_message_id = start_message_id
    todo = lifecycle.create_todo_task(title=title)
    if start_message_id is not None:
        lifecycle.current_assistant_message_id = None
    manager.refresh_active_task()
    return todo


def _finish_todo(
    manager: TaskManager,
    result: str | None = None,
    *,
    end_message_id: int | None = None,
) -> TodoTask:
    lifecycle = manager.active_lifecycle_for_tools()
    if end_message_id is not None:
        lifecycle.current_assistant_message_id = end_message_id
    todo = lifecycle.finish_task(result=result)
    if end_message_id is not None:
        lifecycle.current_assistant_message_id = None
    manager.refresh_active_task()
    return todo


def _error_todo(
    manager: TaskManager,
    error: str,
    *,
    end_message_id: int | None = None,
) -> TodoTask:
    todo = manager.active_lifecycle_for_tools().error_task(
        error=error,
        end_message_id=end_message_id,
    )
    manager.refresh_active_task()
    return todo


def _record_tool_call(manager: TaskManager, tool_call_log_id: int):
    active_task = manager.active_lifecycle_for_tools().task
    tool_call = ToolCallTask(
        id=manager.allocate_task_id(),
        title=f"Tool call {tool_call_log_id}",
        status="done",
        parent_id=active_task.id,
        tool_call_log_id=tool_call_log_id,
    )
    active_task.children.append(tool_call)
    active_task.touch()
    return tool_call


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
    child = TodoTask(title="Runtime child", parent_id=10)
    task = UserTask(
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


def test_managed_task_roundtrip_preserves_message_boundaries():
    db = _make_db()
    task = TodoTask(
        title="Inspect files",
        start_message_id=12,
        end_message_id=15,
    )

    task.id = db.upsert_managed_task(task)
    loaded = db.get_managed_task(task.id)

    assert loaded.start_message_id == 12
    assert loaded.end_message_id == 15


def test_user_task_sync_persists_task_and_direct_children_only():
    db = _make_db()
    user_task = UserTask(id=1, title="Build feature")
    user_lifecycle = _user_lifecycle(user_task)
    todo = user_lifecycle.create_todo_task(task_id=2, title="Inspect files")
    todo.children.append(
        ToolCallTask(id=3, parent_id=todo.id, title="Tool call 7", status="done", tool_call_log_id=7)
    )
    direct_tool_call = ToolCallTask(id=4, parent_id=user_task.id, title="Tool call 8", status="done", tool_call_log_id=8)
    user_task.children.append(direct_tool_call)

    with db.create_session() as session:
        user_lifecycle.sync(db, session)
        session.commit()

    loaded_user_task = db.get_managed_task(1)
    loaded_todo = db.get_managed_task(2)
    loaded_nested_tool_call = db.get_managed_task(3)
    loaded_direct_tool_call = db.get_managed_task(4)
    assert loaded_user_task.title == "Build feature"
    assert loaded_todo.parent_id == user_task.id
    assert loaded_nested_tool_call is None
    assert loaded_direct_tool_call.parent_id == user_task.id
    assert loaded_direct_tool_call.tool_call_log_id == direct_tool_call.tool_call_log_id


def test_todo_task_sync_persists_task_and_direct_tool_calls():
    db = _make_db()
    todo = TodoTask(id=2, parent_id=1, title="Inspect files")
    lifecycle = _todo_lifecycle(todo)
    tool_call = ToolCallTask(id=3, parent_id=todo.id, title="Tool call 7", status="done", tool_call_log_id=7)
    todo.children.append(tool_call)

    with db.create_session() as session:
        lifecycle.sync(db, session)
        session.commit()

    loaded_todo = db.get_managed_task(2)
    loaded_tool_call = db.get_managed_task(3)
    assert loaded_todo.title == "Inspect files"
    assert loaded_tool_call.parent_id == todo.id
    assert loaded_tool_call.tool_call_log_id == tool_call.tool_call_log_id


def test_task_manager_save_syncs_each_task_in_tree():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    user_task = manager.create_user_task("Build feature")
    todo = _create_todo(manager, "Inspect files")
    tool_call = _record_tool_call(manager, 7)

    _save(manager)

    loaded_user_task = db.get_managed_task(user_task.id)
    loaded_todo = db.get_managed_task(todo.id)
    loaded_tool_call = db.get_managed_task(tool_call.id)
    assert loaded_user_task.title == "Build feature"
    assert loaded_todo.parent_id == user_task.id
    assert loaded_tool_call.parent_id == todo.id
    assert loaded_tool_call.tool_call_log_id == 7


def test_create_user_task_sets_active_user_task():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)

    user_task = manager.create_user_task("Build feature")

    assert user_task.id is not None
    assert user_task.kind == "user_task"
    assert user_task.title == "Build feature"
    assert manager.active_task_id == user_task.id
    assert db.get_managed_task(user_task.id) is None


def test_task_manager_maintains_one_active_task_lifecycle():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)

    assert not hasattr(manager, "_active_todo")
    assert not hasattr(manager, "_user_task_lifecycle")
    assert not hasattr(manager, "_active_todo_lifecycle")
    assert not hasattr(manager, "active_user_task")
    assert not hasattr(manager, "active_user_task_id")
    assert not hasattr(manager, "active_todo")
    assert not hasattr(manager, "active_todo_id")


def test_active_task_moves_to_created_child_and_back_to_parent_when_done():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    user_task = manager.create_user_task("Build feature")

    todo = manager.active_lifecycle_for_tools().create_todo_task(title="Inspect files")
    manager.refresh_active_task()

    assert manager.active_task_id == todo.id
    assert manager.active_lifecycle_for_tools().task is todo

    manager.active_lifecycle_for_tools().finish_task(result="Done")
    manager.refresh_active_task()

    assert manager.active_task_id == user_task.id
    assert manager.active_lifecycle_for_tools().task is user_task


def test_active_task_moves_to_parent_when_child_errors():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    user_task = manager.create_user_task("Build feature")
    todo = manager.active_lifecycle_for_tools().create_todo_task(title="Inspect files")
    manager.refresh_active_task()

    manager.active_lifecycle_for_tools().error_task(error="failed")
    manager.refresh_active_task()

    assert todo.status == "error"
    assert manager.active_task_id == user_task.id


def test_create_todo_appends_child_task_to_user_task():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    user_task = manager.create_user_task("Build feature")

    todo = _create_todo(manager, "Inspect files")

    assert todo.parent_id == user_task.id
    assert manager.active_task_id == todo.id
    assert db.get_managed_task(user_task.id) is None

    _save(manager)
    loaded_manager = TaskManager(db)
    _load(loaded_manager, user_task.id)
    loaded_user_task = loaded_manager.user_task

    assert [task.id for task in loaded_user_task.children] == [todo.id]
    assert [task.id for task in loaded_user_task.children] == [todo.id]


def test_task_manager_does_not_expose_lifecycle_mutation_wrappers():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    _create_todo(manager, "Inspect files")

    assert not hasattr(manager, "create_todo")
    assert not hasattr(manager, "finish_task")
    assert not hasattr(manager, "error_task")
    assert not hasattr(manager, "record_tool_call")
    assert not hasattr(manager, "record_turn_tool_calls")
    assert not hasattr(manager, "set_current_assistant_message_id")


def test_finish_todo_marks_done_and_clears_active_todo():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    todo = _create_todo(manager, "Inspect files")

    finished = _finish_todo(manager, "Found app.py", end_message_id=21)

    assert finished.id == todo.id
    assert finished.status == "done"
    assert finished.result == "Found app.py"
    assert finished.end_message_id == 21
    assert manager.active_task_id == todo.parent_id


def test_active_lifecycle_routes_to_user_task_without_active_todo():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    user_task = manager.create_user_task("Build feature")

    assert manager.active_lifecycle_for_tools().task is user_task


def test_create_todo_stores_start_message_id():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")

    todo = _create_todo(manager, "Inspect files", start_message_id=12)

    assert todo.start_message_id == 12


def test_error_todo_stores_end_message_id():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    todo = _create_todo(manager, "Inspect files")

    errored = _error_todo(manager, "failed", end_message_id=13)

    assert errored.id == todo.id
    assert errored.status == "error"
    assert errored.end_message_id == 13


def test_task_tools_use_current_assistant_message_id_for_boundaries():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    manager.active_lifecycle_for_tools().current_assistant_message_id = 22

    todo = _create_todo(manager, "Inspect files")
    manager.active_lifecycle_for_tools().current_assistant_message_id = 22
    finished = _finish_todo(manager, "Done")

    assert todo.start_message_id == 22
    assert finished.end_message_id == 22


def test_task_tool_created_todo_uses_current_lifecycle_message_boundary():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    lifecycle = manager.active_lifecycle_for_tools()
    tools = {tool.name: tool for tool in lifecycle.create_tools()}
    lifecycle.current_assistant_message_id = 31

    async def run():
        return await tools["create_todo"].execute("call_1", {"title": "Inspect files"})

    import asyncio

    asyncio.run(run())
    manager.refresh_active_task()

    active_task = manager.active_lifecycle_for_tools().task
    assert active_task.start_message_id == 31
    assert active_task.kind == "todo"


def test_record_tool_call_without_active_todo_attaches_to_user_task():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    user_task = manager.create_user_task("Build feature")

    tool_call_task = _record_tool_call(manager, 7)
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
    todo = _create_todo(manager, "Inspect files")

    tool_call_task = _record_tool_call(manager, 8)
    _save(manager)
    loaded_tool_call = db.get_managed_task(tool_call_task.id)

    assert loaded_tool_call.parent_id == todo.id
    assert loaded_tool_call.kind == "tool_call"
    assert loaded_tool_call.tool_call_log_id == 8


def test_active_lifecycle_creates_tool_call_entries_under_active_task():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    todo = _create_todo(manager, "Inspect files")
    tool_result = ToolResultMessage(
        toolCallId="call_1",
        toolName="ls",
        content=[TextContent(text="files")],
    )
    lifecycle = manager.active_lifecycle_for_tools()
    lifecycle.load_tool_call_log_id(9)

    records, tasks = lifecycle.create_tool_call_record_task_entries(
        assistant_message=AssistantMessage(role="assistant", content=[]),
        tool_result_messages=[tool_result],
    )
    lifecycle.task.children.extend(tasks)

    assert records == [(9, None, tool_result)]
    assert len(tasks) == 1
    assert tasks[0].parent_id == todo.id
    assert tasks[0].kind == "tool_call"
    assert tasks[0].tool_call_log_id == 9


def test_mixed_user_task_order_is_preserved():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    user_task = manager.create_user_task("Build feature")

    _record_tool_call(manager, 1)
    todo = _create_todo(manager, "Inspect files")
    _finish_todo(manager)
    _record_tool_call(manager, 2)

    _save(manager)
    loaded_manager = TaskManager(db)
    _load(loaded_manager, user_task.id)
    loaded_children = loaded_manager.user_task.children
    assert [(task.kind, task.id if task.kind == "todo" else task.tool_call_log_id) for task in loaded_children] == [
        ("tool_call", 1),
        ("todo", todo.id),
        ("tool_call", 2),
    ]


def test_load_loads_children_and_active_todo():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    user_task = manager.create_user_task("Build feature")
    todo = _create_todo(manager, "Inspect files")
    _record_tool_call(manager, 9)
    _save(manager)

    loaded_manager = TaskManager(db)
    _load(loaded_manager, user_task.id)
    loaded_user_task = loaded_manager.user_task
    loaded_todo = loaded_manager.active_lifecycle_for_tools().task

    assert loaded_manager.active_task_id == todo.id
    assert loaded_user_task.children[0].id == todo.id
    assert loaded_todo.children[0].tool_call_log_id == 9

    _finish_todo(loaded_manager, "Done")
    _save(loaded_manager)

    persisted_todo = db.get_managed_task(todo.id)
    assert persisted_todo.status == "done"
    loaded_children = loaded_todo.children
    assert loaded_children[0].tool_call_log_id == 9
