"""Tests for the replacement task manager."""

from __future__ import annotations

import tempfile

import pytest
from pi.ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage

from simple_agent.db.db import Database
from simple_agent.task_manager import TaskManager, ToolCallReview
from simple_agent.task_manager.lifecycle import TodoTaskLifecycle, UserTaskLifecycle
from simple_agent.task_manager.models import TaskRuntimeContext, TodoTask, UserTask


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


def _runtime_context(manager: TaskManager) -> TaskRuntimeContext:
    return TaskRuntimeContext(
        session_id="session_a",
        context_tokens=100,
        total_tool_calls=0,
        active_task_tool_calls=manager.active_task_tool_call_count(),
    )


def _create_todo(
    manager: TaskManager,
    title: str,
    *,
    start_message_id: int | None = None,
) -> TodoTask:
    todo = manager.active_lifecycle_for_tools().create_todo_task(
        title=title,
        start_message_id=start_message_id,
    )
    manager.refresh_active_task()
    return todo


def _finish_todo(
    manager: TaskManager,
    result: str | None = None,
    *,
    end_message_id: int | None = None,
) -> TodoTask:
    todo = manager.active_lifecycle_for_tools().finish_task(
        result=result,
        end_message_id=end_message_id,
    )
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
    return manager.active_lifecycle_for_tools().record_tool_call(tool_call_log_id)


async def _run_compact_tools(
    manager: TaskManager,
    *,
    description: str,
    tool_call_log_ids: list[int] | None = None,
):
    tools = {tool.name: tool for tool in manager.create_compact_tools()}
    create_result = await tools["create_compacted_user_task"].execute(
        "create",
        {"description": description},
    )
    compacted_id = int(create_result.content[0].text.removeprefix("created compacted user task "))
    for index, tool_call_log_id in enumerate(tool_call_log_ids or [], start=1):
        await tools["record_compacted_tool_call"].execute(
            f"record_{index}",
            {"tool_call_log_id": tool_call_log_id},
        )
    await tools["finish_compacted_user_task"].execute("finish", {})
    return compacted_id


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
    user_lifecycle = UserTaskLifecycle(user_task)
    todo = user_lifecycle.create_todo_task(task_id=2, title="Inspect files")
    TodoTaskLifecycle(todo, user_task=user_task).append_tool_call_task(task_id=3, tool_call_log_id=7)
    direct_tool_call = user_lifecycle.append_tool_call_task(task_id=4, tool_call_log_id=8)

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
    lifecycle = TodoTaskLifecycle(todo)
    tool_call = lifecycle.append_tool_call_task(task_id=3, tool_call_log_id=7)

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
    manager.set_current_assistant_message_id(22)

    todo = _create_todo(manager, "Inspect files")
    manager.set_current_assistant_message_id(22)
    finished = _finish_todo(manager, "Done")

    assert todo.start_message_id == 22
    assert finished.end_message_id == 22


def test_task_manager_routes_message_id_to_active_lifecycle():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    tools = {tool.name: tool for tool in manager.create_tools()}
    manager.set_current_assistant_message_id(31)

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


def test_record_turn_tool_calls_appends_task_under_active_task():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    todo = _create_todo(manager, "Inspect files")
    assistant_message = AssistantMessage(
        role="assistant",
        content=[ToolCall(id="call_1", name="ls", arguments={"path": "."})],
    )
    tool_result = ToolResultMessage(
        toolCallId="call_1",
        toolName="ls",
        content=[TextContent(text="files")],
    )

    tasks = manager.record_turn_tool_calls(
        target_task=todo,
        assistant_message=assistant_message,
        tool_call_records=[(9, None, tool_result)],
    )

    assert len(tasks) == 1
    assert tasks[0].parent_id == todo.id
    assert tasks[0].kind == "tool_call"
    assert tasks[0].tool_call_log_id == 9


def test_user_instruction_without_active_todo_asks_for_complexity_check():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")

    instruction = manager.user_instruction_text(_runtime_context(manager))

    assert "determine whether the user task is complex" in instruction.lower()
    assert "create the next small atomic todo" in instruction


def test_user_instruction_without_active_todo_after_many_tools_requires_todo():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    todo = _create_todo(manager, "Inspect files")
    _finish_todo(manager, "Done")
    for tool_call_id in range(6):
        _record_tool_call(manager, tool_call_id)

    instruction = manager.user_instruction_text(_runtime_context(manager))

    assert todo.status == "done"
    assert "More than 5 tool calls have run since the previous todo" in instruction
    assert "create a small atomic todo before doing more work" in instruction


def test_user_instruction_with_active_todo_focuses_on_current_todo():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    _create_todo(manager, "Inspect files")

    instruction = manager.user_instruction_text(_runtime_context(manager))

    assert "Focus on the active todo" in instruction
    assert "finish_todo immediately when it is complete" in instruction


def test_user_instruction_with_active_todo_after_many_tools_asks_to_finish_if_done():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    _create_todo(manager, "Inspect files")
    for tool_call_id in range(11):
        _record_tool_call(manager, tool_call_id)

    instruction = manager.user_instruction_text(_runtime_context(manager))

    assert "More than 10 tool calls have run for the active todo" in instruction
    assert "determine whether the active todo is finished" in instruction.lower()
    assert "call finish_todo now" in instruction


def test_user_instruction_routes_to_active_todo_before_user_task():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    _create_todo(manager, "Inspect files")

    instruction = manager.user_instruction_text(_runtime_context(manager))

    assert "Focus on the active todo: Inspect files" in instruction
    assert "Determine whether the user task is complex" not in instruction


def test_review_task_tree_renders_tasks_and_tool_calls_with_temp_sequence():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    _record_tool_call(manager, 10)
    todo = _create_todo(manager, "Inspect files")
    _record_tool_call(manager, 11)
    _finish_todo(manager, "Found manager.py")
    _record_tool_call(manager, 12)

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
    _record_tool_call(manager, 10)
    _create_todo(manager, "Inspect files")
    _record_tool_call(manager, 11)

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
    _record_tool_call(manager, 10)
    _create_todo(manager, "Inspect files")
    _record_tool_call(manager, 11)

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


def test_begin_compact_returns_false_when_running():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    _create_todo(manager, "One")
    _finish_todo(manager, "done")
    _create_todo(manager, "Two")
    _finish_todo(manager, "done")
    _create_todo(manager, "Three")

    assert manager.begin_compact(run_done=False) is False


def test_begin_compact_returns_false_without_finished_todo_when_running():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    _create_todo(manager, "Still active")

    assert manager.begin_compact(run_done=False) is False


def test_begin_compact_targets_done_user_task():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    first_tool = _record_tool_call(manager, 10)
    first = _create_todo(manager, "One")
    _finish_todo(manager, "done")
    second_tool = _record_tool_call(manager, 11)
    manager.finish_user_task(end_message_id=8)

    assert manager.begin_compact(run_done=True) is True
    assert manager.active_task_id == manager.user_task.id


def test_compact_instruction_text_includes_task_view_and_tool_call_directives():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    _record_tool_call(manager, 10)
    todo = _create_todo(manager, "Inspect files")
    _record_tool_call(manager, 11)
    _finish_todo(manager, "Found manager.py")
    manager.finish_user_task(end_message_id=8)
    assert manager.begin_compact(run_done=True) is True
    db.insert_runner_tool_call(
        id=10,
        session_id="session_a",
        tool_call_id="call_10",
        tool_name="ls",
        tool_call_json='{"arguments":{"path":"."}}',
        tool_result_json="{}",
    )
    db.insert_runner_tool_call(
        id=11,
        session_id="session_a",
        tool_call_id="call_11",
        tool_name="sed",
        tool_call_json='{"arguments":{"file":"manager.py"}}',
        tool_result_json="{}",
    )

    instruction = manager.compact_instruction_text(
        session_id="session_a",
    )

    assert "Complete the compacted user task information first" in instruction
    assert "define the task result" in instruction
    assert "Record every must-include tool call" in instruction
    assert "Task view to compact:" in instruction
    assert "- user_task [done] Build feature" in instruction
    assert "- todo [done] Inspect files" in instruction
    assert 'tool_call 1. ls args: {"path":"."}' in instruction
    assert 'tool_call 2. sed args: {"file":"manager.py"}' in instruction


@pytest.mark.asyncio
async def test_compact_tools_define_finished_compacted_user_task():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature", start_message_id=1)
    _create_todo(manager, "Inspect files", start_message_id=2)
    _finish_todo(manager, "Done", end_message_id=4)
    manager.finish_user_task(end_message_id=5)
    assert manager.begin_compact(run_done=True) is True

    compacted_id = await _run_compact_tools(
        manager,
        description="Summary",
        tool_call_log_ids=[5],
    )

    assert manager.user_task.id == compacted_id
    assert manager.user_task.status == "done"
    assert manager.user_task.result == "Summary"


@pytest.mark.asyncio
async def test_compacted_messages_uses_user_task_boundaries_when_done():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature", start_message_id=1)
    _record_tool_call(manager, 10)
    manager.finish_user_task(end_message_id=8)
    assert manager.begin_compact(run_done=True) is True
    await _run_compact_tools(manager, description="Summary")

    start_message_id, end_message_id, _messages = manager.compacted_messages()

    assert start_message_id == 1
    assert end_message_id == 8


@pytest.mark.asyncio
async def test_compacted_messages_returns_summary_message():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature", start_message_id=1)
    _create_todo(manager, "Inspect files", start_message_id=2)
    _finish_todo(manager, "Done", end_message_id=4)
    manager.finish_user_task(end_message_id=5)
    assert manager.begin_compact(run_done=True) is True
    await _run_compact_tools(
        manager,
        description="Summary",
        tool_call_log_ids=[5],
    )

    _start_message_id, _end_message_id, messages = manager.compacted_messages()

    assert len(messages) == 1
    assert messages[0].role == "assistant"
    assert messages[0].content[0].text == "Compacted user task: Summary\nUseful tool calls: [5]"


@pytest.mark.asyncio
async def test_sync_compaction_replaces_whole_user_task_when_done():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    user_task = manager.create_user_task("Build feature")
    first_tool = _record_tool_call(manager, 10)
    todo = _create_todo(manager, "One")
    _finish_todo(manager, "done", end_message_id=1)
    second_tool = _record_tool_call(manager, 11)
    manager.finish_user_task(end_message_id=2)
    _save(manager)
    assert manager.begin_compact(run_done=True) is True
    compacted_id = await _run_compact_tools(
        manager,
        description="Whole task summary",
    )

    with db.create_session() as session:
        compacted = manager.sync_compaction(session=session)
        session.commit()

    loaded_manager = TaskManager(db)
    _load(loaded_manager, user_task.id)

    assert compacted.id == user_task.id
    assert compacted_id == user_task.id
    assert db.get_managed_task(todo.id) is None
    assert db.get_managed_task(second_tool.id) is None
    assert loaded_manager.user_task.children == []
    assert loaded_manager.user_task.result == "Whole task summary"


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
