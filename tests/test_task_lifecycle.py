import pytest

from pi.ai.types import AssistantMessage, TextContent

from simple_agent.db.db import Database
from simple_agent.task_manager.lifecycle import TodoTaskLifecycle, UserTaskLifecycle
from simple_agent.task_manager.models import TodoTask, ToolCallTask, UserTask


def _make_db(tmp_path):
    return Database(str(tmp_path / "session.db"))


def test_user_task_instruction_asks_for_complexity_check_when_tool_count_is_small():
    task = UserTask(title="Build feature")
    lifecycle = UserTaskLifecycle(task)

    instruction = lifecycle.instruction_text()

    assert "Runtime instruction for this turn" in instruction
    assert "Determine whether the user task is complex" in instruction
    assert "create the next small atomic todo first" in instruction


def test_user_task_instruction_requires_todo_after_many_tool_calls():
    task = UserTask(title="Build feature")
    task.children = [
        ToolCallTask(title=f"Tool call {index}", tool_call_log_id=index)
        for index in range(6)
    ]
    lifecycle = UserTaskLifecycle(task)

    instruction = lifecycle.instruction_text()

    assert "More than 5 tool calls have run since the previous todo" in instruction
    assert "create a small atomic todo before doing more work" in instruction


def test_todo_task_instruction_focuses_active_todo_when_tool_count_is_small():
    task = TodoTask(title="Inspect files")
    lifecycle = TodoTaskLifecycle(task)

    instruction = lifecycle.instruction_text()

    assert "Focus on the active todo: Inspect files" in instruction
    assert "Call finish_todo immediately when it is complete" in instruction


def test_todo_task_instruction_prompts_finish_check_after_many_tool_calls():
    task = TodoTask(title="Inspect files")
    task.children = [
        ToolCallTask(title=f"Tool call {index}", tool_call_log_id=index)
        for index in range(11)
    ]
    lifecycle = TodoTaskLifecycle(task)

    instruction = lifecycle.instruction_text()

    assert "More than 10 tool calls have run for the active todo" in instruction
    assert "call finish_todo now with a concise result" in instruction


def test_tool_call_task_remains_data_only():
    task = ToolCallTask(title="Tool call 1", tool_call_log_id=1)

    assert not hasattr(task, "instruction_text")


def test_task_data_objects_do_not_expose_lifecycle_methods():
    user_task = UserTask(title="Build feature")
    todo = TodoTask(title="Inspect files")

    for task in [user_task, todo]:
        assert not hasattr(task, "create_tools")
        assert not hasattr(task, "sync")
        assert not hasattr(task, "append_tool_call_task")


def test_user_task_lifecycle_uses_owned_allocator_and_message_id():
    next_id = 10

    def allocate_task_id():
        nonlocal next_id
        task_id = next_id
        next_id += 1
        return task_id

    user_task = UserTask(id=1, title="Build feature")
    lifecycle = UserTaskLifecycle(user_task, allocate_task_id=allocate_task_id)
    lifecycle.current_assistant_message_id = 22

    todo = lifecycle.create_todo_task(title="Inspect files")
    tool_call = lifecycle.record_tool_call(7)

    assert todo.id == 10
    assert todo.start_message_id == 22
    assert tool_call.id == 11
    assert tool_call.parent_id == user_task.id


def test_todo_task_lifecycle_uses_owned_message_id_for_finish():
    todo = TodoTask(id=2, parent_id=1, title="Inspect files")
    lifecycle = TodoTaskLifecycle(todo, allocate_task_id=lambda: 3)
    lifecycle.current_assistant_message_id = 44

    lifecycle.finish_task(result="Inspected files")

    assert todo.status == "done"
    assert todo.end_message_id == 44


def test_lifecycle_tracks_next_task_transition():
    user_task = UserTask(id=1, title="Build feature")
    user_lifecycle = UserTaskLifecycle(user_task, allocate_task_id=lambda: 2)

    todo = user_lifecycle.create_todo_task(title="Inspect files")

    assert user_lifecycle.consume_next_task() is todo
    assert user_lifecycle.consume_next_task() is None

    todo_lifecycle = TodoTaskLifecycle(todo, user_task=user_task)
    todo_lifecycle.finish_task(result="Done")

    assert todo_lifecycle.consume_next_task() is user_task


def test_user_task_lifecycle_begins_compaction_only_when_done():
    user_task = UserTask(id=1, title="Build feature", status="active")
    user_task.children.append(TodoTask(id=2, parent_id=1, title="Inspect files", status="done"))
    lifecycle = UserTaskLifecycle(user_task, allocate_task_id=lambda: 3)

    assert lifecycle.begin_compaction() is False

    user_task.status = "done"

    assert lifecycle.begin_compaction() is True


def test_user_task_lifecycle_compaction_result_uses_user_task_boundaries():
    next_id = 10

    def allocate_task_id():
        nonlocal next_id
        task_id = next_id
        next_id += 1
        return task_id

    user_task = UserTask(
        id=1,
        title="Build feature",
        status="done",
        start_message_id=4,
        end_message_id=9,
        children=[ToolCallTask(id=2, parent_id=1, title="Tool call 7", status="done", tool_call_log_id=7)],
    )
    lifecycle = UserTaskLifecycle(user_task, allocate_task_id=allocate_task_id)
    assert lifecycle.begin_compaction() is True

    result = lifecycle.create_compacted_user_task(description="Summarized work")
    lifecycle.record_compacted_tool_call(tool_call_log_id=7)
    lifecycle.finish_compacted_user_task()

    start_message_id, end_message_id, messages = lifecycle.compaction_result()

    assert result is user_task
    assert user_task.result == "Summarized work"
    assert start_message_id == 4
    assert end_message_id == 9
    assert messages == [
        AssistantMessage(
            role="assistant",
            content=[TextContent(text="Compacted user task: Summarized work\nUseful tool calls: [7]")],
        )
    ]


def test_user_task_lifecycle_compaction_sync_replaces_user_task_children(tmp_path):
    db = _make_db(tmp_path)
    user_task = UserTask(id=1, title="Build feature", status="done", start_message_id=1, end_message_id=5)
    first_tool = ToolCallTask(id=2, parent_id=1, title="Tool call 10", status="done", tool_call_log_id=10)
    todo = TodoTask(id=3, parent_id=1, title="Inspect files", status="done", result="Done")
    user_task.children = [first_tool, todo]
    with db.create_session() as session:
        db.upsert_managed_task(user_task, session=session)
        db.upsert_managed_task(first_tool, session=session)
        db.upsert_managed_task(todo, session=session)
        session.commit()

    next_id = 20

    def allocate_task_id():
        nonlocal next_id
        task_id = next_id
        next_id += 1
        return task_id

    lifecycle = UserTaskLifecycle(user_task, allocate_task_id=allocate_task_id)
    assert lifecycle.begin_compaction() is True
    lifecycle.create_compacted_user_task(description="Whole task summary")
    lifecycle.record_compacted_tool_call(tool_call_log_id=10)
    lifecycle.finish_compacted_user_task()

    with db.create_session() as session:
        compacted = lifecycle.sync_compaction(db, session)
        session.commit()

    assert compacted is user_task
    assert db.get_managed_task(20) is not None
    assert db.get_managed_task(todo.id) is None
    loaded_children = db.list_managed_task_children(user_task.id)
    assert [child.tool_call_log_id for child in loaded_children] == [10]
    assert db.get_managed_task(user_task.id).result == "Whole task summary"


def test_user_task_lifecycle_compaction_requires_finished_compacted_user_task():
    user_task = UserTask(
        id=1,
        title="Build feature",
        status="done",
        start_message_id=1,
        end_message_id=2,
        children=[ToolCallTask(id=2, parent_id=1, title="Tool call 1", status="done", tool_call_log_id=1)],
    )
    lifecycle = UserTaskLifecycle(user_task, allocate_task_id=lambda: 3)
    assert lifecycle.begin_compaction() is True

    with pytest.raises(RuntimeError, match="No compacted user task result"):
        lifecycle.compaction_result()
