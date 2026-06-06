from simple_agent.task_manager.models import (
    TaskRuntimeContext,
    TodoTask,
    ToolCallTask,
    UserTask,
)


def _context(*, active_task_tool_calls: int) -> TaskRuntimeContext:
    return TaskRuntimeContext(
        session_id="session_a",
        context_tokens=100,
        total_tool_calls=active_task_tool_calls,
        active_task_tool_calls=active_task_tool_calls,
    )


def test_user_task_instruction_asks_for_complexity_check_when_tool_count_is_small():
    task = UserTask(title="Build feature")

    instruction = task.instruction_text(_context(active_task_tool_calls=2))

    assert "Runtime instruction for this turn" in instruction
    assert "Determine whether the user task is complex" in instruction
    assert "create the next small atomic todo first" in instruction


def test_user_task_instruction_requires_todo_after_many_tool_calls():
    task = UserTask(title="Build feature")

    instruction = task.instruction_text(_context(active_task_tool_calls=6))

    assert "More than 5 tool calls have run since the previous todo" in instruction
    assert "create a small atomic todo before doing more work" in instruction


def test_todo_task_instruction_focuses_active_todo_when_tool_count_is_small():
    task = TodoTask(title="Inspect files")

    instruction = task.instruction_text(_context(active_task_tool_calls=3))

    assert "Focus on the active todo: Inspect files" in instruction
    assert "Call finish_todo immediately when it is complete" in instruction


def test_todo_task_instruction_prompts_finish_check_after_many_tool_calls():
    task = TodoTask(title="Inspect files")

    instruction = task.instruction_text(_context(active_task_tool_calls=11))

    assert "More than 10 tool calls have run for the active todo" in instruction
    assert "call finish_todo now with a concise result" in instruction


def test_tool_call_task_remains_data_only():
    task = ToolCallTask(title="Tool call 1", tool_call_log_id=1)

    assert not hasattr(task, "instruction_text")
