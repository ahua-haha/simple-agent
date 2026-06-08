"""Tests for next-task builder tools."""

from __future__ import annotations

import pytest

from simple_agent.task_manager.models import RepoMemoryTask, TodoTask, UserTask
from simple_agent.task_manager.lifecycle import SessionState
from simple_agent.task_manager.task_builder import NextTaskBuilder


def test_next_task_builder_instruction_describes_supported_tasks():
    builder = NextTaskBuilder(SessionState(messages=[]))

    instruction = builder.instruction_text()

    assert "create_next_task" in instruction
    assert "todo" in instruction
    assert "repo_memory" in instruction


def test_next_task_builder_can_limit_enabled_task_set():
    builder = NextTaskBuilder(SessionState(messages=[]), enabled_task_kinds=["repo_memory"])

    instruction = builder.instruction_text()
    tool = builder.create_task_tool()

    assert "repo_memory" in instruction
    assert "kind=todo" not in instruction
    assert tool.parameters["properties"]["kind"]["enum"] == ["repo_memory"]


@pytest.mark.asyncio
async def test_next_task_builder_create_task_tool_creates_todo_task():
    parent = UserTask(id=1, title="Build feature")
    session_state = SessionState(
        messages=[],
        next_task=parent,
        next_task_id_to_run=parent.id,
        next_task_id_to_allocate=2,
    )
    tool = NextTaskBuilder(session_state).create_task_tool()

    result = await tool.execute("call_1", {"kind": "todo", "title": "Inspect files"})

    todo = parent.children[0]
    assert isinstance(todo, TodoTask)
    assert todo.id == 2
    assert todo.parent_id == parent.id
    assert todo.title == "Inspect files"
    assert session_state.next_task is todo
    assert session_state.next_task_id_to_run == todo.id
    assert session_state.next_task_id_to_allocate == 3
    assert result.content[0].text == "Created next task: todo Inspect files"


@pytest.mark.asyncio
async def test_next_task_builder_sets_todo_start_message_id_from_runtime_callback():
    parent = UserTask(id=1, title="Build feature")
    current_message_id = 42
    session_state = SessionState(
        messages=[],
        next_task=parent,
        next_task_id_to_run=parent.id,
        next_task_id_to_allocate=2,
    )
    tool = NextTaskBuilder(
        session_state,
        current_assistant_message_id=lambda: current_message_id,
    ).create_task_tool()

    await tool.execute("call_1", {"kind": "todo", "title": "Inspect files"})

    todo = parent.children[0]
    assert isinstance(todo, TodoTask)
    assert todo.start_message_id == 42


@pytest.mark.asyncio
async def test_next_task_builder_create_task_tool_creates_repo_memory_task():
    parent = UserTask(id=1, title="Build feature")
    session_state = SessionState(
        messages=[],
        next_task=parent,
        next_task_id_to_run=parent.id,
        next_task_id_to_allocate=5,
    )
    tool = NextTaskBuilder(session_state).create_task_tool()

    await tool.execute(
        "call_1",
        {
            "kind": "repo_memory",
            "title": "Write repo memory",
            "metadata": {
                "repo_path": "/repo",
                "index_db_path": "/repo/index.db",
            },
        },
    )

    task = parent.children[0]
    assert isinstance(task, RepoMemoryTask)
    assert task.id == 5
    assert task.parent_id == parent.id
    assert task.repo_path == "/repo"
    assert task.index_db_path == "/repo/index.db"
    assert session_state.next_task is task
    assert session_state.next_task_id_to_run == task.id


@pytest.mark.asyncio
async def test_next_task_builder_rejects_disabled_task_kind():
    parent = UserTask(id=1, title="Build feature")
    session_state = SessionState(
        messages=[],
        next_task=parent,
        next_task_id_to_run=parent.id,
        next_task_id_to_allocate=5,
    )
    tool = NextTaskBuilder(session_state, enabled_task_kinds=["todo"]).create_task_tool()

    with pytest.raises(Exception, match="disabled"):
        await tool.execute(
            "call_1",
            {
                "kind": "repo_memory",
                "title": "Write repo memory",
                "metadata": {
                    "repo_path": "/repo",
                    "index_db_path": "/repo/index.db",
                },
            },
        )
