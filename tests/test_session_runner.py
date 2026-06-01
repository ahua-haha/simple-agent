"""Tests for SessionRunner."""

from __future__ import annotations

import asyncio

import pytest

from pi.ai.types import AssistantMessage, TextContent

from simple_agent.db.db import Database
from simple_agent.session.runner import SessionRunner
from simple_agent.task_manager import TaskManager
from simple_agent.tool.execution_logger import ToolExecutionLogger


class FakeAgentProcess:
    def __init__(self):
        self.calls = []

    async def run(self, system_prompt, messages, tools, user_prompt="", cancel_event=None):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "messages": messages,
                "tools": [tool.name for tool in tools],
                "user_prompt": user_prompt,
                "cancel_event": cancel_event,
            }
        )
        return [AssistantMessage(role="assistant", content=[TextContent(text="done")])]


@pytest.mark.asyncio
async def test_session_runner_creates_task_runs_agent_and_persists_messages(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    execution_logger = ToolExecutionLogger(db, task_manager=task_manager, session_id="session_a")
    agent_process = FakeAgentProcess()
    cancel_event = asyncio.Event()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=task_manager,
        execution_logger=execution_logger,
        agent_process=agent_process,
        cancel_event=cancel_event,
    )

    result = await runner.run("Build feature")

    assert result.kind == "user_task"
    assert result.title == "Build feature"
    assert result.status == "done"
    assert len(agent_process.calls) == 1
    assert agent_process.calls[0]["messages"] == []
    assert agent_process.calls[0]["user_prompt"] == "Build feature"
    assert agent_process.calls[0]["cancel_event"] is cancel_event
    assert "create_todo" in agent_process.calls[0]["tools"]
    assert "finish_todo" in agent_process.calls[0]["tools"]
    assert "error_todo" in agent_process.calls[0]["tools"]

    metadata = db.get_runner_state_metadata("session_a")
    assert metadata.phase == "done"
    assert metadata.status == "done"
    assert metadata.active_user_task_id == result.id
    assert db.list_runner_messages("session_a")[0].content[0].text == "done"


class FailingAgentProcess:
    async def run(self, system_prompt, messages, tools, user_prompt="", cancel_event=None):
        raise RuntimeError("agent failed")


@pytest.mark.asyncio
async def test_session_runner_persists_error_and_reraises(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    execution_logger = ToolExecutionLogger(db, task_manager=task_manager, session_id="session_a")
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=task_manager,
        execution_logger=execution_logger,
        agent_process=FailingAgentProcess(),
        cancel_event=asyncio.Event(),
    )

    with pytest.raises(RuntimeError, match="agent failed"):
        await runner.run("Build feature")

    metadata = db.get_runner_state_metadata("session_a")
    assert metadata.phase == "error"
    assert metadata.status == "error"
    assert metadata.last_error == "agent failed"
