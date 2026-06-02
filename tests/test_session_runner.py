"""Tests for SessionRunner."""

from __future__ import annotations

import asyncio

import pytest

from pi.ai.types import AssistantMessage, TextContent, ToolResultMessage

from simple_agent.db.db import Database
from simple_agent.session.runner import SessionRunner
from simple_agent.task_manager import TaskManager


class FakeAgentProcess:
    def __init__(self):
        self.calls = []
        self.subscribers = []
        self.count_persisted = None
        self.persisted_after_turn = None

    async def run(self, system_prompt, messages, tools, user_prompt="", cancel_event=None, hooks=None):
        message = AssistantMessage(role="assistant", content=[TextContent(text="done")])
        tool_result = ToolResultMessage(
            toolCallId="tool_1",
            toolName="example_tool",
            content=[TextContent(text="tool done")],
            timestamp=1,
        )
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "messages": messages,
                "tools": [tool.name for tool in tools],
                "user_prompt": user_prompt,
                "cancel_event": cancel_event,
                "hooks": hooks,
            }
        )
        from pi.agent.types import TurnEndEvent

        for hook in hooks["turn_end"]:
            hook(TurnEndEvent(message=message, tool_results=[tool_result]))
        if self.count_persisted is not None:
            self.persisted_after_turn = self.count_persisted()
        return [message, tool_result]

    def subscribe(self, callback):
        self.subscribers.append(callback)

    def unsubscribe(self, callback):
        if callback in self.subscribers:
            self.subscribers.remove(callback)


@pytest.mark.asyncio
async def test_session_runner_creates_task_runs_agent_and_persists_messages(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    agent_process = FakeAgentProcess()
    cancel_event = asyncio.Event()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=task_manager,
        agent_process=agent_process,
        cancel_event=cancel_event,
    )
    agent_process.count_persisted = lambda: len(db.list_runner_messages("session_a"))

    result = await runner.run("Build feature")

    assert result.kind == "user_task"
    assert result.title == "Build feature"
    assert result.status == "done"
    assert len(agent_process.calls) == 1
    assert agent_process.calls[0]["messages"] == []
    assert agent_process.calls[0]["messages"] is not runner._messages
    assert agent_process.calls[0]["user_prompt"] == "Build feature"
    assert agent_process.calls[0]["cancel_event"] is cancel_event
    assert set(agent_process.calls[0]["hooks"]) == {"turn_end"}
    assert "create_todo" in agent_process.calls[0]["tools"]
    assert "finish_todo" in agent_process.calls[0]["tools"]
    assert "error_todo" in agent_process.calls[0]["tools"]

    metadata = db.get_runner_state_metadata("session_a")
    assert metadata.phase == "done"
    assert metadata.status == "done"
    assert metadata.active_user_task_id == result.id
    persisted_messages = db.list_runner_messages("session_a")
    assert agent_process.persisted_after_turn == 2
    assert len(persisted_messages) == 2
    assert persisted_messages[0].content[0].text == "done"
    assert persisted_messages[1].content[0].text == "tool done"


class FailingAgentProcess:
    async def run(self, system_prompt, messages, tools, user_prompt="", cancel_event=None, hooks=None):
        raise RuntimeError("agent failed")

    def subscribe(self, callback):
        pass


class FailingSaveTaskManager(TaskManager):
    def __init__(self, db):
        super().__init__(db)
        self.save_calls = 0

    def save(self, session=None):
        self.save_calls += 1
        if self.save_calls > 1:
            raise RuntimeError("task save failed")
        return super().save(session=session)


def test_session_runner_subscribe_delegates_to_agent_process(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    agent_process = FakeAgentProcess()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=TaskManager(db),
        agent_process=agent_process,
        cancel_event=asyncio.Event(),
    )

    def callback(event):
        return event

    runner.subscribe(callback)

    assert agent_process.subscribers == [callback]

    runner.unsubscribe(callback)

    assert agent_process.subscribers == []


def test_session_runner_pause_controls_cancel_event(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    cancel_event = asyncio.Event()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=TaskManager(db),
        agent_process=FakeAgentProcess(),
        cancel_event=cancel_event,
    )

    runner.pause()
    assert cancel_event.is_set()


@pytest.mark.asyncio
async def test_session_runner_persists_error_and_reraises(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=task_manager,
        agent_process=FailingAgentProcess(),
        cancel_event=asyncio.Event(),
    )

    with pytest.raises(RuntimeError, match="agent failed"):
        await runner.run("Build feature")

    metadata = db.get_runner_state_metadata("session_a")
    assert metadata.phase == "error"
    assert metadata.status == "error"
    assert metadata.last_error == "agent failed"


@pytest.mark.asyncio
async def test_turn_end_save_rolls_back_messages_when_task_save_fails(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = FailingSaveTaskManager(db)
    agent_process = FakeAgentProcess()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=task_manager,
        agent_process=agent_process,
        cancel_event=asyncio.Event(),
    )

    with pytest.raises(RuntimeError, match="task save failed"):
        await runner.run("Build feature")

    assert db.list_runner_messages("session_a") == []
