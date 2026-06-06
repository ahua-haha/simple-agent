"""Tests for Session."""

from __future__ import annotations

import os

import pytest

from simple_agent.session import Session


class TestSessionInit:
    """Tests for Session initialization."""

    def test_new_session_has_id(self, tmp_path):
        session = Session(base_dir=str(tmp_path))
        assert session.id.startswith("session_")

    def test_existing_session_uses_provided_id(self, tmp_path):
        session = Session(session_id="test", base_dir=str(tmp_path))
        assert session.id == "test"

    def test_new_session_initializes_runner(self, tmp_path):
        session = Session(base_dir=str(tmp_path))
        assert session._runner is not None


class TestSessionManagerList:
    """Tests for session listing via SessionManager."""

    def test_empty_directory(self, tmp_path):
        from simple_agent.session.session_manager import SessionManager
        sm = SessionManager(sessions_dir=str(tmp_path))
        sessions = sm.list()
        assert sessions == []

    def test_lists_db_files(self, tmp_path):
        from simple_agent.session.session_manager import SessionManager
        for name in ["a", "b"]:
            open(os.path.join(str(tmp_path), f"{name}.db"), "w").close()

        sm = SessionManager(sessions_dir=str(tmp_path))
        sessions = sm.list()
        ids = sorted(s["id"] for s in sessions)
        assert ids == ["a", "b"]

    def test_ignores_non_db(self, tmp_path):
        from simple_agent.session.session_manager import SessionManager
        open(os.path.join(str(tmp_path), "test.db"), "w").close()
        with open(os.path.join(str(tmp_path), "notes.txt"), "w") as f:
            f.write("hello")

        sm = SessionManager(sessions_dir=str(tmp_path))
        sessions = sm.list()
        ids = [s["id"] for s in sessions]
        assert ids == ["test"]


class TestSessionEventQueue:
    """Tests for Session event queue lifecycle."""

    @pytest.mark.asyncio
    async def test_queue_created_in_run(self, tmp_path):
        session = Session(base_dir=str(tmp_path))

        async def fake_run(user_input):
            return None

        session._runner.run = fake_run

        queue = session.run("hello")

        assert session._run_task is not None
        await queue.get()
        assert session._run_task is None

    @pytest.mark.asyncio
    async def test_queue_none_after_run(self, tmp_path):
        session = Session(base_dir=str(tmp_path))

        assert session._run_task is None

    @pytest.mark.asyncio
    async def test_agent_event_pushed_to_queue(self, tmp_path):
        import asyncio
        session = Session(base_dir=str(tmp_path))

        async def fake_run(user_input):
            from pi.agent.types import AgentEndEvent
            from pi.ai.types import AssistantMessage, TextContent
            msg = AssistantMessage(role="assistant", content=[TextContent(text="hello")])
            event = AgentEndEvent(messages=[msg])
            session._agent_process._emit(event)

        session._runner.run = fake_run

        queue = session.run("hello")
        received = await queue.get()

        assert received is not None


def test_session_pause_delegates_to_runner(tmp_path):
    session = Session(base_dir=str(tmp_path))

    session.pause()

    assert session._runner._cancel_event.is_set()


def test_session_initializes_task_manager(tmp_path):
    from simple_agent.session.session import Session

    session = Session(base_dir=str(tmp_path))

    assert session._task_manager is not None
    assert session._runner is not None


@pytest.mark.asyncio
async def test_session_run_creates_queue_and_runs_agent_once(tmp_path, monkeypatch):
    from simple_agent.session.session import Session

    calls = []

    async def fake_call_llm_step(self, system_prompt, messages, tools, cancel_event=None):
        from pi.ai.types import AssistantMessage, TextContent, ToolCall
        calls.append(
            {
                "system_prompt": system_prompt,
                "messages": messages,
                "tools": [tool.name for tool in tools],
                "cancel_event": cancel_event,
            }
        )
        if len(calls) > 1:
            return AssistantMessage(
                role="assistant",
                content=[TextContent(text="final answer")],
            )
        return AssistantMessage(
            role="assistant",
            content=[
                TextContent(text="done"),
                ToolCall(id="tool_1", name="example_tool", arguments={}),
            ],
        )

    async def fake_run_tool_calls_step(self, tools, assistant_message, cancel_event=None):
        from pi.ai.types import TextContent, ToolResultMessage
        return [
            ToolResultMessage(
                toolCallId="tool_1",
                toolName="example_tool",
                content=[TextContent(text="tool done")],
            )
        ]

    monkeypatch.setattr("simple_agent.process.agent_process.AgentProcess.call_llm_step", fake_call_llm_step)
    monkeypatch.setattr("simple_agent.process.agent_process.AgentProcess.run_tool_calls_step", fake_run_tool_calls_step)

    session = Session(base_dir=str(tmp_path))
    queue = session.run("Build feature")
    assert isinstance(queue, __import__("asyncio").Queue)

    await queue.get()

    assert len(calls) == 3
    assert "create_todo" in calls[0]["tools"]
    assert "finish_user_task" in calls[0]["tools"]
    assert "finish_todo" not in calls[0]["tools"]
    assert "error_todo" not in calls[0]["tools"]
    assert calls[2]["tools"] == [
        "create_compacted_user_task",
        "record_compacted_tool_call",
        "finish_compacted_user_task",
    ]
    assert calls[0]["cancel_event"] is session._runner._cancel_event
