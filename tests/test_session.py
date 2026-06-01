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
        assert session.event_queue is None

        # Start run and cancel immediately to avoid actual agent execution
        session._running = True
        session.event_queue = __import__("asyncio").Queue()
        session.event_queue.put_nowait(None)

        assert session.event_queue is not None

    @pytest.mark.asyncio
    async def test_queue_none_after_run(self, tmp_path):
        session = Session(base_dir=str(tmp_path))
        session.event_queue = __import__("asyncio").Queue()
        session.event_queue.put_nowait(None)
        session.event_queue = None

        assert session.event_queue is None

    @pytest.mark.asyncio
    async def test_agent_event_pushed_to_queue(self, tmp_path):
        import asyncio
        session = Session(base_dir=str(tmp_path))
        session.event_queue = asyncio.Queue()

        from pi.agent.types import AgentEndEvent
        from pi.ai.types import AssistantMessage, TextContent
        msg = AssistantMessage(role="assistant", content=[TextContent(text="hello")])
        event = AgentEndEvent(messages=[msg])
        session._on_agent_event(event)
        received = session.event_queue.get_nowait()
        assert received is event

    @pytest.mark.asyncio
    async def test_no_push_when_queue_is_none(self, tmp_path):
        session = Session(base_dir=str(tmp_path))
        assert session.event_queue is None

        # Should not raise
        from pi.agent.types import AgentEndEvent
        from pi.ai.types import AssistantMessage, TextContent
        msg = AssistantMessage(role="assistant", content=[TextContent(text="hello")])
        event = AgentEndEvent(messages=[msg])
        session._on_agent_event(event)


def test_session_pause_delegates_to_runner(tmp_path):
    session = Session(base_dir=str(tmp_path))

    session.pause()

    assert session._runner._cancel_event.is_set()


def test_session_initializes_task_manager(tmp_path):
    from simple_agent.session.session import Session

    session = Session(base_dir=str(tmp_path))

    assert session._task_manager is not None
    assert session._execution_logger is not None
    assert session._runner is not None


@pytest.mark.asyncio
async def test_session_run_creates_user_task_and_calls_agent_once(tmp_path, monkeypatch):
    from simple_agent.session.session import Session

    calls = []

    async def fake_run(self, system_prompt, messages, tools, user_prompt="", cancel_event=None):
        calls.append(
            {
                "system_prompt": system_prompt,
                "messages": messages,
                "tools": [tool.name for tool in tools],
                "user_prompt": user_prompt,
                "cancel_event": cancel_event,
            }
        )
        from pi.ai.types import AssistantMessage, TextContent
        return [AssistantMessage(role="assistant", content=[TextContent(text="done")])]

    monkeypatch.setattr("simple_agent.process.agent_process.AgentProcess.run", fake_run)

    session = Session(base_dir=str(tmp_path))
    result = await session.run("Build feature")

    assert result is not None
    assert result.kind == "user_task"
    assert result.title == "Build feature"
    assert len(calls) == 1
    assert "create_todo" in calls[0]["tools"]
    assert "finish_todo" in calls[0]["tools"]
    assert "error_todo" in calls[0]["tools"]
    assert calls[0]["cancel_event"] is session._runner._cancel_event
