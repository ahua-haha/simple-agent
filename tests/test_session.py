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


def test_session_initializes_runner_without_task_manager(tmp_path):
    from simple_agent.session.session import Session

    session = Session(base_dir=str(tmp_path))

    assert not hasattr(session, "_task_manager")
    assert session._runner is not None


def test_session_runner_input_transition_creates_user_task_in_memory(tmp_path):
    from simple_agent.session.session import Session
    from pi.ai.types import UserMessage
    from simple_agent.task_manager.models import UserTask

    session = Session(base_dir=str(tmp_path))
    runner = session._runner
    runner.load()

    runner.run_input_transition("Build feature")

    task = runner._runtime.next_task
    assert isinstance(task, UserTask)
    assert runner.user_task is task
    assert task.id == 1
    assert task.title == "Build feature"
    assert task.start_message_id == 1
    assert runner._runtime.next_task_id_to_run == task.id
    assert runner._runtime.next_task_id_to_allocate == 2

    assert len(runner._runtime.messages) == 1
    message_entry = runner._runtime.messages[0]
    assert message_entry.id == 1
    assert isinstance(message_entry.message, UserMessage)
    assert message_entry.message.content[0].text == "Build feature"
    assert runner._runtime.next_message_id == 2

    runner.run_input_transition("Second task")

    assert runner._runtime.next_task is task
    assert runner._runtime.next_task_id_to_run == task.id
    assert [entry.message.content[0].text for entry in runner._runtime.messages] == ["Build feature"]
    assert session._db.list_runner_messages(session.id) == []
    assert session._db.get_managed_task(task.id) is None
