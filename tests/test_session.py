"""Tests for Session."""

from __future__ import annotations

import os

import pytest

from simple_agent.session import Session


class TestSessionInit:
    """Tests for Session initialization."""

    def test_new_session_has_id(self, tmp_path):
        session = Session(sessions_dir=str(tmp_path), workspace_dir=os.getcwd())
        assert session.id.startswith("session_")

    def test_existing_session_uses_provided_id(self, tmp_path):
        Session(session_id="test", sessions_dir=str(tmp_path), workspace_dir=os.getcwd())
        session = Session(session_id="test", sessions_dir=str(tmp_path))
        assert session.id == "test"
        assert session._runner._session_state.workspace_dir == os.getcwd()

    def test_new_session_initializes_runner(self, tmp_path):
        session = Session(sessions_dir=str(tmp_path), workspace_dir=os.getcwd())
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
        session = Session(sessions_dir=str(tmp_path), workspace_dir=os.getcwd())

        async def fake_run(user_input):
            return None

        session._runner.run = fake_run

        queue = session.run("hello")

        assert session._run_task is not None
        await queue.get()
        assert session._run_task is None

    @pytest.mark.asyncio
    async def test_queue_none_after_run(self, tmp_path):
        session = Session(sessions_dir=str(tmp_path), workspace_dir=os.getcwd())

        assert session._run_task is None

    @pytest.mark.asyncio
    async def test_agent_event_pushed_to_queue(self, tmp_path):
        import asyncio
        session = Session(sessions_dir=str(tmp_path), workspace_dir=os.getcwd())

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

    @pytest.mark.asyncio
    async def test_stop_pauses_and_waits_for_run_task(self, tmp_path):
        import asyncio
        session = Session(sessions_dir=str(tmp_path), workspace_dir=os.getcwd())
        stopped = asyncio.Event()

        async def fake_run(user_input):
            try:
                while not session._runner._cancel_event.is_set():
                    await asyncio.sleep(0.01)
            finally:
                stopped.set()
            return None

        session._runner.run = fake_run

        queue = session.run("hello")
        await session.stop(timeout=1.0)

        assert stopped.is_set()
        assert not session.is_running
        assert session._run_task is None
        assert await queue.get() is None

    @pytest.mark.asyncio
    async def test_stop_cancels_run_task_after_timeout(self, tmp_path):
        import asyncio
        session = Session(sessions_dir=str(tmp_path), workspace_dir=os.getcwd())
        cancelled = asyncio.Event()

        async def fake_run(user_input):
            try:
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        session._runner.run = fake_run

        session.run("hello")
        await session.stop(timeout=0.01)

        assert cancelled.is_set()
        assert not session.is_running

    @pytest.mark.asyncio
    async def test_stop_with_zero_timeout_cancels_run_task_immediately(self, tmp_path):
        import asyncio
        session = Session(sessions_dir=str(tmp_path), workspace_dir=os.getcwd())
        cancelled = asyncio.Event()

        async def fake_run(user_input):
            try:
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        session._runner.run = fake_run

        session.run("hello")
        await asyncio.sleep(0)
        await session.stop(timeout=0.0)

        assert cancelled.is_set()
        assert not session.is_running


def test_session_pause_delegates_to_runner(tmp_path):
    session = Session(sessions_dir=str(tmp_path), workspace_dir=os.getcwd())

    session.pause()

    assert session._runner._cancel_event.is_set()


def test_session_initializes_runner_without_task_manager(tmp_path):
    from simple_agent.session.session import Session

    session = Session(sessions_dir=str(tmp_path), workspace_dir=os.getcwd())

    assert not hasattr(session, "_task_manager")
    assert session._runner is not None


def test_session_runner_input_transition_creates_and_persists_user_task(tmp_path):
    from simple_agent.session.session import Session
    from pi.ai.types import UserMessage
    from simple_agent.task_manager.models import CommonTask

    session = Session(sessions_dir=str(tmp_path), workspace_dir=os.getcwd())
    runner = session._runner
    runner.load()

    assert runner._session_state.workspace_dir == os.getcwd()

    runner.run_input_transition("Build feature")

    task = runner._session_state.next_task
    assert isinstance(task, CommonTask)
    assert runner.user_task is task
    assert task.id == 1
    assert task.title == "Build feature"
    assert task.start_message_id == 1
    assert runner._session_state.next_task_id_to_run == task.id
    assert runner._session_state.next_task_id_to_allocate == 2

    assert len(runner._session_state.messages) == 1
    message_entry = runner._session_state.messages[0]
    assert message_entry.id == 1
    assert isinstance(message_entry.message, UserMessage)
    assert message_entry.message.content[0].text == "Build feature"
    assert runner._session_state.next_message_id == 2

    runner.run_input_transition("Second task")

    assert runner._session_state.next_task is task
    assert runner._session_state.next_task_id_to_run == task.id
    assert [entry.message.content[0].text for entry in runner._session_state.messages] == ["Build feature"]
    persisted_messages = session._db.list_runner_messages(session.id)
    assert len(persisted_messages) == 1
    assert isinstance(persisted_messages[0], UserMessage)
    assert persisted_messages[0].content[0].text == "Build feature"

    persisted_task = session._db.get_managed_task(task.id)
    assert isinstance(persisted_task, CommonTask)
    assert persisted_task.start_message_id == message_entry.id


@pytest.mark.asyncio
async def test_session_runner_does_not_resolve_next_task_after_lifecycle_run(tmp_path):
    from simple_agent.session.session import Session
    from simple_agent.task_manager.models import CommonTask

    class ShiftLifecycle:
        def set_data(self, session_state):
            self.session_state = session_state

        async def run(self, *, agent_process, cancel_event=None):
            self.session_state.next_task_id_to_run = 999
            self.session_state.next_task = None
            return self.session_state

        def clear_data(self):
            pass

    session = Session(sessions_dir=str(tmp_path), workspace_dir=os.getcwd())
    runner = session._runner
    runner.load()
    task = CommonTask(id=1, title="Build feature")
    runner._session_state.next_task_id_to_run = task.id
    runner._session_state.next_task = task
    runner._lifecycles["user_task"] = ShiftLifecycle()

    result = await runner.run_active_lifecycle()

    assert result is runner._session_state
    assert runner._session_state.next_task_id_to_run == 999
    assert runner._session_state.next_task is None


@pytest.mark.asyncio
async def test_session_runner_continues_when_lifecycle_sets_next_task_id_without_instance(tmp_path):
    from simple_agent.session.session import Session
    from simple_agent.task_manager.models import CommonTask, RepoMemoryTask

    class ChildLifecycle:
        def __init__(self, parent_id):
            self.parent_id = parent_id

        def set_data(self, session_state):
            self.session_state = session_state

        async def run(self, *, agent_process, cancel_event=None):
            self.session_state.next_task_id_to_run = self.parent_id
            self.session_state.next_task = None
            return self.session_state

        def clear_data(self):
            pass

    class ParentLifecycle:
        def __init__(self):
            self.run_count = 0

        def set_data(self, session_state):
            self.session_state = session_state

        async def run(self, *, agent_process, cancel_event=None):
            self.run_count += 1
            self.session_state.next_task_id_to_run = None
            self.session_state.next_task = None
            return self.session_state

        def clear_data(self):
            pass

    session = Session(sessions_dir=str(tmp_path), workspace_dir=os.getcwd())
    runner = session._runner
    parent = CommonTask(id=1, title="Build feature")
    child = RepoMemoryTask(id=2, parent_id=parent.id, title="Inspect files", index_db_path=".agent-index.db")
    with session._db.create_session() as db_session:
        session._db.upsert_managed_task(parent, session=db_session)
        session._db.upsert_managed_task(child, session=db_session)
        db_session.commit()

    runner.load = lambda: None
    runner._session_state.next_task_id_to_run = child.id
    runner._session_state.next_task = child
    parent_lifecycle = ParentLifecycle()
    runner._lifecycles["repo_memory"] = ChildLifecycle(parent.id)
    runner._lifecycles["user_task"] = parent_lifecycle

    await runner.run(None)

    assert parent_lifecycle.run_count == 1
