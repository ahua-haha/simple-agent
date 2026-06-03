"""Tests for SessionRunner."""

from __future__ import annotations

import asyncio

import pytest

from pi.ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage

from simple_agent.db.db import Database
from simple_agent.fractional_index import key_after
from simple_agent.session.runner import SessionRunner
from simple_agent.state.state import RunnerMessageEntry
from simple_agent.task_manager import TaskManager


class FakeAgentProcess:
    def __init__(self):
        self.calls = []
        self.subscribers = []
        self.count_persisted = None
        self.persisted_after_turn = None
        self.observe_phase = None
        self.phase_at_run_start = None
        self.observe_metadata_phase = None
        self.metadata_phase_after_turn = None

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
        if self.observe_phase is not None:
            self.phase_at_run_start = self.observe_phase()
        from pi.agent.types import TurnEndEvent

        for hook in hooks["turn_end"]:
            hook(TurnEndEvent(message=message, tool_results=[tool_result]))
        if self.count_persisted is not None:
            self.persisted_after_turn = self.count_persisted()
        if self.observe_metadata_phase is not None:
            self.metadata_phase_after_turn = self.observe_metadata_phase()
        return [message, tool_result]

    def subscribe(self, callback):
        self.subscribers.append(callback)

    def unsubscribe(self, callback):
        if callback in self.subscribers:
            self.subscribers.remove(callback)


class FakeCompactAgentProcess(FakeAgentProcess):
    async def run(self, system_prompt, messages, tools, user_prompt="", cancel_event=None, hooks=None):
        self.calls.append({"tools": [tool.name for tool in tools], "messages": messages})
        by_name = {tool.name: tool for tool in tools}
        await by_name["create_compacted_todo"].execute("compact_create", {"description": "Compact summary"})
        await by_name["record_compacted_tool_call"].execute("compact_record", {"tool_call_log_id": 1})
        await by_name["finish_compacted_todo"].execute("compact_finish", {})
        return []


class FakePausedAgentProcess(FakeAgentProcess):
    def __init__(self):
        super().__init__()
        self.pause = None

    async def run(self, system_prompt, messages, tools, user_prompt="", cancel_event=None, hooks=None):
        self.calls.append({"user_prompt": user_prompt, "cancel_event": cancel_event})
        if len(self.calls) > 1:
            raise RuntimeError("runner should stop after paused agent run")
        self.pause()
        return []


def _message_entries(messages):
    seq = key_after(None)
    entries = []
    for message in messages:
        entries.append(RunnerMessageEntry(seq=seq, message=message))
        seq = key_after(seq)
    return entries


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
    agent_process.observe_phase = lambda: runner._phase
    agent_process.observe_metadata_phase = lambda: db.get_runner_state_metadata("session_a").phase

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
    assert agent_process.phase_at_run_start == "new_user_task"
    assert agent_process.metadata_phase_after_turn == "running"
    assert len(persisted_messages) == 2
    assert persisted_messages[0].content[0].text == "done"
    assert persisted_messages[1].content[0].text == "tool done"


@pytest.mark.asyncio
async def test_session_runner_run_after_done_starts_new_user_task(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=TaskManager(db),
        agent_process=FakeAgentProcess(),
        cancel_event=asyncio.Event(),
    )

    first = await runner.run("First request")
    second = await runner.run("Second request")

    assert first.title == "First request"
    assert second.title == "Second request"
    assert len(runner._agent_process.calls) == 2
    assert runner._agent_process.calls[1]["user_prompt"] == "Second request"


@pytest.mark.asyncio
async def test_session_runner_run_none_continues_running_task_without_user_prompt(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    task_manager.load(None)
    user_task = task_manager.create_user_task("Paused request")
    task_manager.save()
    db.upsert_runner_state_metadata(
        "session_a",
        phase="running",
        status="running",
        active_user_task_id=user_task.id,
    )
    agent_process = FakeAgentProcess()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=TaskManager(db),
        agent_process=agent_process,
        cancel_event=asyncio.Event(),
    )

    result = await runner.run(None)

    assert result.title == "Paused request"
    assert len(agent_process.calls) == 1
    assert agent_process.calls[0]["user_prompt"] is None


@pytest.mark.asyncio
async def test_session_runner_stops_after_paused_running_task(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    task_manager.load(None)
    user_task = task_manager.create_user_task("Paused request")
    task_manager.create_todo("Active todo")
    task_manager.save()
    db.upsert_runner_state_metadata(
        "session_a",
        phase="running",
        status="running",
        active_user_task_id=user_task.id,
    )
    agent_process = FakePausedAgentProcess()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=TaskManager(db),
        agent_process=agent_process,
        cancel_event=asyncio.Event(),
    )
    agent_process.pause = runner.pause

    result = await runner.run(None)

    metadata = db.get_runner_state_metadata("session_a")
    assert result.title == "Paused request"
    assert len(agent_process.calls) == 1
    assert metadata.phase == "running"


@pytest.mark.asyncio
async def test_session_runner_new_input_closes_previous_task_and_creates_new_task(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    task_manager.load(None)
    previous_task = task_manager.create_user_task("Previous request")
    task_manager.save()
    db.upsert_runner_state_metadata(
        "session_a",
        phase="running",
        status="running",
        active_user_task_id=previous_task.id,
    )
    agent_process = FakeAgentProcess()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=TaskManager(db),
        agent_process=agent_process,
        cancel_event=asyncio.Event(),
    )

    result = await runner.run("New request")

    reloaded_previous = db.get_managed_task(previous_task.id)
    assert reloaded_previous.status == "done"
    assert result.title == "New request"
    assert len(agent_process.calls) == 1
    assert agent_process.calls[0]["user_prompt"] == "New request"


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
async def test_turn_end_token_threshold_persists_compact_and_sets_cancel(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    cancel_event = asyncio.Event()
    task_manager = TaskManager(db)
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=task_manager,
        agent_process=FakeAgentProcess(),
        cancel_event=cancel_event,
        context_token_threshold=0,
        tool_call_threshold=100,
    )
    task_manager.load(None)
    user_task = task_manager.create_user_task("Build feature")
    runner._active_user_task_id = user_task.id
    runner._phase = "running"

    await runner.handle_running("Build feature")

    metadata = db.get_runner_state_metadata("session_a")
    assert metadata.phase == "compact"
    assert metadata.status == "compact"
    assert cancel_event.is_set() is True
    assert runner._phase == "compact"


@pytest.mark.asyncio
async def test_turn_end_tool_call_threshold_persists_compact_and_sets_cancel(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    cancel_event = asyncio.Event()
    task_manager = TaskManager(db)
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=task_manager,
        agent_process=FakeAgentProcess(),
        cancel_event=cancel_event,
        context_token_threshold=1000,
        tool_call_threshold=0,
    )
    task_manager.load(None)
    user_task = task_manager.create_user_task("Build feature")
    db.insert_runner_tool_call(
        session_id="session_a",
        tool_call_id="call_99",
        tool_name="example_tool",
        params={},
        result={"content": []},
        status="success",
        started_at=1.0,
        finished_at=2.0,
        error=None,
    )
    runner._active_user_task_id = user_task.id
    runner._phase = "running"

    await runner.handle_running("Build feature")

    metadata = db.get_runner_state_metadata("session_a")
    assert metadata.phase == "compact"
    assert metadata.status == "compact"
    assert cancel_event.is_set() is True
    assert runner._phase == "compact"


@pytest.mark.asyncio
async def test_handle_compact_without_finished_todo_returns_running(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    cancel_event = asyncio.Event()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=TaskManager(db),
        agent_process=FakeAgentProcess(),
        cancel_event=cancel_event,
    )
    runner._task_manager.load(None)
    user_task = runner._task_manager.create_user_task("Build feature")
    runner._task_manager.create_todo("Still active", tool_call_id="call_active")
    runner._active_user_task_id = user_task.id
    runner._phase = "compact"
    cancel_event.set()

    await runner.handle_compact("Build feature")

    metadata = db.get_runner_state_metadata("session_a")
    assert runner._phase == "running"
    assert metadata.phase == "running"
    assert cancel_event.is_set() is False


@pytest.mark.asyncio
async def test_handle_compact_replaces_messages_and_tasks(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    compact_agent = FakeCompactAgentProcess()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=task_manager,
        agent_process=compact_agent,
        cancel_event=asyncio.Event(),
    )
    task_manager.load(None)
    user_task = task_manager.create_user_task("Build feature")
    todo = task_manager.create_todo("Inspect files", tool_call_id="call_create")
    task_manager.finish_task("Done", tool_call_id="call_finish")
    active = task_manager.create_todo("Continue work")
    task_manager.save()
    runner._active_user_task_id = user_task.id
    runner._phase = "compact"
    runner._messages = _message_entries([
        UserMessage(content=[TextContent(text="original request")], timestamp=1),
        AssistantMessage(
            role="assistant",
            content=[
                TextContent(text="old todo"),
                ToolCall(id="call_create", name="create_todo", arguments={"title": todo.title}),
            ],
        ),
        ToolResultMessage(
            toolCallId="call_create",
            toolName="create_todo",
            content=[TextContent(text="created")],
        ),
        AssistantMessage(
            role="assistant",
            content=[
                TextContent(text="finish todo"),
                ToolCall(id="call_finish", name="finish_todo", arguments={"result": "Done"}),
            ],
        ),
        ToolResultMessage(
            toolCallId="call_finish",
            toolName="finish_todo",
            content=[TextContent(text="finished")],
        ),
        AssistantMessage(
            role="assistant",
            content=[
                TextContent(text="active todo"),
                ToolCall(id="call_active", name="create_todo", arguments={"title": active.title}),
            ],
        ),
    ])
    for entry in runner._messages:
        db.insert_runner_message_entry("session_a", entry)
    db.insert_runner_tool_call(
        session_id="session_a",
        tool_call_id="tool_1",
        tool_name="read",
        params={},
        result={"content": []},
        status="success",
        started_at=1.0,
        finished_at=2.0,
        error=None,
    )

    await runner.handle_compact("Build feature")

    messages = db.list_runner_messages("session_a")
    loaded_task_manager = TaskManager(db)
    loaded_task_manager.load(user_task.id)
    assert compact_agent.calls[0]["tools"] == [
        "create_compacted_todo",
        "record_compacted_tool_call",
        "finish_compacted_todo",
    ]
    assert len(compact_agent.calls[0]["messages"]) == 4
    assert compact_agent.calls[0]["messages"][0].content[1].id == "call_create"
    assert compact_agent.calls[0]["messages"][-1].tool_call_id == "call_finish"
    assert messages[0].content[0].text == "original request"
    assert "Compact summary" in messages[1].content[0].text
    assert "active todo" in messages[2].content[0].text
    assert len(loaded_task_manager.active_user_task.children) == 2
    assert db.get_runner_state_metadata("session_a").phase == "running"


@pytest.mark.asyncio
async def test_session_runner_finds_assistant_message_for_tool_call_id(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=TaskManager(db),
        agent_process=FakeAgentProcess(),
        cancel_event=asyncio.Event(),
    )
    runner._messages = _message_entries([
        UserMessage(content=[TextContent(text="request")], timestamp=1),
        AssistantMessage(role="assistant", content=[TextContent(text="thinking")]),
        AssistantMessage(
            role="assistant",
            content=[
                TextContent(text="create todo"),
                ToolCall(id="call_create", name="create_todo", arguments={"title": "Inspect files"}),
            ],
        ),
    ])

    assert runner.find_assistant_message_seq_for_tool_call("call_create") == 2


@pytest.mark.asyncio
async def test_session_runner_finds_tool_result_message_for_tool_call_id(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=TaskManager(db),
        agent_process=FakeAgentProcess(),
        cancel_event=asyncio.Event(),
    )
    runner._messages = _message_entries([
        UserMessage(content=[TextContent(text="request")], timestamp=1),
        AssistantMessage(
            role="assistant",
            content=[
                TextContent(text="finish todo"),
                ToolCall(id="call_finish", name="finish_todo", arguments={}),
            ],
        ),
        ToolResultMessage(
            toolCallId="call_finish",
            toolName="finish_todo",
            content=[TextContent(text="finished")],
        ),
    ])

    assert runner.find_tool_result_message_seq_for_tool_call("call_finish") == 2


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
