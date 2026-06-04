"""Tests for SessionRunner."""

from __future__ import annotations

import asyncio

import pytest

from pi.ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage

from simple_agent.db.db import Database
from simple_agent.session.runner import SessionRunner
from simple_agent.task_manager import TaskManager


class FakeAgentProcess:
    def __init__(self):
        self.calls = []
        self.tool_step_calls = []
        self.subscribers = []
        self.count_persisted = None
        self.persisted_after_turn = None
        self.observe_next_action = None
        self.next_action_at_run_start = None
        self.observe_metadata_next_action = None
        self.metadata_next_action_after_turn = None
        self.pause_on_tool_step = None

    async def call_llm_step(self, system_prompt, messages, tools, cancel_event=None):
        message = AssistantMessage(
            role="assistant",
            content=[
                TextContent(text="done"),
                ToolCall(id="tool_1", name="example_tool", arguments={}),
            ],
        )
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "messages": messages,
                "tools": [tool.name for tool in tools],
                "cancel_event": cancel_event,
            }
        )
        if self.observe_next_action is not None:
            self.next_action_at_run_start = self.observe_next_action()
        return message

    async def run_tool_calls_step(self, tools, assistant_message, cancel_event=None):
        tool_result = ToolResultMessage(
            toolCallId="tool_1",
            toolName="example_tool",
            content=[TextContent(text="tool done")],
            timestamp=1,
        )
        self.tool_step_calls.append(
            {
                "tools": [tool.name for tool in tools],
                "assistant_message": assistant_message,
                "cancel_event": cancel_event,
            }
        )
        if self.pause_on_tool_step is not None:
            self.pause_on_tool_step()
        if self.count_persisted is not None:
            self.persisted_after_turn = self.count_persisted()
        if self.observe_metadata_next_action is not None:
            self.metadata_next_action_after_turn = self.observe_metadata_next_action()
        return [tool_result]

    def subscribe(self, callback):
        self.subscribers.append(callback)

    def unsubscribe(self, callback):
        if callback in self.subscribers:
            self.subscribers.remove(callback)


class FakeFinalAgentProcess(FakeAgentProcess):
    async def call_llm_step(self, system_prompt, messages, tools, cancel_event=None):
        message = AssistantMessage(role="assistant", content=[TextContent(text="final answer")])
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "messages": messages,
                "tools": [tool.name for tool in tools],
                "cancel_event": cancel_event,
            }
        )
        if self.observe_next_action is not None:
            self.next_action_at_run_start = self.observe_next_action()
        return message

    async def run_tool_calls_step(self, tools, assistant_message, cancel_event=None):
        raise AssertionError("tool step should not run for a final assistant response")


class FakeToolThenFinalAgentProcess(FakeAgentProcess):
    async def call_llm_step(self, system_prompt, messages, tools, cancel_event=None):
        if not self.calls:
            return await super().call_llm_step(system_prompt, messages, tools, cancel_event=cancel_event)
        message = AssistantMessage(role="assistant", content=[TextContent(text="final answer")])
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "messages": messages,
                "tools": [tool.name for tool in tools],
                "cancel_event": cancel_event,
            }
        )
        return message


class FakeCompactAgentProcess(FakeAgentProcess):
    async def run(self, system_prompt, messages, tools, user_prompt="", cancel_event=None):
        self.calls.append({"tools": [tool.name for tool in tools], "messages": messages})
        by_name = {tool.name: tool for tool in tools}
        await by_name["create_compacted_todo"].execute("compact_create", {"description": "Compact summary"})
        await by_name["record_compacted_tool_call"].execute("compact_record", {"tool_call_log_id": 1})
        await by_name["finish_compacted_todo"].execute("compact_finish", {})
        return []


@pytest.mark.asyncio
async def test_session_runner_creates_task_runs_agent_and_persists_messages(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    agent_process = FakeFinalAgentProcess()
    cancel_event = asyncio.Event()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=task_manager,
        agent_process=agent_process,
        cancel_event=cancel_event,
    )
    agent_process.count_persisted = lambda: len(db.list_runner_messages("session_a"))
    agent_process.observe_next_action = lambda: runner._next_action
    agent_process.observe_metadata_next_action = lambda: db.get_runner_state_metadata("session_a").next_action

    result = await runner.run("Build feature")

    assert result.kind == "user_task"
    assert result.title == "Build feature"
    assert result.status == "done"
    assert len(agent_process.calls) == 1
    assert agent_process.calls[0]["messages"][0].content[0].text == "Build feature"
    assert agent_process.calls[0]["messages"] is not runner._messages
    assert agent_process.calls[0]["cancel_event"] is cancel_event
    assert "create_todo" in agent_process.calls[0]["tools"]
    assert "finish_todo" in agent_process.calls[0]["tools"]
    assert "error_todo" in agent_process.calls[0]["tools"]
    assert len(agent_process.tool_step_calls) == 0

    metadata = db.get_runner_state_metadata("session_a")
    assert metadata.next_action == "wait_user_input"
    assert metadata.active_user_task_id == result.id
    persisted_messages = db.list_runner_messages("session_a")
    tool_calls = db.list_runner_tool_calls("session_a")
    assert agent_process.next_action_at_run_start == "normal_run"
    assert len(persisted_messages) == 2
    assert persisted_messages[0].content[0].text == "Build feature"
    assert persisted_messages[1].content[0].text == "final answer"
    assert tool_calls == []


@pytest.mark.asyncio
async def test_session_runner_routes_normal_run_until_waiting_for_input(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    agent_process = FakeToolThenFinalAgentProcess()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=task_manager,
        agent_process=agent_process,
        cancel_event=asyncio.Event(),
    )

    result = await runner.run("Build feature")

    metadata = db.get_runner_state_metadata("session_a")
    persisted_messages = db.list_runner_messages("session_a")
    assert result.status == "done"
    assert metadata.next_action == "wait_user_input"
    assert len(agent_process.calls) == 2
    assert len(agent_process.tool_step_calls) == 1
    assert [message.content[0].text for message in persisted_messages] == [
        "Build feature",
        "done",
        "tool done",
        "final answer",
    ]


@pytest.mark.asyncio
async def test_session_runner_run_after_done_starts_new_user_task(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=TaskManager(db),
        agent_process=FakeFinalAgentProcess(),
        cancel_event=asyncio.Event(),
    )

    first = await runner.run("First request")
    second = await runner.run("Second request")

    assert first.title == "First request"
    assert second.title == "Second request"
    assert len(runner._agent_process.calls) == 2
    assert runner._agent_process.calls[1]["messages"][-1].content[0].text == "Second request"


@pytest.mark.asyncio
async def test_session_runner_run_none_continues_running_task_without_user_prompt(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    task_manager.load(None)
    user_task = task_manager.create_user_task("Paused request")
    task_manager.save()
    db.upsert_runner_state_metadata(
        "session_a",
        next_action="normal_run",
        active_user_task_id=user_task.id,
    )
    agent_process = FakeFinalAgentProcess()
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
    assert agent_process.calls[0]["messages"] == []


@pytest.mark.asyncio
async def test_session_runner_stops_after_step_pause(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    task_manager.load(None)
    user_task = task_manager.create_user_task("Paused request")
    task_manager.create_todo("Active todo")
    task_manager.save()
    db.upsert_runner_state_metadata(
        "session_a",
        next_action="normal_run",
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

    agent_process.pause_on_tool_step = runner.pause

    result = await runner.run(None)

    metadata = db.get_runner_state_metadata("session_a")
    assert result.title == "Paused request"
    assert len(agent_process.calls) == 1
    assert metadata.next_action == "normal_run"


@pytest.mark.asyncio
async def test_session_runner_run_none_returns_done_task_without_agent_call(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    agent_process = FakeFinalAgentProcess()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=TaskManager(db),
        agent_process=agent_process,
        cancel_event=asyncio.Event(),
    )

    result = await runner.run("Build feature")
    continued = await runner.run(None)

    metadata = db.get_runner_state_metadata("session_a")
    assert result.status == "done"
    assert continued.id == result.id
    assert len(agent_process.calls) == 1
    assert metadata.next_action == "wait_user_input"


@pytest.mark.asyncio
async def test_session_runner_new_input_closes_previous_task_and_creates_new_task(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    task_manager.load(None)
    previous_task = task_manager.create_user_task("Previous request")
    task_manager.save()
    db.upsert_runner_state_metadata(
        "session_a",
        next_action="normal_run",
        active_user_task_id=previous_task.id,
    )
    agent_process = FakeFinalAgentProcess()
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
    assert agent_process.calls[0]["messages"][-1].content[0].text == "New request"


class FailingAgentProcess:
    async def call_llm_step(self, system_prompt, messages, tools, cancel_event=None):
        raise RuntimeError("agent failed")

    async def run_tool_calls_step(self, tools, assistant_message, cancel_event=None):
        raise AssertionError("tool step should not run after llm failure")

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


def test_append_messages_buffers_until_sync(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=TaskManager(db),
        agent_process=FakeAgentProcess(),
        cancel_event=asyncio.Event(),
    )
    runner.load()
    message = UserMessage(content=[TextContent(text="hello")], timestamp=1)

    runner.append_messages([message])

    assert len(runner._messages) == 1
    assert runner._messages[0] is message
    assert len(runner._uncommitted_messages) == 1
    assert db.list_runner_messages("session_a") == []

    runner.sync_messages()

    persisted = db.list_runner_messages("session_a")
    assert persisted[0].content[0].text == "hello"
    assert runner._uncommitted_messages == []


def test_handle_input_appends_user_message(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=TaskManager(db),
        agent_process=FakeAgentProcess(),
        cancel_event=asyncio.Event(),
    )
    runner.load()

    next_action = runner.handle_input("Build feature")

    messages = db.list_runner_messages("session_a")
    assert next_action == "normal_run"
    assert runner._next_action == "normal_run"
    assert messages[0].content[0].text == "Build feature"


def test_record_tool_call_buffers_until_sync(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    task_manager.load(None)
    user_task = task_manager.create_user_task("Build feature")
    todo = task_manager.create_todo("Inspect files")
    task_manager.save()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=task_manager,
        agent_process=FakeAgentProcess(),
        cancel_event=asyncio.Event(),
    )
    tool_call = ToolCall(id="call_1", name="example_tool", arguments={"path": "."})
    tool_result = ToolResultMessage(
        toolCallId="call_1",
        toolName="example_tool",
        content=[TextContent(text="done")],
    )

    runner.record_tool_call(tool_call, tool_result)

    assert len(runner._uncommitted_tool_calls) == 1
    assert db.list_runner_tool_calls("session_a") == []

    runner.sync_tool_calls()

    records = db.list_runner_tool_calls("session_a")
    loaded_manager = TaskManager(db)
    loaded_manager.load(user_task.id)
    loaded_todo = next(child for child in loaded_manager.active_user_task.children if child.id == todo.id)
    assert len(records) == 1
    assert records[0].id == 0
    assert records[0].tool_call_id == "call_1"
    assert loaded_todo.children[0].tool_call_log_id == 0
    assert runner._uncommitted_tool_calls == []


def test_failed_save_keeps_uncommitted_buffers(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = FailingSaveTaskManager(db)
    task_manager.load(None)
    task_manager.create_user_task("Build feature")
    task_manager.save_calls = 1
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=task_manager,
        agent_process=FakeAgentProcess(),
        cancel_event=asyncio.Event(),
    )
    runner.load()
    message = UserMessage(content=[TextContent(text="hello")], timestamp=1)

    with pytest.raises(RuntimeError, match="task save failed"):
        runner.save_current_data(messages=[message])

    assert len(runner._uncommitted_messages) == 1
    assert db.list_runner_messages("session_a") == []


@pytest.mark.asyncio
async def test_step_token_threshold_persists_compact_and_sets_cancel(tmp_path):
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
    runner._next_action = "normal_run"

    next_action = await runner.handle_running("Build feature")

    metadata = db.get_runner_state_metadata("session_a")
    assert metadata.next_action == "compact"
    assert cancel_event.is_set() is True
    assert runner._next_action == "compact"
    assert next_action == "compact"


@pytest.mark.asyncio
async def test_handle_running_with_tool_calls_returns_normal_run_without_compaction(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=task_manager,
        agent_process=FakeAgentProcess(),
        cancel_event=asyncio.Event(),
    )
    task_manager.load(None)
    user_task = task_manager.create_user_task("Build feature")
    runner._active_user_task_id = user_task.id
    runner._next_action = "normal_run"

    next_action = await runner.handle_running(None)

    metadata = db.get_runner_state_metadata("session_a")
    assert metadata.next_action == "normal_run"
    assert runner._next_action == "normal_run"
    assert next_action == "normal_run"


@pytest.mark.asyncio
async def test_handle_running_without_tool_calls_returns_done(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=task_manager,
        agent_process=FakeFinalAgentProcess(),
        cancel_event=asyncio.Event(),
    )
    task_manager.load(None)
    user_task = task_manager.create_user_task("Build feature")
    runner._active_user_task_id = user_task.id
    runner._next_action = "normal_run"

    next_action = await runner.handle_running(None)

    metadata = db.get_runner_state_metadata("session_a")
    assert metadata.next_action == "wait_user_input"
    assert runner._next_action == "wait_user_input"
    assert next_action == "wait_user_input"


@pytest.mark.asyncio
async def test_step_tool_call_threshold_persists_compact_and_sets_cancel(tmp_path):
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
        tool_call_json="{}",
        tool_result_json='{"content":[]}',
    )
    runner._active_user_task_id = user_task.id
    runner._next_action = "normal_run"

    next_action = await runner.handle_running("Build feature")

    metadata = db.get_runner_state_metadata("session_a")
    assert metadata.next_action == "compact"
    assert cancel_event.is_set() is True
    assert runner._next_action == "compact"
    assert next_action == "compact"


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
    runner._next_action = "compact"
    cancel_event.set()

    next_action = await runner.handle_compact("Build feature")

    metadata = db.get_runner_state_metadata("session_a")
    assert runner._next_action == "normal_run"
    assert metadata.next_action == "normal_run"
    assert cancel_event.is_set() is False
    assert next_action == "normal_run"


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
    runner._next_action = "compact"
    runner._messages = [
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
    ]
    db.replace_runner_messages("session_a", runner._messages)
    db.insert_runner_tool_call(
        session_id="session_a",
        tool_call_id="tool_1",
        tool_name="read",
        tool_call_json="{}",
        tool_result_json='{"content":[]}',
    )

    next_action = await runner.handle_compact("Build feature")

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
    assert db.get_runner_state_metadata("session_a").next_action == "normal_run"
    assert next_action == "normal_run"


@pytest.mark.asyncio
async def test_session_runner_finds_assistant_message_index_for_tool_call_id(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=TaskManager(db),
        agent_process=FakeAgentProcess(),
        cancel_event=asyncio.Event(),
    )
    runner._messages = [
        UserMessage(content=[TextContent(text="request")], timestamp=1),
        AssistantMessage(role="assistant", content=[TextContent(text="thinking")]),
        AssistantMessage(
            role="assistant",
            content=[
                TextContent(text="create todo"),
                ToolCall(id="call_create", name="create_todo", arguments={"title": "Inspect files"}),
            ],
        ),
    ]

    assert runner.find_assistant_message_index_for_tool_call("call_create") == 2


@pytest.mark.asyncio
async def test_session_runner_finds_tool_result_message_index_for_tool_call_id(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=TaskManager(db),
        agent_process=FakeAgentProcess(),
        cancel_event=asyncio.Event(),
    )
    runner._messages = [
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
    ]

    assert runner.find_tool_result_message_index_for_tool_call("call_finish") == 2


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
    assert metadata.next_action == "wait_user_input"
    assert metadata.last_error == "agent failed"


@pytest.mark.asyncio
async def test_step_save_rolls_back_messages_when_task_save_fails(tmp_path):
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

    messages = db.list_runner_messages("session_a")
    assert len(messages) == 1
    assert messages[0].content[0].text == "Build feature"
