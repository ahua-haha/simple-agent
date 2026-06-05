"""Tests for SessionRunner."""

from __future__ import annotations

import asyncio
import sqlite3

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


class FakeAssistantErrorAgentProcess(FakeAgentProcess):
    async def call_llm_step(self, system_prompt, messages, tools, cancel_event=None):
        message = AssistantMessage(
            role="assistant",
            content=[],
            stopReason="error",
            errorMessage="HTTP 400 Bad Request",
        )
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "messages": messages,
                "tools": [tool.name for tool in tools],
                "cancel_event": cancel_event,
            }
        )
        return message

    async def run_tool_calls_step(self, tools, assistant_message, cancel_event=None):
        raise AssertionError("tool step should not run after assistant error")


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


class FakeCreateTodoAgentProcess(FakeAgentProcess):
    async def call_llm_step(self, system_prompt, messages, tools, cancel_event=None):
        message = AssistantMessage(
            role="assistant",
            content=[
                TextContent(text="creating todo"),
                ToolCall(id="call_create", name="create_todo", arguments={"title": "Inspect files"}),
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
        return message

    async def run_tool_calls_step(self, tools, assistant_message, cancel_event=None):
        by_name = {tool.name: tool for tool in tools}
        results = []
        for content in assistant_message.content:
            if isinstance(content, ToolCall):
                result = await by_name[content.name].execute(content.id, content.arguments)
                results.append(
                    ToolResultMessage(
                        toolCallId=content.id,
                        toolName=content.name,
                        content=result.content,
                    )
                )
        return results


class FakeCompactAgentProcess(FakeAgentProcess):
    async def call_llm_step(self, system_prompt, messages, tools, cancel_event=None):
        self.calls.append({"tools": [tool.name for tool in tools], "messages": list(messages)})
        if len(self.calls) > 1:
            return AssistantMessage(role="assistant", content=[TextContent(text="compact done")])
        return AssistantMessage(
            role="assistant",
            content=[
                ToolCall(id="compact_create", name="create_compacted_todo", arguments={"description": "Compact summary"}),
                ToolCall(id="compact_record", name="record_compacted_tool_call", arguments={"tool_call_log_id": 1}),
                ToolCall(id="compact_finish", name="finish_compacted_todo", arguments={}),
            ],
        )

    async def run_tool_calls_step(self, tools, assistant_message, cancel_event=None):
        by_name = {tool.name: tool for tool in tools}
        results = []
        for content in assistant_message.content:
            if isinstance(content, ToolCall):
                result = await by_name[content.name].execute(content.id, content.arguments)
                results.append(
                    ToolResultMessage(
                        toolCallId=content.id,
                        toolName=content.name,
                        content=result.content,
                    )
                )
        return results


class FakeCompactErrorAgentProcess(FakeAgentProcess):
    async def call_llm_step(self, system_prompt, messages, tools, cancel_event=None):
        message = AssistantMessage(
            role="assistant",
            content=[],
            stopReason="error",
            errorMessage="HTTP 400 Bad Request",
        )
        self.calls.append({"tools": [tool.name for tool in tools], "messages": list(messages)})
        return message

    async def run_tool_calls_step(self, tools, assistant_message, cancel_event=None):
        raise AssertionError("tool step should not run for compact error message")


def _load_task_manager(manager: TaskManager, active_user_task_id: int | None) -> None:
    with manager._db.create_session() as session:
        manager.load(active_user_task_id, session=session)


def _save_task_manager(manager: TaskManager) -> None:
    with manager._db.create_session() as session:
        manager.save(session=session)
        session.commit()


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
    assert len(agent_process.calls) == 3
    assert len(agent_process.tool_step_calls) == 1
    assert agent_process.calls[2]["tools"] == [
        "create_compacted_todo",
        "record_compacted_tool_call",
        "finish_compacted_todo",
    ]
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
    assert runner._agent_process.calls[1]["messages"][-2].content[0].text == "Second request"
    instruction = runner._agent_process.calls[1]["messages"][-1].content[0].text
    assert "determine whether the user task is complex" in instruction.lower()


@pytest.mark.asyncio
async def test_session_runner_run_none_continues_running_task_without_user_prompt(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    _load_task_manager(task_manager, None)
    user_task = task_manager.create_user_task("Paused request")
    _save_task_manager(task_manager)
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
    assert len(agent_process.calls[0]["messages"]) == 1
    instruction = agent_process.calls[0]["messages"][0].content[0].text
    assert "determine whether the user task is complex" in instruction.lower()


@pytest.mark.asyncio
async def test_session_runner_stops_after_step_pause(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    _load_task_manager(task_manager, None)
    user_task = task_manager.create_user_task("Paused request")
    task_manager.create_todo("Active todo")
    _save_task_manager(task_manager)
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
    _load_task_manager(task_manager, None)
    previous_task = task_manager.create_user_task("Previous request")
    _save_task_manager(task_manager)
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
    assert agent_process.calls[0]["messages"][-2].content[0].text == "New request"
    instruction = agent_process.calls[0]["messages"][-1].content[0].text
    assert "determine whether the user task is complex" in instruction.lower()


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

    def save(self, *, session):
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

    with db.create_session() as session:
        runner.sync_messages(session=session)
        session.commit()
    runner._uncommitted_messages.clear()

    persisted = db.list_runner_messages("session_a")
    assert persisted[0].content[0].text == "hello"
    assert runner._uncommitted_messages == []


def test_save_current_data_only_syncs_existing_buffers(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=TaskManager(db),
        agent_process=FakeAgentProcess(),
        cancel_event=asyncio.Event(),
    )
    runner.load()
    runner._next_action = "normal_run"

    runner.save_current_data()

    assert runner._messages == []
    assert runner._uncommitted_messages == []
    assert db.list_runner_messages("session_a") == []
    assert db.get_runner_state_metadata("session_a").next_action == "normal_run"


def test_save_current_data_rejects_message_mutation(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=TaskManager(db),
        agent_process=FakeAgentProcess(),
        cancel_event=asyncio.Event(),
    )
    message = UserMessage(content=[TextContent(text="hello")], timestamp=1)

    with pytest.raises(TypeError):
        runner.save_current_data(messages=[message])


def test_save_current_data_rejects_metadata_mutation_args(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=TaskManager(db),
        agent_process=FakeAgentProcess(),
        cancel_event=asyncio.Event(),
    )

    with pytest.raises(TypeError):
        runner.save_current_data(next_action="normal_run")

    with pytest.raises(TypeError):
        runner.save_current_data(last_error="failed")


def test_sync_metadata_uses_runner_fields(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=TaskManager(db),
        agent_process=FakeAgentProcess(),
        cancel_event=asyncio.Event(),
    )
    runner._next_action = "compact"
    runner._active_user_task_id = 42
    runner._last_error = "boom"

    with db.create_session() as session:
        runner.sync_metadata(session=session)
        session.commit()

    metadata = db.get_runner_state_metadata("session_a")
    assert metadata.next_action == "compact"
    assert metadata.active_user_task_id == 42
    assert metadata.last_error == "boom"


def test_handle_error_sets_action_then_persists_error(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    cancel_event = asyncio.Event()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=TaskManager(db),
        agent_process=FakeAgentProcess(),
        cancel_event=cancel_event,
    )
    cancel_event.set()

    next_action = runner.handle_error(RuntimeError("boom"))
    persisted_action = runner.handle_error()

    metadata = db.get_runner_state_metadata("session_a")
    assert next_action == "handle_error"
    assert persisted_action == "wait_user_input"
    assert runner._next_action == "wait_user_input"
    assert metadata.next_action == "wait_user_input"
    assert metadata.last_error == "boom"
    assert cancel_event.is_set() is False


def test_sync_messages_requires_session(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=TaskManager(db),
        agent_process=FakeAgentProcess(),
        cancel_event=asyncio.Event(),
    )

    with pytest.raises(TypeError):
        runner.sync_messages()


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


def test_record_tool_calls_buffers_results_until_sync(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    _load_task_manager(task_manager, None)
    user_task = task_manager.create_user_task("Build feature")
    todo = task_manager.create_todo("Inspect files")
    _save_task_manager(task_manager)
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=task_manager,
        agent_process=FakeAgentProcess(),
        cancel_event=asyncio.Event(),
    )
    tool_call = ToolCall(id="call_1", name="example_tool", arguments={"path": "."})
    assistant_message = AssistantMessage(
        role="assistant",
        content=[TextContent(text="using tool"), tool_call],
    )
    tool_result = ToolResultMessage(
        toolCallId="call_1",
        toolName="example_tool",
        content=[TextContent(text="done")],
    )

    runner.record_tool_calls(assistant_message, [tool_result])

    assert len(runner._uncommitted_tool_calls) == 1
    assert runner._uncommitted_tool_calls[0].tool_call is tool_call
    assert runner._uncommitted_tool_calls[0].tool_result is tool_result
    assert db.list_runner_tool_calls("session_a") == []

    with db.create_session() as session:
        runner.sync_tool_calls(session=session)
        task_manager.save(session=session)
        session.commit()
    runner._uncommitted_tool_calls.clear()

    records = db.list_runner_tool_calls("session_a")
    loaded_manager = TaskManager(db)
    _load_task_manager(loaded_manager, user_task.id)
    loaded_todo = next(child for child in loaded_manager.active_user_task.children if child.id == todo.id)
    assert len(records) == 1
    assert records[0].id == 0
    assert records[0].tool_call_id == "call_1"
    assert loaded_todo.children[0].tool_call_log_id == 0
    assert runner._uncommitted_tool_calls == []


def test_sync_tool_calls_requires_session(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    _load_task_manager(task_manager, None)
    task_manager.create_user_task("Build feature")
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=task_manager,
        agent_process=FakeAgentProcess(),
        cancel_event=asyncio.Event(),
    )

    with pytest.raises(TypeError):
        runner.sync_tool_calls()


def test_failed_save_keeps_uncommitted_buffers(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = FailingSaveTaskManager(db)
    _load_task_manager(task_manager, None)
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
    runner.append_messages([message])

    with pytest.raises(RuntimeError, match="task save failed"):
        runner.save_current_data()

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
    _load_task_manager(task_manager, None)
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
    agent_process = FakeAgentProcess()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=task_manager,
        agent_process=agent_process,
        cancel_event=asyncio.Event(),
    )
    _load_task_manager(task_manager, None)
    user_task = task_manager.create_user_task("Build feature")
    task_manager.create_todo("Inspect files")
    runner._active_user_task_id = user_task.id
    runner._next_action = "normal_run"

    next_action = await runner.handle_running(None)

    metadata = db.get_runner_state_metadata("session_a")
    assert metadata.next_action == "normal_run"
    assert runner._next_action == "normal_run"
    assert next_action == "normal_run"
    instruction = agent_process.calls[0]["messages"][-1].content[0].text
    assert "Focus on the active todo: Inspect files" in instruction


@pytest.mark.asyncio
async def test_handle_running_sets_current_assistant_message_id_for_task_tools(tmp_path):
    db_path = tmp_path / "session.db"
    db = Database(str(db_path))
    task_manager = TaskManager(db)
    agent_process = FakeCreateTodoAgentProcess()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=task_manager,
        agent_process=agent_process,
        cancel_event=asyncio.Event(),
    )
    _load_task_manager(task_manager, None)
    user_task = task_manager.create_user_task("Build feature")
    runner._active_user_task_id = user_task.id
    runner._next_action = "normal_run"

    next_action = await runner.handle_running(None)

    loaded_manager = TaskManager(db)
    _load_task_manager(loaded_manager, user_task.id)
    todo = next(child for child in loaded_manager.active_user_task.children if child.kind == "todo")
    with sqlite3.connect(db_path) as conn:
        assistant_message_id = conn.execute(
            """
            SELECT id
            FROM runnermessagerecord
            WHERE session_id = ? AND role = ?
            ORDER BY seq
            """,
            ("session_a", "assistant"),
        ).fetchone()[0]

    assert next_action == "normal_run"
    assert todo.start_message_id == assistant_message_id
    assert task_manager.current_assistant_message_id is None


@pytest.mark.asyncio
async def test_handle_running_adds_transient_steering_user_message(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    agent_process = FakeFinalAgentProcess()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=task_manager,
        agent_process=agent_process,
        cancel_event=asyncio.Event(),
    )
    _load_task_manager(task_manager, None)
    user_task = task_manager.create_user_task("Build feature")
    request_message = UserMessage(content=[TextContent(text="Build feature")], timestamp=1)
    runner._active_user_task_id = user_task.id
    runner._next_action = "normal_run"
    runner._messages = [request_message]

    await runner.handle_running(None)

    llm_messages = agent_process.calls[0]["messages"]
    persisted_messages = db.list_runner_messages("session_a")
    assert llm_messages[:-1] == [request_message]
    assert llm_messages[-1].role == "user"
    assert "determine whether the user task is complex" in llm_messages[-1].content[0].text.lower()
    assert [message.role for message in runner._messages] == ["user", "assistant"]
    assert [message.role for message in persisted_messages] == ["assistant"]
    assert persisted_messages[0].content[0].text == "final answer"


@pytest.mark.asyncio
async def test_handle_running_without_tool_calls_finishes_user_task_then_compacts(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=task_manager,
        agent_process=FakeFinalAgentProcess(),
        cancel_event=asyncio.Event(),
    )
    _load_task_manager(task_manager, None)
    user_task = task_manager.create_user_task("Build feature")
    runner._active_user_task_id = user_task.id
    runner._next_action = "normal_run"
    save_calls = []
    original_save_current_data = runner.save_current_data

    def spy_save_current_data(*args, **kwargs):
        save_calls.append((args, kwargs))
        return original_save_current_data(*args, **kwargs)

    runner.save_current_data = spy_save_current_data

    next_action = await runner.handle_running(None)

    metadata = db.get_runner_state_metadata("session_a")
    assert metadata.next_action == "compact"
    assert runner._next_action == "compact"
    assert next_action == "compact"
    assert len(save_calls) == 1


@pytest.mark.asyncio
async def test_handle_running_error_message_sets_handle_error_action_without_handling(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=task_manager,
        agent_process=FakeAssistantErrorAgentProcess(),
        cancel_event=asyncio.Event(),
    )
    _load_task_manager(task_manager, None)
    user_task = task_manager.create_user_task("Build feature")
    runner._active_user_task_id = user_task.id
    runner._next_action = "normal_run"

    def fail_if_handled(error=None):
        raise AssertionError("handle_running should not call handle_error")

    runner.handle_error = fail_if_handled

    next_action = await runner.handle_running(None)

    assert next_action == "handle_error"
    assert runner._next_action == "handle_error"
    assert runner._last_error == "HTTP 400 Bad Request"
    assert db.get_runner_state_metadata("session_a") is None


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
    _load_task_manager(task_manager, None)
    user_task = task_manager.create_user_task("Build feature")
    db.insert_runner_tool_call(
        session_id="session_a",
        tool_call_id="call_99",
        tool_name="example_tool",
        tool_call_json="{}",
        tool_result_json='{"content":[]}',
    )
    with db.create_session() as session:
        runner._next_tool_call_log_id = db.next_runner_tool_call_id("session_a", session=session)
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
    _load_task_manager(runner._task_manager, None)
    user_task = runner._task_manager.create_user_task("Build feature")
    runner._task_manager.create_todo("Still active", start_message_id=1)
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
async def test_handle_compact_runs_loop_then_replaces_scoped_messages_and_tasks(tmp_path):
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
    _load_task_manager(task_manager, None)
    user_task = task_manager.create_user_task("Build feature")
    todo = task_manager.create_todo("Inspect files", start_message_id=2)
    task_manager.finish_task("Done", end_message_id=4)
    active = task_manager.create_todo("Continue work")
    _save_task_manager(task_manager)
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
    original_messages = list(runner._messages)
    db.insert_runner_tool_call(
        id=1,
        session_id="session_a",
        tool_call_id="tool_1",
        tool_name="read",
        tool_call_json="{}",
        tool_result_json='{"content":[]}',
    )

    next_action = await runner.handle_compact("Build feature")

    messages = db.list_runner_messages("session_a")
    loaded_task_manager = TaskManager(db)
    _load_task_manager(loaded_task_manager, user_task.id)
    assert compact_agent.calls[0]["tools"] == [
        "create_compacted_todo",
        "record_compacted_tool_call",
        "finish_compacted_todo",
    ]
    assert len(compact_agent.calls[0]["messages"]) == len(original_messages) + 1
    assert compact_agent.calls[0]["messages"][:-1] == original_messages
    compact_instruction = compact_agent.calls[0]["messages"][-1].content[0].text
    assert "Runtime instruction for compacting phase" in compact_instruction
    assert "Task view to compact:" in compact_instruction
    assert "- todo [done] Inspect files" in compact_instruction
    assert len(compact_agent.calls[1]["messages"]) == len(original_messages) + 5
    assert [message.content[0].text for message in runner._messages] == [
        "original request",
        "Compacted todo: Compact summary\nUseful tool calls: [1]",
        "active todo",
    ]
    assert [message.content for message in messages] == [message.content for message in runner._messages]
    assert [task.id for task in loaded_task_manager.active_user_task.children] == [todo.id, active.id]
    assert loaded_task_manager.active_user_task.children[0].result == "Compact summary"
    assert db.get_runner_state_metadata("session_a").next_action == "normal_run"
    assert next_action == "normal_run"


@pytest.mark.asyncio
async def test_handle_compact_done_user_task_waits_for_user_input(tmp_path):
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
    _load_task_manager(task_manager, None)
    user_task = task_manager.create_user_task("Build feature", start_message_id=1)
    task_manager.record_tool_call(1)
    task_manager.finish_user_task(end_message_id=1)
    _save_task_manager(task_manager)
    runner._active_user_task_id = user_task.id
    runner._next_action = "compact"
    runner._messages = [UserMessage(content=[TextContent(text="Build feature")], timestamp=1)]
    db.replace_runner_messages("session_a", runner._messages, ids=[1])

    next_action = await runner.handle_compact("Build feature", run_done=True)

    metadata = db.get_runner_state_metadata("session_a")
    assert next_action == "wait_user_input"
    assert runner._next_action == "wait_user_input"
    assert metadata.next_action == "wait_user_input"
    assert len(compact_agent.calls) == 2
    assert [message.content[0].text for message in db.list_runner_messages("session_a")] == [
        "Compacted todo: Compact summary\nUseful tool calls: [1]"
    ]


@pytest.mark.asyncio
async def test_handle_compact_routes_error_assistant_message_to_handle_error(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    compact_agent = FakeCompactErrorAgentProcess()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=task_manager,
        agent_process=compact_agent,
        cancel_event=asyncio.Event(),
    )
    _load_task_manager(task_manager, None)
    user_task = task_manager.create_user_task("Build feature")
    task_manager.record_tool_call(1)
    task_manager.finish_user_task()
    _save_task_manager(task_manager)
    runner._active_user_task_id = user_task.id
    runner._next_action = "compact"
    runner._messages = [UserMessage(content=[TextContent(text="Build feature")], timestamp=1)]
    handle_error = runner.handle_error

    def fail_if_handled(error=None):
        raise AssertionError("handle_compact should not call handle_error")

    runner.handle_error = fail_if_handled

    next_action = await runner.handle_compact("Build feature", run_done=True)
    runner.handle_error = handle_error
    routed_action = await runner.route_next_action(next_action, "Build feature")

    metadata = db.get_runner_state_metadata("session_a")
    assert next_action == "handle_error"
    assert routed_action == "wait_user_input"
    assert metadata.next_action == "wait_user_input"
    assert metadata.last_error == "HTTP 400 Bad Request"
    assert len(compact_agent.calls) == 1


@pytest.mark.asyncio
async def test_session_runner_routes_runtime_error_to_handle_error(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=task_manager,
        agent_process=FailingAgentProcess(),
        cancel_event=asyncio.Event(),
    )

    result = await runner.run("Build feature")

    metadata = db.get_runner_state_metadata("session_a")
    assert result.title == "Build feature"
    assert metadata.next_action == "wait_user_input"
    assert metadata.last_error == "agent failed"


@pytest.mark.asyncio
async def test_session_runner_routes_assistant_error_message_to_handle_error(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=task_manager,
        agent_process=FakeAssistantErrorAgentProcess(),
        cancel_event=asyncio.Event(),
    )

    result = await runner.run("Build feature")

    metadata = db.get_runner_state_metadata("session_a")
    messages = db.list_runner_messages("session_a")
    assert result.title == "Build feature"
    assert metadata.next_action == "wait_user_input"
    assert metadata.last_error == "HTTP 400 Bad Request"
    assert len(messages) == 1
    assert messages[0].content[0].text == "Build feature"


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

    result = await runner.run("Build feature")

    metadata = db.get_runner_state_metadata("session_a")
    messages = db.list_runner_messages("session_a")
    assert result.title == "Build feature"
    assert metadata.next_action == "wait_user_input"
    assert metadata.last_error == "task save failed"
    assert len(messages) == 1
    assert messages[0].content[0].text == "Build feature"
