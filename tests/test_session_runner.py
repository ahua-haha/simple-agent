"""Tests for SessionRunner."""

from __future__ import annotations

import asyncio
import json

import pytest

from pi.ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage

from simple_agent.db.db import Database
from simple_agent.message_store import MessageEntry
from simple_agent.run_log import runtime_logger
from simple_agent.session.runner import SessionRunner
from simple_agent.task_manager import TaskManager
from simple_agent.task_manager.lifecycle import TaskLifecycleError
from simple_agent.task_manager.models import ToolCallTask


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
        if len(self.calls) == 1:
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
        if len(self.calls) == 2:
            message = AssistantMessage(
                role="assistant",
                content=[
                    ToolCall(
                        id="compact_create",
                        name="create_compacted_user_task",
                        arguments={"description": "Compact summary"},
                    ),
                    ToolCall(id="compact_record", name="record_compacted_tool_call", arguments={"tool_call_log_id": 1}),
                    ToolCall(id="compact_finish", name="finish_compacted_user_task", arguments={}),
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

    async def run_tool_calls_step(self, tools, assistant_message, cancel_event=None):
        by_name = {tool.name: tool for tool in tools}
        if any(isinstance(content, ToolCall) and content.name in by_name for content in assistant_message.content):
            results = []
            for content in assistant_message.content:
                if isinstance(content, ToolCall) and content.name in by_name:
                    result = await by_name[content.name].execute(content.id, content.arguments)
                    results.append(
                        ToolResultMessage(
                            toolCallId=content.id,
                            toolName=content.name,
                            content=result.content,
                        )
                    )
            return results
        return await super().run_tool_calls_step(tools, assistant_message, cancel_event=cancel_event)


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
                ToolCall(
                    id="compact_create",
                    name="create_compacted_user_task",
                    arguments={"description": "Compact summary"},
                ),
                ToolCall(id="compact_record", name="record_compacted_tool_call", arguments={"tool_call_log_id": 1}),
                ToolCall(id="compact_finish", name="finish_compacted_user_task", arguments={}),
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


def _create_todo(
    manager: TaskManager,
    title: str,
    *,
    start_message_id: int | None = None,
):
    lifecycle = manager.active_lifecycle_for_tools()
    if start_message_id is not None:
        lifecycle.current_assistant_message_id = start_message_id
    todo = lifecycle.create_todo_task(title=title)
    if start_message_id is not None:
        lifecycle.current_assistant_message_id = None
    manager.refresh_active_task()
    return todo


def _finish_todo(
    manager: TaskManager,
    result: str | None = None,
    *,
    end_message_id: int | None = None,
):
    lifecycle = manager.active_lifecycle_for_tools()
    if end_message_id is not None:
        lifecycle.current_assistant_message_id = end_message_id
    todo = lifecycle.finish_task(result=result)
    if end_message_id is not None:
        lifecycle.current_assistant_message_id = None
    manager.refresh_active_task()
    return todo


def _record_tool_call(manager: TaskManager, tool_call_log_id: int):
    active_task = manager.active_lifecycle_for_tools().task
    tool_call = ToolCallTask(
        id=manager.allocate_task_id(),
        title=f"Tool call {tool_call_log_id}",
        status="done",
        parent_id=active_task.id,
        tool_call_log_id=tool_call_log_id,
    )
    active_task.children.append(tool_call)
    active_task.touch()
    return tool_call


def _run_log_dir(tmp_path):
    return tmp_path / "logs"


def _run_log_path(tmp_path, session_id: str = "session_a"):
    return _run_log_dir(tmp_path) / f"{session_id}.jsonl"


def _use_run_log_dir(tmp_path) -> None:
    runtime_logger.set_log_dir(_run_log_dir(tmp_path))


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


async def _run_lifecycle_step(runner: SessionRunner):
    result = await runner.run_active_lifecycle()
    if result.error is not None:
        runner._raise_lifecycle_error(result.error)

    runner._next_action = result.next_action
    runner._active_user_task_id = (
        runner.user_task.id
        if runner.user_task is not None and runner.user_task.status == "active"
        else None
    )
    with runner._db.create_session() as session:
        runner.sync_metadata(session=session)
        session.commit()
    return result.next_action


def _seed_runner_user_lifecycle(runner: SessionRunner, user_task) -> None:
    with runner._db.create_session() as session:
        runner._load_runtime(session=session)
    runner._user_task = user_task
    runner._active_user_task_id = user_task.id
    runner._runtime.next_task_id_to_run = user_task.id
    runner._runtime.next_task = user_task


@pytest.mark.asyncio
async def test_session_runner_creates_task_runs_agent_and_persists_messages(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    agent_process = FakeFinalAgentProcess()
    cancel_event = asyncio.Event()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        agent_process=agent_process,
        cancel_event=cancel_event,
    )
    agent_process.count_persisted = lambda: len(db.list_runner_messages("session_a"))
    agent_process.observe_next_action = lambda: runner._next_action
    agent_process.observe_metadata_next_action = lambda: db.get_runner_state_metadata("session_a").next_action

    result = await runner.run("Build feature")

    assert result.kind == "user_task"
    assert result.title == "Build feature"
    assert result.status == "active"
    assert len(agent_process.calls) == 1
    assert agent_process.calls[0]["messages"][0].content[0].text == "Build feature"
    assert agent_process.calls[0]["cancel_event"] is cancel_event
    assert "create_todo" in agent_process.calls[0]["tools"]
    assert "finish_user_task" in agent_process.calls[0]["tools"]
    assert "finish_todo" not in agent_process.calls[0]["tools"]
    assert "error_todo" not in agent_process.calls[0]["tools"]
    assert len(agent_process.tool_step_calls) == 0

    metadata = db.get_runner_state_metadata("session_a")
    assert metadata.next_action == "wait_user_input"
    assert metadata.active_user_task_id is None
    persisted_messages = db.list_runner_messages("session_a")
    tool_calls = db.list_runner_tool_calls("session_a")
    assert agent_process.next_action_at_run_start == "normal_run"
    assert len(persisted_messages) == 1
    assert persisted_messages[0].content[0].text == "Build feature"
    assert tool_calls == []


@pytest.mark.asyncio
async def test_session_runner_routes_normal_run_until_waiting_for_input(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    agent_process = FakeToolThenFinalAgentProcess()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        agent_process=agent_process,
        cancel_event=asyncio.Event(),
    )

    result = await runner.run("Build feature")

    metadata = db.get_runner_state_metadata("session_a")
    persisted_messages = db.list_runner_messages("session_a")
    assert result.status == "active"
    assert metadata.next_action == "wait_user_input"
    assert len(agent_process.calls) == 4
    assert len(agent_process.tool_step_calls) == 1
    assert agent_process.calls[2]["tools"] == [
        "create_compacted_user_task",
        "record_compacted_tool_call",
        "finish_compacted_user_task",
    ]
    assert [message.content[0].text for message in persisted_messages] == [
        "Build feature",
    ]


@pytest.mark.asyncio
async def test_session_runner_passes_runtime_context_to_task_instruction(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    agent_process = FakeFinalAgentProcess()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        agent_process=agent_process,
        cancel_event=asyncio.Event(),
    )

    await runner.run("Build feature")

    instruction_message = agent_process.calls[0]["messages"][-1]
    instruction = instruction_message.content[0].text
    assert "Runtime instruction for this turn" in instruction
    assert "Determine whether the user task is complex" in instruction


@pytest.mark.asyncio
async def test_lifecycle_run_writes_message_change_log(tmp_path):
    _use_run_log_dir(tmp_path)
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    agent_process = FakeAgentProcess()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        agent_process=agent_process,
        cancel_event=asyncio.Event(),
    )
    runner.load()
    runner.handle_input("Build feature")

    next_action = await _run_lifecycle_step(runner)

    assert next_action == "normal_run"
    assert not _run_log_path(tmp_path).exists()


@pytest.mark.asyncio
async def test_session_runner_run_after_done_starts_new_user_task(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    runner = SessionRunner(
        session_id="session_a",
        db=db,
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
async def test_session_runner_run_none_does_not_continue_until_load_reconstructs_lifecycle(tmp_path):
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
        agent_process=agent_process,
        cancel_event=asyncio.Event(),
    )

    result = await runner.run(None)

    assert result is None
    assert len(agent_process.calls) == 0
    assert db.get_runner_state_metadata("session_a").next_action == "normal_run"


@pytest.mark.asyncio
async def test_session_runner_stops_after_step_pause(tmp_path):
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
    agent_process = FakeAgentProcess()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        agent_process=agent_process,
        cancel_event=asyncio.Event(),
    )

    agent_process.pause_on_tool_step = runner.pause

    result = await runner.run(None)

    metadata = db.get_runner_state_metadata("session_a")
    assert result is None
    assert len(agent_process.calls) == 0
    assert metadata.next_action == "normal_run"


@pytest.mark.asyncio
async def test_session_runner_run_none_returns_done_task_without_agent_call(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    agent_process = FakeFinalAgentProcess()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        agent_process=agent_process,
        cancel_event=asyncio.Event(),
    )

    result = await runner.run("Build feature")
    continued = await runner.run(None)

    metadata = db.get_runner_state_metadata("session_a")
    assert result.status == "active"
    assert continued is None
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
        agent_process=agent_process,
        cancel_event=asyncio.Event(),
    )

    result = await runner.run("New request")

    reloaded_previous = db.get_managed_task(previous_task.id)
    assert reloaded_previous.status == "active"
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


def test_session_runner_subscribe_delegates_to_agent_process(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    agent_process = FakeAgentProcess()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
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
        agent_process=FakeAgentProcess(),
        cancel_event=cancel_event,
    )

    runner.pause()
    assert cancel_event.is_set()


def test_sync_metadata_uses_runner_fields(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    runner = SessionRunner(
        session_id="session_a",
        db=db,
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


def test_handle_input_appends_user_message(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        agent_process=FakeAgentProcess(),
        cancel_event=asyncio.Event(),
    )
    runner.load()

    next_action = runner.handle_input("Build feature")

    messages = db.list_runner_messages("session_a")
    assert next_action == "normal_run"
    assert runner._next_action == "normal_run"
    assert messages[0].content[0].text == "Build feature"


@pytest.mark.asyncio
async def test_step_token_threshold_persists_compact_and_sets_cancel(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    cancel_event = asyncio.Event()
    task_manager = TaskManager(db)
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        agent_process=FakeAgentProcess(),
        cancel_event=cancel_event,
        context_token_threshold=0,
        tool_call_threshold=100,
    )
    runner.load()
    user_task = runner.create_user_task("Build feature")
    runner._active_user_task_id = user_task.id
    runner._next_action = "normal_run"

    next_action = await _run_lifecycle_step(runner)

    metadata = db.get_runner_state_metadata("session_a")
    assert metadata.next_action == "compact"
    assert cancel_event.is_set() is True
    assert runner._next_action == "compact"
    assert next_action == "compact"


@pytest.mark.asyncio
async def test_lifecycle_run_with_tool_calls_returns_normal_run_without_compaction(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    agent_process = FakeAgentProcess()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        agent_process=agent_process,
        cancel_event=asyncio.Event(),
    )
    runner.load()
    user_task = runner.create_user_task("Build feature")
    runner._active_user_task_id = user_task.id
    runner._next_action = "normal_run"

    next_action = await _run_lifecycle_step(runner)

    metadata = db.get_runner_state_metadata("session_a")
    assert metadata.next_action == "normal_run"
    assert runner._next_action == "normal_run"
    assert next_action == "normal_run"
    instruction = agent_process.calls[0]["messages"][-1].content[0].text
    assert "Determine whether the user task is complex" in instruction


@pytest.mark.asyncio
async def test_lifecycle_run_sets_current_assistant_message_id_for_task_tools(tmp_path):
    db_path = tmp_path / "session.db"
    db = Database(str(db_path))
    task_manager = TaskManager(db)
    agent_process = FakeCreateTodoAgentProcess()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        agent_process=agent_process,
        cancel_event=asyncio.Event(),
    )
    runner.load()
    user_task = runner.create_user_task("Build feature")
    runner._active_user_task_id = user_task.id
    runner._next_action = "normal_run"

    next_action = await _run_lifecycle_step(runner)

    todo = next(child for child in runner.user_task.children if child.kind == "todo")

    assert next_action == "normal_run"
    assert todo.start_message_id == 1
    assert runner._runtime.next_task is todo
    assert runner._lifecycles["user_task"].current_assistant_message_id is None


@pytest.mark.asyncio
async def test_lifecycle_run_adds_transient_steering_user_message(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    agent_process = FakeFinalAgentProcess()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        agent_process=agent_process,
        cancel_event=asyncio.Event(),
    )
    runner.load()
    user_task = runner.create_user_task("Build feature")
    request_message = UserMessage(content=[TextContent(text="Build feature")], timestamp=1)
    runner._active_user_task_id = user_task.id
    runner._next_action = "normal_run"
    db.replace_runner_messages("session_a", [request_message], ids=[1])
    runner._runtime.messages = [MessageEntry(id=1, message=request_message)]
    runner._runtime.next_message_id = 2

    await _run_lifecycle_step(runner)

    llm_messages = agent_process.calls[0]["messages"]
    persisted_messages = db.list_runner_messages("session_a")
    assert llm_messages[:-1] == [request_message]
    assert llm_messages[-1].role == "user"
    assert "determine whether the user task is complex" in llm_messages[-1].content[0].text.lower()
    assert [message.role for message in persisted_messages] == ["user"]


@pytest.mark.asyncio
async def test_lifecycle_run_without_tool_calls_finishes_user_task_then_compacts(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        agent_process=FakeFinalAgentProcess(),
        cancel_event=asyncio.Event(),
    )
    runner.load()
    user_task = runner.create_user_task("Build feature")
    runner._active_user_task_id = user_task.id
    runner._next_action = "normal_run"
    next_action = await _run_lifecycle_step(runner)

    metadata = db.get_runner_state_metadata("session_a")
    messages = db.list_runner_messages("session_a")
    assert metadata.next_action == "compact"
    assert runner._next_action == "compact"
    assert next_action == "compact"
    assert messages == []


@pytest.mark.asyncio
async def test_lifecycle_run_error_message_raises_without_handling(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        agent_process=FakeAssistantErrorAgentProcess(),
        cancel_event=asyncio.Event(),
    )
    runner.load()
    user_task = runner.create_user_task("Build feature")
    runner._active_user_task_id = user_task.id
    runner._next_action = "normal_run"

    with pytest.raises(RuntimeError, match="HTTP 400 Bad Request"):
        await _run_lifecycle_step(runner)

    metadata = db.get_runner_state_metadata("session_a")
    assert metadata is None


@pytest.mark.asyncio
async def test_step_tool_call_threshold_persists_compact_and_sets_cancel(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    cancel_event = asyncio.Event()
    task_manager = TaskManager(db)
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        agent_process=FakeAgentProcess(),
        cancel_event=cancel_event,
        context_token_threshold=1000,
        tool_call_threshold=0,
    )
    runner.load()
    user_task = runner.create_user_task("Build feature")
    db.insert_runner_tool_call(
        session_id="session_a",
        tool_call_id="call_99",
        tool_name="example_tool",
        tool_call_json="{}",
        tool_result_json='{"content":[]}',
    )
    runner._active_user_task_id = user_task.id
    runner._next_action = "normal_run"

    next_action = await _run_lifecycle_step(runner)

    metadata = db.get_runner_state_metadata("session_a")
    assert metadata.next_action == "compact"
    assert cancel_event.is_set() is True
    assert runner._next_action == "compact"
    assert next_action == "compact"


@pytest.mark.asyncio
async def test_lifecycle_run_without_finished_todo_returns_running(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    cancel_event = asyncio.Event()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        agent_process=FakeAgentProcess(),
        cancel_event=cancel_event,
    )
    runner.load()
    user_task = runner.create_user_task("Build feature")
    runner._active_user_task_id = user_task.id
    runner._next_action = "compact"
    cancel_event.set()

    next_action = await _run_lifecycle_step(runner)

    metadata = db.get_runner_state_metadata("session_a")
    assert runner._next_action == "normal_run"
    assert metadata.next_action == "normal_run"
    assert cancel_event.is_set() is True
    assert next_action == "normal_run"


@pytest.mark.asyncio
async def test_lifecycle_run_runs_loop_then_replaces_scoped_messages_and_tasks(tmp_path):
    _use_run_log_dir(tmp_path)
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    compact_agent = FakeCompactAgentProcess()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        agent_process=compact_agent,
        cancel_event=asyncio.Event(),
    )
    _load_task_manager(task_manager, None)
    user_task = task_manager.create_user_task("Build feature", start_message_id=1)
    _record_tool_call(task_manager, 1)
    todo = _create_todo(task_manager, "Inspect files", start_message_id=2)
    _finish_todo(task_manager, "Done", end_message_id=4)
    task_manager.finish_user_task(end_message_id=6)
    _save_task_manager(task_manager)
    runner._active_user_task_id = user_task.id
    runner._next_action = "compact"
    seeded_messages = [
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
        AssistantMessage(role="assistant", content=[TextContent(text="final answer")]),
    ]
    db.replace_runner_messages(
        "session_a",
        seeded_messages,
        ids=list(range(1, len(seeded_messages) + 1)),
    )
    original_messages = list(seeded_messages)
    db.insert_runner_tool_call(
        id=1,
        session_id="session_a",
        tool_call_id="tool_1",
        tool_name="read",
        tool_call_json="{}",
        tool_result_json='{"content":[]}',
    )
    _seed_runner_user_lifecycle(runner, task_manager.user_task)

    next_action = await _run_lifecycle_step(runner)

    messages = db.list_runner_messages("session_a")
    loaded_task_manager = TaskManager(db)
    _load_task_manager(loaded_task_manager, user_task.id)
    assert compact_agent.calls[0]["tools"] == [
        "create_compacted_user_task",
        "record_compacted_tool_call",
        "finish_compacted_user_task",
    ]
    assert len(compact_agent.calls[0]["messages"]) == len(original_messages) + 1
    assert compact_agent.calls[0]["messages"][:-1] == original_messages
    compact_instruction = compact_agent.calls[0]["messages"][-1].content[0].text
    assert "Runtime instruction for compacting phase" in compact_instruction
    assert "Task view to compact:" in compact_instruction
    assert "- todo [done] Inspect files" in compact_instruction
    assert len(compact_agent.calls[1]["messages"]) == len(original_messages) + 5
    assert [message.content[0].text for message in messages] == [
        "original request",
        "old todo",
        "created",
        "finish todo",
        "finished",
        "final answer",
    ]
    assert [task.tool_call_log_id for task in loaded_task_manager.user_task.children if task.kind == "tool_call"] == [1]
    assert loaded_task_manager.user_task.result is None
    assert runner.user_task.result == "Compact summary"
    assert db.get_runner_state_metadata("session_a").next_action == "wait_user_input"
    assert next_action == "wait_user_input"
    assert not _run_log_path(tmp_path).exists()


@pytest.mark.asyncio
async def test_lifecycle_run_done_user_task_waits_for_user_input(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    compact_agent = FakeCompactAgentProcess()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        agent_process=compact_agent,
        cancel_event=asyncio.Event(),
    )
    _load_task_manager(task_manager, None)
    user_task = task_manager.create_user_task("Build feature", start_message_id=1)
    _record_tool_call(task_manager, 1)
    task_manager.finish_user_task(end_message_id=1)
    _save_task_manager(task_manager)
    runner._active_user_task_id = user_task.id
    runner._next_action = "compact"
    request_message = UserMessage(content=[TextContent(text="Build feature")], timestamp=1)
    db.replace_runner_messages("session_a", [request_message], ids=[1])
    _seed_runner_user_lifecycle(runner, task_manager.user_task)

    next_action = await _run_lifecycle_step(runner)

    metadata = db.get_runner_state_metadata("session_a")
    assert next_action == "wait_user_input"
    assert runner._next_action == "wait_user_input"
    assert metadata.next_action == "wait_user_input"
    assert len(compact_agent.calls) == 2
    assert [message.content[0].text for message in db.list_runner_messages("session_a")] == [
        "Build feature"
    ]
    assert runner.user_task.result == "Compact summary"


@pytest.mark.asyncio
async def test_lifecycle_run_raises_compact_error_message(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    compact_agent = FakeCompactErrorAgentProcess()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        agent_process=compact_agent,
        cancel_event=asyncio.Event(),
    )
    _load_task_manager(task_manager, None)
    user_task = task_manager.create_user_task("Build feature")
    _record_tool_call(task_manager, 1)
    task_manager.finish_user_task()
    _save_task_manager(task_manager)
    runner._active_user_task_id = user_task.id
    runner._next_action = "compact"
    db.replace_runner_messages(
        "session_a",
        [UserMessage(content=[TextContent(text="Build feature")], timestamp=1)],
        ids=[1],
    )
    _seed_runner_user_lifecycle(runner, task_manager.user_task)
    with pytest.raises(TaskLifecycleError, match="HTTP 400 Bad Request"):
        await _run_lifecycle_step(runner)

    metadata = db.get_runner_state_metadata("session_a")
    assert metadata is None
    assert len(compact_agent.calls) == 1


@pytest.mark.asyncio
async def test_session_runner_propagates_runtime_error(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        agent_process=FailingAgentProcess(),
        cancel_event=asyncio.Event(),
    )

    with pytest.raises(RuntimeError, match="agent failed"):
        await runner.run("Build feature")

    metadata = db.get_runner_state_metadata("session_a")
    assert metadata.next_action == "normal_run"
    assert metadata.last_error is None


@pytest.mark.asyncio
async def test_session_runner_propagates_assistant_error_message(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        agent_process=FakeAssistantErrorAgentProcess(),
        cancel_event=asyncio.Event(),
    )

    with pytest.raises(RuntimeError, match="HTTP 400 Bad Request"):
        await runner.run("Build feature")

    metadata = db.get_runner_state_metadata("session_a")
    messages = db.list_runner_messages("session_a")
    assert metadata.next_action == "normal_run"
    assert metadata.last_error is None
    assert len(messages) == 1
    assert messages[0].content[0].text == "Build feature"
