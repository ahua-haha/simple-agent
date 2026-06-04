"""Tests for SessionRunner tool-call recording after tool execution."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import pytest

from pi.ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage

from simple_agent.db.db import Database
from simple_agent.session.runner import SessionRunner
from simple_agent.task_manager import TaskManager


@dataclass
class _NestedDetails:
    truncated: bool


@dataclass
class _ToolDetails:
    exit_code: int
    nested: _NestedDetails


class _FakeAgentProcess:
    def __init__(self, *, result_details=None, is_error: bool = False):
        self.result_details = result_details
        self.is_error = is_error
        self.seen_tools = []

    async def call_llm_step(self, system_prompt, messages, tools, cancel_event=None):
        self.seen_tools = [tool.name for tool in tools]
        return AssistantMessage(
            role="assistant",
            content=[
                ToolCall(id="call_1", name="example", arguments={"name": "Ada"}),
            ],
        )

    async def run_tool_calls_step(self, tools, assistant_message, cancel_event=None):
        return [
            ToolResultMessage(
                toolCallId="call_1",
                toolName="example",
                content=[TextContent(text="boom" if self.is_error else "hello")],
                details=self.result_details or {},
                isError=self.is_error,
            )
        ]

    def subscribe(self, callback):
        pass

    def unsubscribe(self, callback):
        pass


def _make_runner(db: Database, agent_process: _FakeAgentProcess | None = None) -> SessionRunner:
    manager = TaskManager(db)
    manager.load(None)
    manager.create_user_task("Build feature")
    return SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=manager,
        agent_process=agent_process or _FakeAgentProcess(),
        cancel_event=asyncio.Event(),
    )


@pytest.mark.asyncio
async def test_runner_records_tool_call_after_tool_step_success(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    runner = _make_runner(db)

    await runner.run("Build feature")

    records = db.list_runner_tool_calls("session_a")
    assert len(records) == 1
    assert records[0].tool_call_id == "call_1"
    assert records[0].tool_name == "example"
    assert records[0].params_json == '{"name": "Ada"}'
    assert records[0].status == "success"
    assert records[0].error is None


@pytest.mark.asyncio
async def test_runner_records_dataclass_tool_result_details_as_json(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    runner = _make_runner(
        db,
        _FakeAgentProcess(result_details=_ToolDetails(exit_code=0, nested=_NestedDetails(truncated=False))),
    )

    await runner.run("Build feature")

    records = db.list_runner_tool_calls("session_a")
    payload = json.loads(records[0].result_json)
    assert payload["details"] == {
        "exit_code": 0,
        "nested": {"truncated": False},
    }


@pytest.mark.asyncio
async def test_runner_records_tool_call_error_result(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    runner = _make_runner(db, _FakeAgentProcess(is_error=True))

    await runner.run("Build feature")

    records = db.list_runner_tool_calls("session_a")
    assert len(records) == 1
    assert records[0].tool_call_id == "call_1"
    assert records[0].tool_name == "example"
    assert records[0].status == "error"
    assert records[0].error == "boom"
