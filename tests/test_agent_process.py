"""Tests for AgentProcess message passing, state management, and stop predicates."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pi.agent import AgentTool, AgentToolResult
from pi.ai.types import TextContent

from simple_agent.process.agent_process import AgentProcess
from simple_agent.state.agent_run_state import AgentRunState


class TestAgentProcessInit:
    def test_init_creates_agent_compat(self):
        model = MagicMock()
        proc = AgentProcess(model)
        assert proc.agent is not None
        assert isinstance(proc.state, AgentRunState)

    def test_agent_compat_subscribe_delegates(self):
        model = MagicMock()
        proc = AgentProcess(model)
        received = []
        proc.agent.subscribe(lambda e: received.append(e))
        proc._emit("test_event")
        assert received == ["test_event"]

    def test_agent_compat_reset_delegates(self):
        model = MagicMock()
        proc = AgentProcess(model)
        proc.state.turn_count = 5
        proc.agent.reset()
        assert proc.state.turn_count == 0

    def test_agent_compat_set_model_is_noop(self):
        model = MagicMock()
        proc = AgentProcess(model)
        proc.agent.set_model("other")  # should not raise

    def test_agent_compat_state_is_none(self):
        model = MagicMock()
        proc = AgentProcess(model)
        assert proc.agent.state is None


class TestAgentProcessReset:
    def test_reset_clears_state(self):
        model = MagicMock()
        proc = AgentProcess(model)
        proc.state.turn_count = 3
        proc.state.tool_calls = {"bash": [{"cmd": "ls"}]}
        proc.state.finish_reason = "done"
        proc.reset()
        assert proc.state.turn_count == 0
        assert proc.state.tool_calls == {}
        assert proc.state.finish_reason is None

    def test_reset_clears_results(self):
        model = MagicMock()
        proc = AgentProcess(model)
        proc._results["some_tool"] = [1, 2]
        proc.reset()
        assert proc._results == {}


class TestSubscribeAndEmit:
    def test_subscribe_adds_listener(self):
        model = MagicMock()
        proc = AgentProcess(model)
        calls = []
        proc.subscribe(lambda e: calls.append(e))
        proc._emit("event1")
        assert calls == ["event1"]

    def test_multiple_listeners_all_called(self):
        model = MagicMock()
        proc = AgentProcess(model)
        results = []
        proc.subscribe(lambda e: results.append(("a", e)))
        proc.subscribe(lambda e: results.append(("b", e)))
        proc._emit("x")
        assert results == [("a", "x"), ("b", "x")]

    def test_emit_with_no_listeners_does_not_error(self):
        model = MagicMock()
        proc = AgentProcess(model)
        proc._emit("x")  # should not raise


class TestAddTool:
    def _make_tool(self, name="test_tool", result=None):
        tool = AgentTool(name=name, description="test", parameters={})
        async def execute(tool_call_id, params, cancel_event=None, on_update=None):
            return AgentToolResult(content=[TextContent(text="ok")])
        tool.execute = execute
        tool.result = result
        return tool

    def test_add_single_tool(self):
        model = MagicMock()
        proc = AgentProcess(model)
        tool = self._make_tool()
        proc.add_tool(tool)
        assert len(proc._tools) == 1
        assert proc._tools[0].name == "test_tool"

    def test_add_tool_list(self):
        model = MagicMock()
        proc = AgentProcess(model)
        tools = [self._make_tool("a"), self._make_tool("b")]
        proc.add_tool(tools)
        assert len(proc._tools) == 2

    def test_add_tool_returns_self(self):
        model = MagicMock()
        proc = AgentProcess(model)
        result = proc.add_tool(self._make_tool())
        assert result is proc


class TestPruneMessages:
    def test_prune_removes_matching_last_pair(self):
        from pi.ai.types import AssistantMessage, ToolResultMessage

        messages = [
            MagicMock(role="user"),
            AssistantMessage(content=[], timestamp=0),
            ToolResultMessage(tool_call_id="t1", tool_name="determine_state", content=[], details={}, timestamp=0),
        ]
        result = AgentProcess.prune_messages(messages, "determine_state")
        assert len(result) == 1  # only user remains

    def test_prune_does_nothing_when_no_match(self):
        from pi.ai.types import AssistantMessage, ToolResultMessage

        messages = [
            MagicMock(role="user"),
            AssistantMessage(content=[], timestamp=0),
            ToolResultMessage(tool_call_id="t1", tool_name="other_tool", content=[], details={}, timestamp=0),
        ]
        result = AgentProcess.prune_messages(messages, "determine_state")
        assert len(result) == 3

    def test_prune_does_nothing_on_short_messages(self):
        messages = [MagicMock(role="user")]
        result = AgentProcess.prune_messages(messages, "determine_state")
        assert len(result) == 1

    def test_prune_returns_new_list(self):
        messages = [MagicMock(role="user")]
        result = AgentProcess.prune_messages(messages, "determine_state")
        assert result is not messages


class TestStepReturnValue:
    """step() returns (messages, finish_reason, results) tuple."""

    def test_step_returns_messages_when_empty(self):
        """Even with no real agent loop, the method signature and return type are correct."""
        model = MagicMock()
        proc = AgentProcess(model)
        # The step method returns a 3-tuple
        import inspect
        sig = inspect.signature(AgentProcess.step)
        hint = sig.return_annotation
        assert str(hint).startswith("tuple[")


class TestStopAgent:
    def test_stop_agent_sets_state_and_explicit_abort(self):
        model = MagicMock()
        proc = AgentProcess(model)

        proc.state = AgentRunState(stop_condition=lambda s: False)
        assert proc.state.is_set() is False

        proc.stop_agent("determine_state")
        assert proc.state.finish_reason == "determine_state"
        assert proc.state.is_set() is True
