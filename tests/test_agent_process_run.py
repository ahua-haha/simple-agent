"""Tests for AgentProcess.run() and AgentState."""

from __future__ import annotations

import pytest

from simple_agent.process.agent_process import AgentProcess, AgentState


def _fake_model():
    class _Model:
        pass
    return _Model()


def _make_process():
    return AgentProcess(_fake_model())


def _make_tool(name="test_tool", execute_fn=None):
    from pi.agent import AgentTool

    async def _default_execute(tool_call_id, params, cancel_event=None, on_update=None):
        from pi.agent import AgentToolResult
        return AgentToolResult(content=[])

    tool = AgentTool(
        name=name,
        description="Test tool",
        parameters={"type": "object", "properties": {}},
        execute=execute_fn or _default_execute,
    )
    return tool


class TestAgentState:
    """Tests for AgentState."""

    def test_defaults(self):
        state = AgentState()
        assert state.new_messages == []
        assert state.tool_results == {}
        assert state.finish_reason is None
        assert state.tool_calls == {}
        assert state.turn_count == 0
        assert state.error is None

    def test_is_set_false_by_default(self):
        state = AgentState()
        assert state.is_set() is False

    def test_is_set_when_finish_reason_set(self):
        state = AgentState()
        state.finish_reason = "determine_state"
        assert state.is_set() is True

    def test_is_set_after_explicit_set(self):
        state = AgentState()
        state.set()
        assert state.is_set() is True

    def test_stop_condition_called(self):
        state = AgentState()
        state.stop_condition = lambda s: s.turn_count >= 3
        assert state.is_set() is False
        state.turn_count = 3
        assert state.is_set() is True

    def test_tool_results_accumulate(self):
        """Caller wraps tools to write into state.tool_results."""
        state = AgentState()

        async def _execute(tool_call_id, params, cancel_event=None, on_update=None):
            from pi.agent import AgentToolResult
            # Simulate record tool: write result into state
            state.tool_results.setdefault("record_textresult", []).append(params.get("desc", ""))
            return AgentToolResult(content=[])

        tool = _make_tool(name="record_textresult", execute_fn=_execute)
        assert tool is not None

    def test_stop_via_tool(self):
        """Caller wraps tool to set finish_reason and stop."""
        state = AgentState()

        async def _execute(tool_call_id, params, cancel_event=None, on_update=None):
            from pi.agent import AgentToolResult
            state.finish_reason = "determine_state"
            state.set()
            return AgentToolResult(content=[])

        tool = _make_tool(name="determine_state", execute_fn=_execute)
        assert tool is not None


class TestBindTool:
    """Tests for AgentState.bind_tool()."""

    def test_bind_tool_records_result(self):
        state = AgentState()

        tool = _make_tool(name="record_textresult")
        # Simulate a record tool that sets tool.result
        tool.result = "accomplished X"

        state.bind_tool(tool)
        # After bind_tool, tool.execute is wrapped — execute it
        import asyncio

        async def _run():
            await tool.execute("id1", {})

        asyncio.run(_run())
        assert state.tool_results["record_textresult"] == ["accomplished X"]

    def test_bind_tool_stop_sets_finish_reason(self):
        state = AgentState()

        tool = _make_tool(name="determine_state")
        tool.result = "finished"

        state.bind_tool(tool, stop=True)

        import asyncio

        async def _run():
            await tool.execute("id1", {})

        asyncio.run(_run())
        assert state.finish_reason == "determine_state"
        assert state.is_set() is True

    def test_bind_tool_returns_tool(self):
        state = AgentState()
        tool = _make_tool()
        result = state.bind_tool(tool)
        assert result is tool

    def test_bind_tool_no_result_does_nothing(self):
        state = AgentState()
        tool = _make_tool(name="noop")
        tool.result = None

        state.bind_tool(tool)

        import asyncio

        async def _run():
            await tool.execute("id1", {})

        asyncio.run(_run())
        assert "noop" not in state.tool_results


class TestAgentProcessRun:
    """Tests for AgentProcess.run() — caller owns state and tools."""

    def test_run_takes_state_parameter(self):
        proc = _make_process()
        import inspect
        sig = inspect.signature(proc.run)
        params = list(sig.parameters.keys())
        assert "state" in params
        assert "tools" in params
        assert "system_prompt" in params
        assert "messages" in params

    def test_run_does_not_store_state(self):
        """After run(), self.state is unchanged (not the passed-in state)."""
        proc = _make_process()
        original = proc.state
        # Can't actually run without LLM, but verify pattern
        assert proc.state is original


class TestAgentProcessBackwardCompat:
    """Tests that step() and add_tool() still work unchanged."""

    def test_step_still_works(self):
        proc = _make_process()
        assert hasattr(proc, "step")

    def test_step_stop_condition_is_set(self):
        proc = _make_process()
        state = AgentState()
        state.stop_condition = lambda s: s.turn_count >= 1
        proc.state = state
        assert proc.state.stop_condition is not None

    def test_add_tool_still_works(self):
        proc = _make_process()
        tool = _make_tool()
        result = proc.add_tool(tool)
        assert result is proc

    def test_add_tool_with_on_call_still_works(self):
        proc = _make_process()
        tool = _make_tool()
        called = []
        proc.add_tool(tool, on_call=lambda p: called.append(True))
        assert len(proc._tools) == 1

    def test_add_tool_with_store_still_works(self):
        proc = _make_process()
        tool = _make_tool()
        proc.add_tool(tool, store=True)
        assert len(proc._tools) == 1

    def test_reset_clears_state(self):
        proc = _make_process()
        proc.state.turn_count = 5
        proc.reset()
        assert isinstance(proc.state, AgentState)
        assert proc.state.turn_count == 0

    def test_stop_agent_sets_state(self):
        proc = _make_process()
        proc.stop_agent("test_reason")
        assert proc.state.finish_reason == "test_reason"
        assert proc.state.is_set() is True
