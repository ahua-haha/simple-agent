"""Tests for SingleRunProcess."""

from __future__ import annotations

import pytest

from simple_agent.process.single_run_process import SingleRunProcess, SYSTEM_PROMPT
from simple_agent.state.state import Task


class TestSingleRunProcess:
    """Tests for SingleRunProcess."""

    @pytest.mark.asyncio
    async def test_single_run_process(self):
        """SingleRunProcess.process should handle a Task."""
        task = Task(
            input="summarize what this project do, what the core module do",
        )
        proc = SingleRunProcess()
        await proc.process(task)

    def test_single_run_process_init(self):
        """SingleRunProcess should initialize without errors."""
        proc = SingleRunProcess()
        assert proc.agent is not None

    def test_single_run_process_has_state_collector(self):
        """SingleRunProcess should have a state_collector attribute."""
        proc = SingleRunProcess()
        assert hasattr(proc, 'state_collector')
        assert proc.state_collector is not None

    def test_state_collector_has_determine_state_tool(self):
        """State collector should have determine_state tool."""
        proc = SingleRunProcess()
        tool_names = [t.name for t in proc.state_collector.tools]
        assert "determine_state" in tool_names

    def test_system_prompt_mentions_determine_state(self):
        """SYSTEM_PROMPT should mention determine_state tool."""
        assert "determine_state" in SYSTEM_PROMPT

    def test_determine_state_tool_has_state_and_reason(self):
        """determine_state tool should have state and reason parameters."""
        proc = SingleRunProcess()
        for tool in proc.state_collector.tools:
            if tool.name == "determine_state":
                props = tool.parameters.get("properties", {})
                assert "state" in props
                assert "reason" in props
                return
        pytest.fail("determine_state tool not found")
