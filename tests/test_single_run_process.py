"""Tests for SingleRunProcess."""

from __future__ import annotations

import pytest

from simple_agent.process.single_run_process import SingleRunProcess, SYSTEM_PROMPT
from simple_agent.state.state import SingleRunTask, TextResult, Task, StateClarification


class TestSingleRunProcess:
    """Tests for SingleRunProcess."""

    @pytest.mark.asyncio
    async def test_single_run_process(self):
        """SingleRunProcess.process should handle a SingleRunTask."""
        task = SingleRunTask(
            input="summarize what this project do, what the core module do",
            message=[]
        )
        proc = SingleRunProcess()
        # Should complete without raising
        await proc.process(task)

    def test_single_run_process_init(self):
        """SingleRunProcess should initialize without errors."""
        proc = SingleRunProcess()
        assert proc.agent is not None

    def test_single_run_process_has_task_collector(self):
        """SingleRunProcess should have a task_collector attribute."""
        proc = SingleRunProcess()
        assert hasattr(proc, 'task_collector')
        assert proc.task_collector is not None

    def test_task_collector_has_define_task_tool(self):
        """Task collector should have define_task tool."""
        proc = SingleRunProcess()
        tool_names = [t.name for t in proc.task_collector.tools]
        assert "define_task" in tool_names

    def test_system_prompt_mentions_define_task(self):
        """SYSTEM_PROMPT should mention define_task tool."""
        assert "define_task" in SYSTEM_PROMPT

    def test_system_prompt_mentions_final_response(self):
        """SYSTEM_PROMPT should mention final response path."""
        assert "final response" in SYSTEM_PROMPT.lower()

    def test_define_task_tool_has_input_and_scope_index(self):
        """define_task tool should have input and scope_index parameters."""
        proc = SingleRunProcess()
        for tool in proc.task_collector.tools:
            if tool.name == "define_task":
                props = tool.parameters.get("properties", {})
                assert "input" in props
                assert "scope_index" in props
                return
        pytest.fail("define_task tool not found")