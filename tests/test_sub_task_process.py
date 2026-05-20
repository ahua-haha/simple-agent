"""Tests for SubTaskProcess."""

from __future__ import annotations

import os

import pytest

from simple_agent.process.sub_task_process import SubTaskProcess, SYSTEM_PROMPT
from simple_agent.state.state import Task, SessionState

requires_api_key = pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY not set",
)


class TestSubTaskProcess:
    """Tests for SubTaskProcess."""

    def test_process_init(self):
        """SubTaskProcess should initialize without errors."""
        proc = SubTaskProcess()
        assert proc.proc is not None
        assert proc.tools_mgr is not None

    def test_planner_has_define_task_and_determine_state(self):
        """Planner should have define_task and determine_state tools."""
        proc = SubTaskProcess()
        tool_names = [t.name for t in proc.proc._tools]
        assert "define_task" in tool_names
        assert "determine_state" in tool_names

    def test_define_task_tool_has_input(self):
        """define_task tool should have input parameter."""
        proc = SubTaskProcess()
        for tool in proc.proc._tools:
            if tool.name == "define_task":
                props = tool.parameters.get("properties", {})
                assert "input" in props
                return
        pytest.fail("define_task tool not found")

    def test_determine_state_tool_has_state_and_reason(self):
        """determine_state tool should have state and reason."""
        proc = SubTaskProcess()
        for tool in proc.proc._tools:
            if tool.name == "determine_state":
                props = tool.parameters.get("properties", {})
                assert "state" in props
                assert "reason" in props
                return
        pytest.fail("determine_state tool not found")

    def test_system_prompt_mentions_define_task(self):
        """SYSTEM_PROMPT should mention define_task."""
        assert "define_task" in SYSTEM_PROMPT

    def test_system_prompt_mentions_determine_state(self):
        """SYSTEM_PROMPT should mention determine_state."""
        assert "determine_state" in SYSTEM_PROMPT

    @requires_api_key
    @pytest.mark.asyncio
    async def test_process_single_sub_task(self):
        """process() should handle a task with sub-tasks."""
        task = Task(input="explore the project and summarize findings")
        state = SessionState(name="test")
        proc = SubTaskProcess()
        await proc.process(task, state)

        assert task.subTasks is not None

    @requires_api_key
    @pytest.mark.asyncio
    async def test_process_accumulates_sub_tasks(self):
        """process() should accumulate sub-tasks in task.subTasks."""
        task = Task(input="explore src/ and tests/ separately then summarize")
        state = SessionState(name="test")
        proc = SubTaskProcess()
        await proc.process(task, state)
        assert task.subTasks is not None
