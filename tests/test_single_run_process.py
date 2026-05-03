"""Tests for SingleRunProcess."""

from __future__ import annotations

import pytest

from simple_agent.process.singleRunProcess import SingleRunProcess
from simple_agent.state.state import SingleRunTask, TextResult


class TestSingleRunProcess:
    """Tests for SingleRunProcess."""

    def test_single_run_process_init(self):
        """SingleRunProcess should initialize without errors."""
        proc = SingleRunProcess()
        assert proc.agent is not None

    @pytest.mark.asyncio
    async def test_process_single_run_task(self):
        """SingleRunProcess.process should handle a SingleRunTask."""
        task = SingleRunTask()
        task.input = "show me the directory structure, and the main entry file content"
        task.result = []
        task.message = []
        task.tasks = []
        proc = SingleRunProcess()
        # Should complete without raising
        await proc.process(task)

    @pytest.mark.asyncio
    async def test_process_returns_none(self):
        """SingleRunProcess.process should return None."""
        task = SingleRunTask()
        task.input = "Say 'hello'"
        task.result = []
        task.message = []
        task.tasks = []
        proc = SingleRunProcess()
        result = await proc.process(task)
        assert result is None