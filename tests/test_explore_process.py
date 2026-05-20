"""Tests for ExploreProcess."""

from __future__ import annotations

import pytest

from simple_agent.process.explore_process import ExploreProcess
from simple_agent.state.state import Task, SessionState


class TestExploreProcess:
    """Tests for ExploreProcess."""

    @pytest.mark.asyncio
    async def test_process_explore_task(self):
        """ExploreProcess.process should handle a task with shared state."""
        task = Task(
            input="use tool calls to show the directory structure, and the main entry file conten",
            message=[]
        )
        state = SessionState(name="test")
        proc = ExploreProcess()
        # Should complete without raising
        await proc.process(task, state)
