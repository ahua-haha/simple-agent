"""Tests for CommitCollectResultProcess."""

from __future__ import annotations

import os

import pytest

from simple_agent.process.commit_collect_result_process import (
    CommitCollectResultProcess,
    SYSTEM_PROMPT,
)
from simple_agent.state.state import Task

requires_api_key = pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY not set",
)


class TestCommitCollectResultProcess:
    """Tests for CommitCollectResultProcess."""

    def test_process_init(self):
        """CommitCollectResultProcess should initialize without errors."""
        proc = CommitCollectResultProcess()
        assert proc.agent is not None
        assert proc.instruction_collector is not None
        assert proc.result_collector is not None

    def test_process_has_extract_instruction_tool(self):
        """Should have extract_instruction tool via instruction_collector."""
        proc = CommitCollectResultProcess()
        tool_names = [t.name for t in proc.instruction_collector.tools]
        assert "extract_instruction" in tool_names

    def test_process_has_record_textresult_tool(self):
        """Should have record_textresult tool via result_collector."""
        proc = CommitCollectResultProcess()
        tool_names = [t.name for t in proc.result_collector.tools]
        assert "record_textresult" in tool_names

    def test_commit_data_property_empty_by_default(self):
        """commit_data property should return empty CommitData by default."""
        proc = CommitCollectResultProcess()
        cd = proc.commit_data
        assert cd.extracted_instructions == []
        assert cd.aggregated_results == []

    def test_system_prompt_mentions_extract_instruction(self):
        """SYSTEM_PROMPT should mention extract_instruction."""
        assert "extract_instruction" in SYSTEM_PROMPT

    def test_system_prompt_mentions_aggregate(self):
        """SYSTEM_PROMPT should mention aggregation."""
        assert "aggregat" in SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_tool_inspect(self):
        """SYSTEM_PROMPT should mention tool-inspect."""
        assert "tool-inspect" in SYSTEM_PROMPT

    def test_system_prompt_mentions_finish(self):
        """SYSTEM_PROMPT should mention FINISH."""
        assert "FINISH" in SYSTEM_PROMPT

    @requires_api_key
    @pytest.mark.asyncio
    async def test_process_populates_task_result(self):
        """process() should populate task.result with TextResults."""
        task = Task(input="")
        proc = CommitCollectResultProcess()
        await proc.process(task, context=[])

        assert task.result is not None
