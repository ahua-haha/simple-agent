"""Tests for CollectResultProcess."""

from __future__ import annotations

import pytest

from simple_agent.process.collect_result_process import CollectResultProcess, SYSTEM_PROMPT
from simple_agent.state.state import Task, TextResult


class TestCollectResultProcess:
    """Tests for CollectResultProcess."""

    def test_collect_result_process_init(self):
        """CollectResultProcess should initialize without errors."""
        proc = CollectResultProcess()
        assert proc.proc is not None
        assert proc.tools_mgr is not None

    def test_system_prompt_contains_finish_instruction(self):
        """SYSTEM_PROMPT should mention FINISH termination."""
        assert "FINISH" in SYSTEM_PROMPT

    def test_system_prompt_mentions_tool_inspect(self):
        """SYSTEM_PROMPT should mention tool-inspect."""
        assert "tool-inspect" in SYSTEM_PROMPT

    def test_system_prompt_mentions_record_textresult(self):
        """SYSTEM_PROMPT should mention record_textresult."""
        assert "record_textresult" in SYSTEM_PROMPT

    def test_task_creation(self):
        """Task should be creatable with input and result."""
        task = Task(input="test input", result=[])
        assert task.input == "test input"
        assert task.result == []

    def test_text_result_creation(self):
        """TextResult should be creatable with desc and toolCallLogID."""
        result = TextResult(desc="Found error", toolCallLogID=[1, 2])
        assert result.desc == "Found error"
        assert result.toolCallLogID == [1, 2]