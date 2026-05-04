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
        assert proc.agent is not None
        assert proc.collector is not None
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

    def test_collector_has_record_tool(self):
        """Collector should have record_textresult tool."""
        proc = CollectResultProcess()
        tool_names = [t.name for t in proc.collector.tools]
        assert "record_textresult" in tool_names

    def test_process_has_finish_detection_flag(self):
        """Process should have _finish_detected flag."""
        proc = CollectResultProcess()
        assert hasattr(proc, "_finish_detected")
        assert proc._finish_detected == False

    def test_on_event_detects_finish(self):
        """on_event should detect FINISH in text delta."""
        proc = CollectResultProcess()
        proc._finish_detected = False

        # Create a mock event with text_delta containing FINISH
        class MockDeltaEvent:
            type = "message_update"
            class MockAssistantMessage:
                type = "text_delta"
                delta = "FINISH"
            assistant_message_event = MockAssistantMessage()

        proc.on_event(MockDeltaEvent())
        assert proc._finish_detected == True

    def test_on_event_ignores_non_finish(self):
        """on_event should not set flag for non-FINISH text."""
        proc = CollectResultProcess()
        proc._finish_detected = False

        class MockDeltaEvent:
            type = "message_update"
            class MockAssistantMessage:
                type = "text_delta"
                delta = "Some regular text"
            assistant_message_event = MockAssistantMessage()

        proc.on_event(MockDeltaEvent())
        assert proc._finish_detected == False

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