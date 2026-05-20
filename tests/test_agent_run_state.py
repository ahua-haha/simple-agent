"""Tests for AgentRunState as an asyncio.Event with stop logic."""

from __future__ import annotations

import pytest

from simple_agent.state.agent_run_state import AgentRunState


class TestAgentRunState:
    def _make_state(self, stop_condition=None):
        kwargs = {}
        if stop_condition is not None:
            kwargs["stop_condition"] = stop_condition
        return AgentRunState(**kwargs)

    def test_is_set_false_by_default(self):
        state = self._make_state(stop_condition=lambda s: False)
        assert state.is_set() is False

    def test_is_set_true_when_should_stop_returns_true(self):
        state = self._make_state(stop_condition=lambda s: True)
        assert state.is_set() is True

    def test_is_set_true_after_explicit_set(self):
        state = self._make_state(stop_condition=lambda s: False)
        state.set()
        assert state.is_set() is True

    def test_explicit_set_then_clear(self):
        state = self._make_state(stop_condition=lambda s: False)
        state.set()
        assert state.is_set() is True
        state.clear()
        assert state.is_set() is False

    def test_is_set_reads_current_state(self):
        state = self._make_state(stop_condition=lambda s: s.turn_count >= 3)
        assert state.is_set() is False
        state.turn_count = 3
        assert state.is_set() is True

    def test_default_should_stop_sees_finish_reason(self):
        state = self._make_state()
        assert state.is_set() is False
        state.finish_reason = "determine_state"
        assert state.is_set() is True

    def test_both_conditions_is_set_when_either_true(self):
        state = self._make_state(stop_condition=lambda s: False)
        assert state.is_set() is False
        state.set()
        assert state.is_set() is True

    def test_different_stop_conditions(self):
        state = self._make_state(
            stop_condition=lambda s: s.finish_reason is not None and "determine_state" in s.tool_calls
        )
        assert state.is_set() is False

        state.tool_calls = {"determine_state": [{"state": "finished"}], "bash": [{"command": "ls"}]}
        assert state.is_set() is False

        state.finish_reason = "determine_state"
        assert state.is_set() is True

    def test_default_stop_condition_stops_on_finish_reason(self):
        state = self._make_state()
        assert state.should_stop() is False
        state.finish_reason = "done"
        assert state.should_stop() is True

    def test_is_asyncio_event(self):
        import asyncio
        state = self._make_state(stop_condition=lambda s: False)
        assert isinstance(state, asyncio.Event)
