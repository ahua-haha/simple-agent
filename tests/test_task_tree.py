"""Tests for Task tree model, context inheritance, serialization, and TaskRunner."""

from __future__ import annotations

import json

import pytest

from simple_agent.state.state import Task, TextResult


def _make_task(**kwargs) -> Task:
    defaults = {"input": "test input", "type": "single_run", "state": "PENDING"}
    defaults.update(kwargs)
    return Task(**defaults)


class TestTaskTree:
    """Tests for tree structure."""

    def test_root_has_no_parent(self):
        task = _make_task()
        assert task.parent is None

    def test_child_references_parent(self):
        parent = _make_task()
        child = _make_task(parent=parent)
        assert child.parent is parent

    def test_running_task_chain(self):
        root = _make_task(type="plan", state="WAITING")
        child = _make_task(type="explore", state="RUNNING", parent=root)
        root.running_task = child
        assert root.running_task is child
        assert root.running_task.state == "RUNNING"

    def test_finished_tasks(self):
        parent = _make_task(state="WAITING")
        child = _make_task(state="FINISHED", parent=parent, result=[TextResult(desc="done", toolCallLogID=[])])
        parent.finished_tasks.append(child)
        assert len(parent.finished_tasks) == 1
        assert parent.finished_tasks[0].result[0].desc == "done"

    def test_defaults(self):
        task = _make_task()
        assert task.type == "single_run"
        assert task.state == "PENDING"
        assert task.result == []
        assert task.messages == []
        assert task.finished_tasks == []
        assert task.running_task is None


class TestContextInheritance:
    """Tests for task.context()."""

    def test_root_context_returns_own_messages(self):
        from pi.ai.types import UserMessage, TextContent
        msg = UserMessage(content=[TextContent(text="hello")], timestamp=0)
        task = _make_task(messages=[msg])
        ctx = task.context()
        assert ctx == [msg]

    def test_child_inherits_parent(self):
        from pi.ai.types import UserMessage, TextContent
        msg_p = UserMessage(content=[TextContent(text="parent")], timestamp=0)
        msg_c = UserMessage(content=[TextContent(text="child")], timestamp=1)
        parent = _make_task(messages=[msg_p])
        child = _make_task(messages=[msg_c], parent=parent)
        ctx = child.context()
        assert len(ctx) == 2
        assert ctx[0] is msg_p
        assert ctx[1] is msg_c

    def test_deep_nesting(self):
        from pi.ai.types import UserMessage, TextContent
        root = _make_task(messages=[UserMessage(content=[TextContent(text="root")], timestamp=0)])
        child = _make_task(messages=[UserMessage(content=[TextContent(text="child")], timestamp=1)], parent=root)
        grandchild = _make_task(messages=[UserMessage(content=[TextContent(text="grandchild")], timestamp=2)], parent=child)
        ctx = grandchild.context()
        assert len(ctx) == 3

    def test_context_empty_tree(self):
        task = _make_task(messages=[])
        ctx = task.context()
        assert ctx == []


class TestSerialization:
    """Tests for checkpoint / reload."""

    def test_parent_excluded_from_json(self):
        parent = _make_task()
        child = _make_task(parent=parent)
        parent.running_task = child
        data = json.loads(parent.to_checkpoint())
        assert "parent" not in data

    def test_reconstruct_tree_restores_parents(self):
        root = _make_task(type="plan", state="WAITING")
        child = _make_task(type="explore", state="RUNNING", parent=root)
        root.running_task = child

        json_str = root.to_checkpoint()
        rebuilt = Task.from_checkpoint(json_str)

        assert rebuilt.parent is None
        assert rebuilt.running_task is not None
        assert rebuilt.running_task.parent is rebuilt

    def test_reconstruct_tree_with_finished_children(self):
        root = _make_task(type="plan", state="WAITING")
        child1 = _make_task(type="explore", state="FINISHED", parent=root,
                            result=[TextResult(desc="found X", toolCallLogID=[])])
        child2 = _make_task(type="explore", state="RUNNING", parent=root)
        root.finished_tasks.append(child1)
        root.running_task = child2

        json_str = root.to_checkpoint()
        rebuilt = Task.from_checkpoint(json_str)

        assert rebuilt.finished_tasks[0].parent is rebuilt
        assert rebuilt.running_task.parent is rebuilt

    def test_checkpoint_cycle_preserves_state(self):
        root = _make_task(type="plan", state="WAITING")
        child = _make_task(type="explore", state="RUNNING", parent=root)
        root.running_task = child

        json_str = root.to_checkpoint()
        rebuilt = Task.from_checkpoint(json_str)

        assert rebuilt.state == "WAITING"
        assert rebuilt.running_task.state == "RUNNING"


class TestFindActive:
    """Tests for task.find_active()."""

    def test_find_active_self(self):
        task = _make_task(state="RUNNING")
        assert task.find_active() is task

    def test_find_active_descends(self):
        root = _make_task(state="FINISHED")
        child = _make_task(state="WAITING", parent=root)
        grandchild = _make_task(state="RUNNING", parent=child)
        root.running_task = child
        child.running_task = grandchild
        assert root.find_active() is grandchild

    def test_find_active_finished_root(self):
        root = _make_task(state="FINISHED")
        assert root.find_active() is root


