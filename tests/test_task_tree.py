"""Tests for Task tree model, context inheritance, and DB persistence."""

from __future__ import annotations

import tempfile

import pytest

from simple_agent.state.state import Task, TextResult
from simple_agent.db.db import Database


def _make_task(**kwargs) -> Task:
    defaults = {"input": "test input", "type": "single_run", "state": "PENDING"}
    defaults.update(kwargs)
    return Task(**defaults)


class TestTaskTree:
    """Tests for tree structure."""

    def test_root_has_no_parent_id(self):
        task = _make_task()
        assert task.parent_id is None

    def test_child_has_parent_id(self):
        parent = _make_task()
        parent.id = 1
        child = _make_task(parent_id=parent.id)
        assert child.parent_id == 1

    def test_running_task_chain(self):
        root = _make_task(type="plan", state="WAITING")
        child = _make_task(type="explore", state="RUNNING")
        root.running_task_id = child.id
        root.running_task = child
        assert root.running_task is child
        assert root.running_task.state == "RUNNING"

    def test_finished_task_ids(self):
        parent = _make_task(state="WAITING")
        parent.finished_task_ids = [2, 3]
        assert parent.finished_task_ids == [2, 3]

    def test_defaults(self):
        task = _make_task()
        assert task.type == "single_run"
        assert task.state == "PENDING"
        assert task.result == []
        assert task.messages == []
        assert task.finished_task_ids == []
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
        parent = _make_task(id=1, messages=[msg_p])
        child = _make_task(id=2, parent_id=1, messages=[msg_c])
        tasks = {1: parent, 2: child}
        ctx = child.context(tasks)
        assert len(ctx) == 2
        assert ctx[0] is msg_p
        assert ctx[1] is msg_c

    def test_deep_nesting(self):
        from pi.ai.types import UserMessage, TextContent
        root = _make_task(id=1, messages=[UserMessage(content=[TextContent(text="root")], timestamp=0)])
        child = _make_task(id=2, parent_id=1, messages=[UserMessage(content=[TextContent(text="child")], timestamp=1)])
        grandchild = _make_task(id=3, parent_id=2, messages=[UserMessage(content=[TextContent(text="grandchild")], timestamp=2)])
        tasks = {1: root, 2: child, 3: grandchild}
        ctx = grandchild.context(tasks)
        assert len(ctx) == 3

    def test_context_empty_tree(self):
        task = _make_task(messages=[])
        ctx = task.context()
        assert ctx == []


class TestDBRoundtrip:
    """Tests for DB upsert / load cycle."""

    def _make_db(self) -> Database:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        return Database(path)

    def _find_root(self, tasks: dict) -> Task | None:
        for t in tasks.values():
            if t.parent_id is None:
                return t
        return None

    def test_upsert_and_load_root(self):
        db = self._make_db()
        root = _make_task(input="hello")
        root.id = db.upsert_task(root)
        assert root.id == 1

        rows = db.load_all_tasks()
        assert len(rows) == 1
        assert rows[0].input == "hello"

    def test_load_single_root(self):
        db = self._make_db()
        root = _make_task(input="hello", state="FINISHED")
        root.id = db.upsert_task(root)

        tasks = Task.from_db_rows(db.load_all_tasks())
        loaded = self._find_root(tasks)
        assert loaded is not None
        assert loaded.input == "hello"
        assert loaded.state == "FINISHED"
        assert loaded.id == 1
        assert loaded.parent_id is None

    def test_parent_child_roundtrip(self):
        db = self._make_db()
        root = _make_task(type="plan", state="WAITING")
        root.id = db.upsert_task(root)

        child = _make_task(type="explore", state="RUNNING", parent_id=root.id)
        child.id = db.upsert_task(child)
        root.running_task_id = child.id
        db.upsert_task(root)

        tasks = Task.from_db_rows(db.load_all_tasks())
        loaded = self._find_root(tasks)
        assert loaded is not None
        assert loaded.running_task is not None
        assert loaded.running_task.input == "test input"

    def test_finished_task_ids_roundtrip(self):
        db = self._make_db()
        root = _make_task(type="plan", state="WAITING")
        root.id = db.upsert_task(root)

        done = _make_task(type="explore", state="FINISHED", parent_id=root.id,
                          result=[TextResult(desc="done", toolCallLogID=[])])
        done.id = db.upsert_task(done)

        active = _make_task(type="explore", state="RUNNING", parent_id=root.id)
        active.id = db.upsert_task(active)
        root.running_task_id = active.id
        root.finished_task_ids = [done.id]
        db.upsert_task(root)

        tasks = Task.from_db_rows(db.load_all_tasks())
        loaded = self._find_root(tasks)
        assert loaded is not None
        assert loaded.finished_task_ids == [done.id]
        assert loaded.running_task is not None
        assert loaded.running_task.id == active.id

    def test_reloaded_tree_context_works(self):
        db = self._make_db()
        from pi.ai.types import UserMessage, TextContent

        root = _make_task(type="plan", state="WAITING",
                          messages=[UserMessage(content=[TextContent(text="root msg")], timestamp=0)])
        root.id = db.upsert_task(root)

        child = _make_task(type="explore", state="RUNNING",
                           messages=[UserMessage(content=[TextContent(text="child msg")], timestamp=1)])
        child.parent_id = root.id
        child.id = db.upsert_task(child)
        root.running_task_id = child.id
        db.upsert_task(root)

        tasks = Task.from_db_rows(db.load_all_tasks())
        loaded = self._find_root(tasks)
        assert loaded is not None
        ctx = loaded.running_task.context(tasks)
        assert len(ctx) == 2


class TestFindActive:
    """Tests for task.find_active()."""

    def test_find_active_self(self):
        task = _make_task(state="RUNNING")
        assert task.find_active() is task

    def test_find_active_descends(self):
        root = _make_task(state="FINISHED")
        child = _make_task(state="WAITING")
        grandchild = _make_task(state="RUNNING")
        root.running_task = child
        child.running_task = grandchild
        assert root.find_active() is grandchild

    def test_find_active_finished_root(self):
        root = _make_task(state="FINISHED")
        assert root.find_active() is root
