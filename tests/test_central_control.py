"""Tests for CentralControl and RunnerResult."""

from __future__ import annotations

import pytest

from simple_agent.process.runners import (
    RunnerResult,
    BaseRunner,
    PlanRunner,
    ExploreRunner,
    CollectRunner,
    SingleRunRunner,
)
from simple_agent.process.central_control import CentralControl
from simple_agent.state.state import Task, TextResult
from simple_agent.db.db import Database


class FakeDB:
    """A fake DB that returns rows from a preloaded dict."""

    def __init__(self, tasks: dict[int, dict] | None = None):
        self._tasks = tasks or {}

    def get_task(self, task_id: int) -> dict | None:
        return self._tasks.get(task_id)

    def upsert_task(self, task) -> int:
        if task.id is None:
            task.id = len(self._tasks) + 1
        return task.id


class TestRunnerResult:
    """Tests for RunnerResult signal."""

    def test_continue_defaults(self):
        r = RunnerResult(kind="continue")
        assert r.kind == "continue"
        assert r.child is None

    def test_finished_defaults(self):
        r = RunnerResult(kind="finished")
        assert r.kind == "finished"
        assert r.child is None

    def test_sub_task_with_child(self):
        child = Task(input="sub", type="explore", state="PENDING")
        r = RunnerResult(kind="sub_task", child=child)
        assert r.kind == "sub_task"
        assert r.child is child


class TestBaseRunner:
    """Tests for BaseRunner ABC."""

    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            BaseRunner()  # type: ignore[abstract]

    def test_subclass_must_implement_run(self):
        class Incomplete(BaseRunner):
            type = "test"

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]


class TestRunnerSubclasses:
    """Tests for stub runner subclasses."""

    def test_plan_runner_type(self):
        r = PlanRunner()
        assert r.type == "plan"

    def test_explore_runner_type(self):
        r = ExploreRunner()
        assert r.type == "explore"

    def test_collect_runner_type(self):
        r = CollectRunner()
        assert r.type == "collect"

    def test_single_run_runner_type(self):
        r = SingleRunRunner()
        assert r.type == "single_run"

    @pytest.mark.asyncio
    async def test_plan_runner_returns_continue(self):
        r = PlanRunner()
        task = Task(input="test", type="plan", state="RUNNING")
        result = await r.run(task)
        assert result.kind == "continue"

    @pytest.mark.asyncio
    async def test_explore_runner_returns_continue(self):
        r = ExploreRunner()
        task = Task(input="test", type="explore", state="RUNNING")
        result = await r.run(task)
        assert result.kind == "continue"

    @pytest.mark.asyncio
    async def test_collect_runner_returns_continue(self):
        r = CollectRunner()
        task = Task(input="test", type="collect", state="RUNNING")
        result = await r.run(task)
        assert result.kind == "continue"


class _ContinueRunner(BaseRunner):
    type = "test"
    async def run(self, task):
        return RunnerResult(kind="continue")


class _FinishedRunner(BaseRunner):
    type = "test"
    async def run(self, task):
        return RunnerResult(kind="finished")


class _SubTaskRunner(BaseRunner):
    type = "test"
    async def run(self, task):
        child = Task(input="child", type="test", state="PENDING")
        return RunnerResult(kind="sub_task", child=child)


class TestCentralControl:
    """Tests for CentralControl single-transition."""

    def _make_cc(self, db=None):
        if db is None:
            db = FakeDB()
        return CentralControl(db, runners={"test": _ContinueRunner()})

    @pytest.mark.asyncio
    async def test_continue_returns_same_cursor(self):
        """continue signal → (cursor, [cursor], [])."""
        cc = self._make_cc()
        cc._runners = {"test": _ContinueRunner()}

        cursor = Task(id=1, input="root", type="test", state="RUNNING")
        new, updates, inserts = await cc.run(cursor)

        assert new is cursor
        assert updates == [cursor]
        assert inserts == []

    @pytest.mark.asyncio
    async def test_finished_absorbs_into_parent(self):
        """finished signal → child added to parent.finished_task_ids."""
        parent_data = {
            "id": 1, "parent_id": None, "running_task_id": 2,
            "finished_task_ids": [], "type": "plan", "state": "WAITING",
            "input": "parent", "messages": [], "result": [],
            "start_snapshot": None, "end_snapshot": None,
        }
        db = FakeDB({1: parent_data})
        cc = CentralControl(db, runners={"test": _FinishedRunner()})

        cursor = Task(id=2, input="child", type="test", state="RUNNING",
                      parent_id=1, result=[TextResult(desc="done", toolCallLogID=[])])
        new, updates, inserts = await cc.run(cursor)

        assert new is not None
        assert new.id == 1
        assert 2 in new.finished_task_ids
        assert new.running_task_id is None
        assert len(updates) == 2
        assert cursor in updates
        assert new in updates
        assert inserts == []

    @pytest.mark.asyncio
    async def test_finished_root_returns_none(self):
        """finished on root → (None, [cursor], [])."""
        cc = CentralControl(FakeDB(), runners={"test": _FinishedRunner()})

        cursor = Task(id=1, input="root", type="test", state="RUNNING")
        new, updates, inserts = await cc.run(cursor)

        assert new is None
        assert cursor.state == "FINISHED"
        assert updates == [cursor]
        assert inserts == []

    @pytest.mark.asyncio
    async def test_sub_task_wires_child(self):
        """sub_task → child.parent_id = cursor.id, cursor = child."""
        db = FakeDB()
        cc = CentralControl(db, runners={"test": _SubTaskRunner()})

        parent = Task(id=1, input="parent", type="test", state="RUNNING")
        new, updates, inserts = await cc.run(parent)

        assert new is not None
        assert new.parent_id == 1
        assert parent.running_task is new
        assert parent.running_task_id == new.id
        assert parent.state == "WAITING"
        assert updates == [parent]
        assert inserts == [new]

    @pytest.mark.asyncio
    async def test_sub_task_no_child_no_op(self):
        """sub_task with None child is no-op."""
        class _NullChildRunner(BaseRunner):
            type = "test"
            async def run(self, task):
                return RunnerResult(kind="sub_task", child=None)

        cc = CentralControl(FakeDB(), runners={"test": _NullChildRunner()})
        parent = Task(id=1, input="parent", type="test", state="RUNNING")
        new, updates, inserts = await cc.run(parent)

        assert new is parent
        assert parent.running_task is None
