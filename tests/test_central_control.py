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
    """Tests for CentralControl signal handling."""

    def _make_cc(self, root, runner, checkpoints=None, tasks=None):
        if checkpoints is None:
            checkpoints = []
        if tasks is None:
            tasks = {root.id: root} if root.id else {}
        return CentralControl(
            root,
            tasks_by_id=tasks,
            runners={"test": runner},
            checkpoint_fn=lambda: checkpoints.append(True),
        )

    @pytest.mark.asyncio
    async def test_continue_checkpoints_and_loops(self):
        """continue signal should checkpoint and re-dispatch."""
        checkpoints = []
        root = Task(input="root", type="test", state="RUNNING")
        cc = self._make_cc(root, _ContinueRunner(), checkpoints)

        cc._checkpoint()
        assert len(checkpoints) == 1
        assert cc.cursor is root

    @pytest.mark.asyncio
    async def test_finished_absorbs_into_parent(self):
        """finished signal: child → parent.finished_task_ids, cursor ↑."""
        root = Task(id=1, input="root", type="test", state="WAITING")
        child = Task(id=2, input="child", type="test", state="RUNNING", parent_id=1,
                     result=[TextResult(desc="done", toolCallLogID=[])])
        root.running_task = child
        root.running_task_id = child.id
        tasks = {1: root, 2: child}

        cc = self._make_cc(root, _FinishedRunner(), tasks=tasks)
        cc.cursor = child

        cc._handle_finished()
        cc._checkpoint()

        assert cc.cursor is root
        assert root.finished_task_ids == [2]
        assert root.running_task is None
        assert root.running_task_id is None
        assert len(root.messages) == 1

    @pytest.mark.asyncio
    async def test_finished_root_terminates(self):
        """finished on root sets cursor to None."""
        root = Task(id=1, input="root", type="test", state="RUNNING")
        tasks = {1: root}

        cc = self._make_cc(root, _FinishedRunner(), tasks=tasks)
        cc._handle_finished()

        assert cc.cursor is None

    @pytest.mark.asyncio
    async def test_sub_task_wires_child(self):
        """sub_task signal: child wired into tree, cursor ↓."""
        parent = Task(id=1, input="parent", type="test", state="RUNNING")
        child = Task(id=2, input="child", type="test", state="PENDING")
        tasks = {1: parent}

        cc = self._make_cc(parent, _SubTaskRunner(), tasks=tasks)
        result = RunnerResult(kind="sub_task", child=child)

        cc._handle_sub_task(result)
        cc._checkpoint()

        assert cc.cursor is child
        assert child.parent_id == 1
        assert parent.running_task is child
        assert parent.running_task_id == 2
        assert parent.state == "WAITING"

    @pytest.mark.asyncio
    async def test_sub_task_no_child(self):
        """sub_task with no child is a no-op."""
        parent = Task(id=1, input="parent", type="test", state="RUNNING")
        tasks = {1: parent}
        cc = self._make_cc(parent, _SubTaskRunner(), tasks=tasks)
        result = RunnerResult(kind="sub_task", child=None)

        cc._handle_sub_task(result)
        assert cc.cursor is parent
        assert parent.running_task is None

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Simulate a complete plan → explore → finish workflow."""
        checkpoints = []
        root = Task(id=1, input="build tests", type="test", state="RUNNING")
        tasks = {1: root}

        cc = self._make_cc(root, _SubTaskRunner(), checkpoints, tasks=tasks)

        # Step 1: sub_task — cursor ↓ to child
        child = Task(id=2, input="explore", type="test", state="PENDING")
        cc._handle_sub_task(RunnerResult(kind="sub_task", child=child))
        cc._checkpoint()
        assert cc.cursor is child
        assert child.parent_id == 1
        assert root.state == "WAITING"
        assert root.running_task_id == 2

        # Step 2: finished on child — cursor ↑ to parent
        cc._handle_finished()
        cc._checkpoint()
        assert cc.cursor is root
        assert root.finished_task_ids == [2]

        # Step 3: finished on root
        cc._handle_finished()
        cc._checkpoint()
        assert cc.cursor is None
        assert len(checkpoints) == 3
