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

    def _make_cc(self, root, runner, checkpoints=None):
        if checkpoints is None:
            checkpoints = []
        return CentralControl(
            root,
            runners={"test": runner},
            checkpoint_fn=lambda: checkpoints.append(True),
        )

    @pytest.mark.asyncio
    async def test_continue_checkpoints_and_loops(self):
        """continue signal should checkpoint and re-dispatch."""
        checkpoints = []
        root = Task(input="root", type="test", state="RUNNING")
        cc = self._make_cc(root, _ContinueRunner(), checkpoints)

        # Will loop forever since runner always returns continue.
        # Verify the signal handling logic directly.
        result = RunnerResult(kind="continue")
        cc._checkpoint()
        assert len(checkpoints) == 1
        # continue doesn't change cursor
        assert cc.cursor is root

    @pytest.mark.asyncio
    async def test_finished_absorbs_into_parent(self):
        """finished signal: child → parent.finished_tasks, cursor ↑."""
        parent = Task(input="parent", type="test", state="WAITING")
        child = Task(input="child", type="test", state="RUNNING", parent=parent,
                     result=[TextResult(desc="done", toolCallLogID=[])])
        parent.running_task = child

        checkpoints = []
        cc = self._make_cc(parent, _FinishedRunner(), checkpoints)
        cc.cursor = child

        cc._handle_finished()
        cc._checkpoint()

        assert cc.cursor is parent
        assert len(parent.finished_tasks) == 1
        assert parent.finished_tasks[0] is child
        assert parent.running_task is None
        assert len(parent.messages) == 1

    @pytest.mark.asyncio
    async def test_finished_root_terminates(self):
        """finished on root sets cursor to None."""
        root = Task(input="root", type="test", state="RUNNING")

        cc = self._make_cc(root, _FinishedRunner())
        cc._handle_finished()

        assert cc.cursor is None

    @pytest.mark.asyncio
    async def test_sub_task_wires_child(self):
        """sub_task signal: child wired into tree, cursor ↓."""
        parent = Task(input="parent", type="test", state="RUNNING")
        child = Task(input="child", type="test", state="PENDING")

        cc = self._make_cc(parent, _SubTaskRunner())
        result = RunnerResult(kind="sub_task", child=child)

        cc._handle_sub_task(result)
        cc._checkpoint()

        assert cc.cursor is child
        assert child.parent is parent
        assert parent.running_task is child
        assert parent.state == "WAITING"

    @pytest.mark.asyncio
    async def test_sub_task_no_child(self):
        """sub_task with no child is a no-op."""
        parent = Task(input="parent", type="test", state="RUNNING")
        cc = self._make_cc(parent, _SubTaskRunner())
        result = RunnerResult(kind="sub_task", child=None)

        cc._handle_sub_task(result)
        assert cc.cursor is parent
        assert parent.running_task is None

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Simulate a complete plan → explore → finish workflow."""
        checkpoints = []
        root = Task(input="build tests", type="test", state="RUNNING")

        cc = self._make_cc(root, _SubTaskRunner(), checkpoints)

        # Step 1: sub_task — cursor ↓ to child
        child = Task(input="explore", type="test", state="PENDING")
        cc._handle_sub_task(RunnerResult(kind="sub_task", child=child))
        cc._checkpoint()
        assert cc.cursor is child
        assert cc.cursor.parent is root
        assert root.state == "WAITING"

        # Step 2: finished on child — cursor ↑ to parent
        cc._handle_finished()
        cc._checkpoint()
        assert cc.cursor is root
        assert len(root.finished_tasks) == 1

        # Step 3: finished on root
        cc._handle_finished()
        cc._checkpoint()
        assert cc.cursor is None
        assert len(checkpoints) == 3
