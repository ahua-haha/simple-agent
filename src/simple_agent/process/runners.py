"""Task runners — one per task type. Each runs a single agent cycle and returns a signal."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from simple_agent.state.state import Task


@dataclass
class RunnerResult:
    """Signal returned by a runner after one agent cycle.

    kind="continue":  agent ran, no terminal tool called — keep same cursor
    kind="finished":  task complete — central control moves cursor up
    kind="sub_task":  sub-task created — central control moves cursor down
    """

    kind: Literal["continue", "finished", "sub_task"]
    child: "Task | None" = None


class BaseRunner(ABC):
    """Abstract runner for a single task type.

    Subclasses set ``self.type`` and implement ``run()``.
    """

    type: str

    @abstractmethod
    async def run(self, task: "Task") -> RunnerResult:
        """Execute one agent cycle for *task* and return a signal."""
        ...


class PlanRunner(BaseRunner):
    """Runner for plan tasks — uses define_task and determine_state tools."""

    type = "plan"

    async def run(self, task: "Task") -> RunnerResult:
        # TODO: implement plan runner
        # - build AgentState, wrap define_task (stop) + determine_state (stop)
        # - run agent via AgentProcess.run(system_prompt, task.context(), tools, state)
        # - on define_task: return RunnerResult("sub_task", child=child_task)
        # - on determine_state: return RunnerResult("finished")
        # - otherwise: return RunnerResult("continue")
        return RunnerResult(kind="continue")


class ExploreRunner(BaseRunner):
    """Runner for explore tasks — uses determine_state and coding tools."""

    type = "explore"

    async def run(self, task: "Task") -> RunnerResult:
        # TODO: implement explore runner
        # - build AgentState, wrap determine_state (stop) + coding tools
        # - run agent via AgentProcess.run(...)
        # - on determine_state: return RunnerResult("finished")
        # - otherwise: return RunnerResult("continue")
        return RunnerResult(kind="continue")


class CollectRunner(BaseRunner):
    """Runner for collect tasks — uses record_textresult and coding tools."""

    type = "collect"

    async def run(self, task: "Task") -> RunnerResult:
        # TODO: implement collect runner
        # - build AgentState, wrap record_textresult + coding tools
        # - run agent via AgentProcess.run(...)
        # - on finish or no stop tool: return RunnerResult("finished")
        return RunnerResult(kind="continue")


class SingleRunRunner(BaseRunner):
    """Runner for single_run tasks — uses determine_state and coding tools."""

    type = "single_run"

    async def run(self, task: "Task") -> RunnerResult:
        # TODO: implement single_run runner
        # - like ExploreRunner but for single_run task type
        return RunnerResult(kind="continue")
