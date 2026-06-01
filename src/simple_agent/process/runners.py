"""Task runners — base classes and stub runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

from simple_agent.process.agent_process import AgentProcess
from simple_agent.tool.execution_logger import ToolExecutionLogger
from simple_agent.db.db import Database

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
    """Abstract runner for a single task type."""

    type: str

    @abstractmethod
    async def run(self, task: "Task") -> RunnerResult:
        """Execute one agent cycle for *task* and return a signal."""
        ...


class CollectRunner(BaseRunner):
    """Runner for collect tasks — uses record_textresult and coding tools."""

    type = "collect"

    def __init__(self, db: Database, execution_logger: ToolExecutionLogger, agent_process: AgentProcess):
        self._db = db
        self._execution_logger = execution_logger
        self._agent_process = agent_process

    async def run(self, task: "Task") -> RunnerResult:
        # TODO: implement collect runner
        return RunnerResult(kind="continue")


class SingleRunRunner(BaseRunner):
    """Runner for single_run tasks — uses determine_state and coding tools."""

    type = "single_run"

    def __init__(self, db: Database, execution_logger: ToolExecutionLogger, agent_process: AgentProcess):
        self._db = db
        self._execution_logger = execution_logger
        self._agent_process = agent_process

    async def run(self, task: "Task") -> RunnerResult:
        # TODO: implement single_run runner
        return RunnerResult(kind="continue")
