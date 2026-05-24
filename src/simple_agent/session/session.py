"""Session — stores a task tree, saves/loads checkpoints, runs via CentralControl."""

from __future__ import annotations

import os

from pi.ai import get_model

from simple_agent.process.agent_process import AgentProcess
from simple_agent.process.central_control import CentralControl
from simple_agent.process.runners import PlanRunner, ExploreRunner, CollectRunner, SingleRunRunner
from simple_agent.state.state import Task
from simple_agent.models import register_custom_models

RUNNERS = {
    "plan": PlanRunner(),
    "explore": ExploreRunner(),
    "collect": CollectRunner(),
    "single_run": SingleRunRunner(),
}


class Session:
    """A simple session that stores a task tree and runs it via CentralControl.

    Usage::

        session = Session("my-task")
        task = await session.run("build a test suite")
        # task tree is checkpointed after every agent cycle
    """

    def __init__(self, name: str, base_dir: str = "./sessions"):
        self._name = name
        self._base_dir = base_dir
        self._filepath = os.path.join(base_dir, f"{name}.json")

        if os.path.exists(self._filepath):
            with open(self._filepath) as f:
                self._root = Task.from_checkpoint(f.read())
        else:
            self._root = None

    @property
    def root(self) -> Task | None:
        return self._root

    async def run(self, user_input: str) -> Task:
        if self._root is not None and self._root.state != "FINISHED":
            root = self._root
        else:
            root = Task(input=user_input, state="PENDING")
        self._root = root

        register_custom_models()
        model = get_model("deepseek", "deepseek-v4-pro")
        agent_process = AgentProcess(model)

        cc = CentralControl(
            root,
            RUNNERS,
            checkpoint_fn=self.checkpoint,
        )
        await cc.run()
        return root

    def checkpoint(self) -> str:
        os.makedirs(os.path.dirname(self._filepath) or ".", exist_ok=True)
        if self._root is not None:
            self._root.to_checkpoint()
            with open(self._filepath, "w") as f:
                f.write(self._root.to_checkpoint())
        return self._filepath

    @staticmethod
    def list_sessions(base_dir: str = "./sessions") -> list[str]:
        if not os.path.isdir(base_dir):
            return []
        return sorted(
            f[:-5] for f in os.listdir(base_dir) if f.endswith(".json")
        )
