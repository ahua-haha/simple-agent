"""Session — stores a task tree in SQLite, runs via CentralControl."""

from __future__ import annotations

import os

from pi.ai import get_model

from simple_agent.process.agent_process import AgentProcess
from simple_agent.process.central_control import CentralControl
from simple_agent.process.runners import PlanRunner, ExploreRunner, CollectRunner, SingleRunRunner
from simple_agent.state.state import Task
from simple_agent.db.db import Database
from simple_agent.models import register_custom_models

RUNNERS = {
    "plan": PlanRunner(),
    "explore": ExploreRunner(),
    "collect": CollectRunner(),
    "single_run": SingleRunRunner(),
}


class Session:
    """A session that stores a task tree in SQLite and runs via CentralControl.

    Usage::

        session = Session("my-task")
        task = await session.run("build a test suite")
    """

    def __init__(self, name: str, base_dir: str = "./sessions"):
        self._name = name
        self._base_dir = base_dir
        self._db_path = os.path.join(base_dir, f"{name}.db")
        self._db = Database(self._db_path)
        self._tasks: dict[int, Task] = {}

        rows = self._db.load_all_tasks()
        if rows:
            self._tasks = Task.from_db_rows(rows)
            # Root is the task with no parent
            for task in self._tasks.values():
                if task.parent_id is None:
                    self._root = task
                    break
            else:
                self._root = None
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
            self._tasks = {}
        self._root = root
        self._tasks[root.id] = root

        register_custom_models()
        model = get_model("deepseek", "deepseek-v4-pro")
        agent_process = AgentProcess(model)

        cc = CentralControl(root, self._tasks, RUNNERS, checkpoint_fn=self.checkpoint)
        await cc.run()
        return root

    def checkpoint(self) -> None:
        """Upsert cursor and its ancestor chain into the DB."""
        cursor = self._root
        if cursor is None:
            return

        # Walk running_task chain to find the cursor
        while cursor.running_task is not None:
            cursor = cursor.running_task

        # Upsert cursor (gets ID if new) and register in tasks dict
        cursor_id = self._db.upsert_task(cursor)
        if cursor.id is None:
            cursor.id = cursor_id
            self._tasks[cursor_id] = cursor

        # Upsert ancestors up the tree via parent_id
        ancestor_id = cursor.parent_id
        while ancestor_id is not None:
            ancestor = self._tasks.get(ancestor_id)
            if ancestor is not None:
                self._db.upsert_task(ancestor)
                ancestor_id = ancestor.parent_id
            else:
                break

    @staticmethod
    def list_sessions(base_dir: str = "./sessions") -> list[str]:
        if not os.path.isdir(base_dir):
            return []
        return sorted(
            f[:-3] for f in os.listdir(base_dir) if f.endswith(".db")
        )
