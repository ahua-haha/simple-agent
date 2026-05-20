"""Session - orchestrates multiple processes with shared SessionState."""

from __future__ import annotations

import os
import time

from simple_agent.process.single_run_process import SingleRunProcess
from simple_agent.process.commit_collect_result_process import CommitCollectResultProcess
from simple_agent.state.state import SessionState, Task
from simple_agent.tool.tool_mgr import ToolMgr
from simple_agent.db.db import Database


class Session:
    """Orchestrates runs with a single shared SessionState instance.

    Infrastructure (tools_mgr, db) lives on Session. State lives on SessionState.
    """

    _tools_mgr: ToolMgr
    _db: Database

    def __init__(self, name: str, base_dir: str = "./sessions", tools_mgr: ToolMgr | None = None, db: Database | None = None):
        self._name = name
        self._base_dir = base_dir
        self._tools_mgr = tools_mgr or ToolMgr()
        self._db = db or Database()

        filepath = self._session_filepath()
        if os.path.exists(filepath):
            self.state = SessionState.load(filepath)
        else:
            self.state = SessionState(name=name)

    def _session_filepath(self) -> str:
        return os.path.join(self._base_dir, f"{self._name}.json")

    def _commit_filepath(self, index: int) -> str:
        return os.path.join(self._base_dir, self._name, f"commit_{index:04d}.json")

    async def run(self, user_input: str) -> Task:
        task = Task(input=user_input, messages=[])
        self.state.current_task = task

        proc = SingleRunProcess(tools_mgr=self._tools_mgr, db=self._db)
        await proc.process(task, self.state)

        self.state.current_task = None
        self.state.uncommitted_task.append(task)
        self.state.checkpoint(self._session_filepath())
        return task

    def checkpoint(self) -> str:
        filepath = self._session_filepath()
        self.state.checkpoint(filepath)
        return filepath

    async def commit(self) -> str:
        task = Task(input="")

        proc = CommitCollectResultProcess(tools_mgr=self._tools_mgr, db=self._db)
        await proc.process(task, self.state)

        task.subTasks = list(self.state.uncommitted_task)

        self.state.commit_index += 1
        task_path = self._commit_filepath(self.state.commit_index)
        os.makedirs(os.path.dirname(task_path), exist_ok=True)

        with open(task_path, "w") as f:
            f.write(task.model_dump_json(indent=2))

        self.state.uncommitted_task.clear()
        self.state.checkpoint(self._session_filepath())

        return task_path

    @staticmethod
    def list_sessions(base_dir: str = "./sessions") -> list[str]:
        if not os.path.isdir(base_dir):
            return []
        return sorted(
            f[:-5] for f in os.listdir(base_dir) if f.endswith(".json")
        )
