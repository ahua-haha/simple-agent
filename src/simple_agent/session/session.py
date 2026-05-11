"""Session - orchestrates multiple SingleRunProcess runs with persistence."""

from __future__ import annotations

import os
import tempfile
import time

from pi.agent.types import AgentMessage
from pydantic import TypeAdapter

from simple_agent.process.single_run_process import SingleRunProcess
from simple_agent.process.commit_collect_result_process import CommitCollectResultProcess
from simple_agent.state.state import CommitData, RunRecord, SessionData, Task
from simple_agent.tool.tool_mgr import ToolMgr
from simple_agent.db.db import Database


class Session:
    messages: list[AgentMessage]
    uncommitted_task: list[Task]
    _name: str
    _base_dir: str
    _created_at: float
    _commit_index: int
    _tools_mgr: ToolMgr
    _db: Database

    def __init__(self, name: str, base_dir: str = "./sessions", tools_mgr: ToolMgr | None = None, db: Database | None = None):
        self._name = name
        self._base_dir = base_dir
        self._tools_mgr = tools_mgr or ToolMgr()
        self._db = db or Database()
        self._created_at = time.time()

        filepath = self._session_filepath()
        if os.path.exists(filepath):
            self._load(filepath)
        else:
            self.messages = []
            self.uncommitted_task = []
            self._commit_index = 0

    def _session_filepath(self) -> str:
        return os.path.join(self._base_dir, f"{self._name}.json")

    def _commit_filepath(self, index: int) -> str:
        return os.path.join(self._base_dir, self._name, f"commit_{index:04d}.json")

    def _load(self, filepath: str) -> None:
        with open(filepath, "r") as f:
            data = SessionData.model_validate_json(f.read())
        self.messages = data.messages
        self._created_at = data.created_at
        self._commit_index = data.commit_index
        self.uncommitted_task = data.uncommitted_task

    async def run(self, user_input: str) -> Task:
        task = Task(input=user_input)

        proc = SingleRunProcess(tools_mgr=self._tools_mgr, db=self._db)
        new_msgs = await proc.process(task, context=self.messages)
        self.messages.extend(new_msgs)

        self.uncommitted_task.append(task)
        return task

    def checkpoint(self) -> str:
        os.makedirs(self._base_dir, exist_ok=True)

        model = SessionData(
            name=self._name,
            messages=self.messages,
            commit_index=self._commit_index,
            uncommitted_task=self.uncommitted_task,
            created_at=self._created_at,
            updated_at=time.time(),
        )

        filepath = self._session_filepath()
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            dir=self._base_dir,
            prefix=f".{self._name}.",
            suffix=".tmp",
            delete=False,
        )
        try:
            tmp.write(model.model_dump_json(indent=2))
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp.close()
            os.rename(tmp.name, filepath)
        except Exception:
            tmp.close()
            os.unlink(tmp.name)
            raise

        return filepath

    async def commit(self) -> str:
        task = Task(input="")

        proc = CommitCollectResultProcess(tools_mgr=self._tools_mgr, db=self._db)
        commit_msgs = await proc.process(task, self.messages)

        task.subTasks = list(self.uncommitted_task)

        # Write committed task to its own file
        self._commit_index += 1
        task_path = self._commit_filepath(self._commit_index)
        os.makedirs(os.path.dirname(task_path), exist_ok=True)

        with open(task_path, "w") as f:
            f.write(task.model_dump_json(indent=2))

        # Checkpoint session context

        self.messages.extend(commit_msgs)
        self.uncommitted_task.clear()

        self.checkpoint()

        return task_path

    @staticmethod
    def list_sessions(base_dir: str = "./sessions") -> list[str]:
        if not os.path.isdir(base_dir):
            return []
        return sorted(
            f[:-5] for f in os.listdir(base_dir) if f.endswith(".json")
        )
