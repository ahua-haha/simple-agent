"""Session - orchestrates multiple SingleRunProcess runs with persistence."""

from __future__ import annotations

import os
import tempfile
import time

from pi.agent.types import AgentMessage

from simple_agent.process.single_run_process import SingleRunProcess
from simple_agent.process.commit_collect_result_process import CommitCollectResultProcess
from simple_agent.state.state import CommitData, RunRecord, SessionData, SingleRunTask, Task


class Session:
    messages: list[AgentMessage]
    runs: list[RunRecord]
    commit_data: CommitData | None
    _name: str
    _base_dir: str
    _created_at: float

    def __init__(self, name: str, base_dir: str = "./sessions"):
        self._name = name
        self._base_dir = base_dir
        self._created_at = time.time()

        filepath = self._filepath()
        if os.path.exists(filepath):
            self._load(filepath)
        else:
            self.messages = []
            self.runs = []
            self.commit_data = None

    def _filepath(self) -> str:
        return os.path.join(self._base_dir, f"{self._name}.json")

    def _load(self, filepath: str) -> None:
        with open(filepath, "r") as f:
            data = SessionData.model_validate_json(f.read())
        self.messages = data.messages
        self.runs = data.runs
        self.commit_data = data.commit_data
        self._created_at = data.created_at

    async def run(self, user_input: str) -> SingleRunTask:
        task = SingleRunTask(input=user_input)
        started_at = time.time()

        proc = SingleRunProcess()
        new_msgs = await proc.process(task, context=self.messages)
        self.messages.extend(new_msgs)

        finished_at = time.time()
        record = RunRecord(
            input=user_input,
            results=task.result or [],
            new_message_count=len(new_msgs),
            status="finished",
            started_at=started_at,
            finished_at=finished_at,
        )
        self.runs.append(record)
        return task

    async def commit(self) -> str:
        task = Task(input="")

        proc = CommitCollectResultProcess()
        await proc.process(task, self.messages)

        self.commit_data = proc.commit_data

        # Persist to JSON
        os.makedirs(self._base_dir, exist_ok=True)

        model = SessionData(
            name=self._name,
            messages=self.messages,
            runs=self.runs,
            commit_data=self.commit_data,
            created_at=self._created_at,
            updated_at=time.time(),
        )

        filepath = self._filepath()
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

    @staticmethod
    def list_sessions(base_dir: str = "./sessions") -> list[str]:
        if not os.path.isdir(base_dir):
            return []
        return sorted(
            f[:-5] for f in os.listdir(base_dir) if f.endswith(".json")
        )
