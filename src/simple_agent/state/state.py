"""State module for agent state management."""

from __future__ import annotations

import os
import tempfile
import time
from typing import Generic, TypeVar

from pydantic import BaseModel

from pi.agent.types import AgentMessage, AgentToolResult
from pi.ai.types import ToolCall

from simple_agent.snapshot.ghost_indexer import RepoWatcher


class ToolExecMessage(BaseModel):
    tool_call: ToolCall
    raw_output: str
    tool_result: AgentToolResult


class TextResult(BaseModel):
    desc: str
    toolCallLogID: list[int]


TEXT_RESULT_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "desc": {"type": "string", "description": "Description of the result"},
        "toolCallLogID": {"type": "array", "items": {"type": "integer"}, "description": "List of tool call log IDs"},
    },
    "required": ["desc", "toolCallLogID"],
}


class Task(BaseModel):
    """A node in the task tree.

    Persisted as a flat row in SQLite.  Relationships use IDs (no
    object refs) — ``parent_id``, ``running_task_id``, and
    ``finished_task_ids`` encode the tree structure.

    Use ``Task.from_db_rows(rows)`` to rebuild the in-memory tree with
    ``running_task`` object ref for fast navigation.
    """

    type: str = "single_run"
    state: str = "PENDING"
    input: str
    result: list[TextResult] = None
    messages: list[AgentMessage] = None
    start_snapshot: str | None = None
    end_snapshot: str | None = None
    repo_watcher: RepoWatcher | None = None  # backward compat for old process classes
    # tree structure — ID-based, flat
    id: int | None = None
    parent_id: int | None = None
    running_task_id: int | None = None
    finished_task_ids: list[int] = []
    # in-memory object ref (wired by from_db_rows)
    running_task: "Task | None" = None

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data):
        super().__init__(**data)
        if self.result is None:
            self.result = []
        if self.messages is None:
            self.messages = []

    def context(self, tasks_by_id: dict[int, "Task"] | None = None) -> list[AgentMessage]:
        """Return the ancestor message chain.

        If *tasks_by_id* is provided, walks ancestors via ``parent_id``
        lookups.  Otherwise only returns ``self.messages``.
        """
        msgs: list[AgentMessage] = []
        if tasks_by_id is not None and self.parent_id is not None:
            parent = tasks_by_id.get(self.parent_id)
            if parent is not None:
                msgs.extend(parent.context(tasks_by_id))
        msgs.extend(self.messages)
        return msgs

    def find_active(self) -> "Task":
        """Walk ``running_task`` chain to find the single active node."""
        if self.state != "FINISHED" and self.running_task is None:
            return self
        if self.running_task is not None:
            return self.running_task.find_active()
        return self

    @staticmethod
    def from_db_rows(rows: list[dict]) -> dict[int, "Task"]:
        """Build Task objects from flat DB rows.

        Returns a dict mapping ``id → Task`` with ``running_task``
        object refs wired.  Callers can find the root via
        ``parent_id is None``.
        """
        tasks_by_id: dict[int, Task] = {}
        for r in rows:
            task = Task(
                id=r["id"],
                parent_id=r.get("parent_id"),
                running_task_id=r.get("running_task_id"),
                finished_task_ids=r.get("finished_task_ids", []),
                type=r["type"],
                state=r["state"],
                input=r["input"],
                messages=r.get("messages", []),
                result=r.get("result", []),
                start_snapshot=r.get("start_snapshot"),
                end_snapshot=r.get("end_snapshot"),
            )
            tasks_by_id[task.id] = task

        for task in tasks_by_id.values():
            if task.running_task_id is not None:
                task.running_task = tasks_by_id.get(task.running_task_id)

        return tasks_by_id

class RunRecord(BaseModel):
    input: str
    results: list[TextResult]
    new_message_count: int
    status: str
    started_at: float
    finished_at: float


class SessionState(BaseModel):
    """Live mutable state shared across all processes in a session.

    A single instance is passed by reference to every process.
    """

    name: str
    messages: list[AgentMessage] = []
    current_task: "Task | None" = None
    commit_index: int = 0
    uncommitted_task: list["Task"] = []
    created_at: float = 0.0
    updated_at: float = 0.0

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data):
        now = time.time()
        data.setdefault("created_at", now)
        data.setdefault("updated_at", now)
        super().__init__(**data)

    def checkpoint(self, filepath: str) -> None:
        """Persist state to *filepath* atomically (temp file + rename)."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        self.updated_at = time.time()

        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            dir=os.path.dirname(filepath) or ".",
            prefix=f".{self.name}.",
            suffix=".tmp",
            delete=False,
        )
        try:
            tmp.write(self.model_dump_json(indent=2))
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp.close()
            os.rename(tmp.name, filepath)
        except Exception:
            tmp.close()
            os.unlink(tmp.name)
            raise

    @classmethod
    def load(cls, filepath: str) -> "SessionState":
        """Restore state from a checkpoint file."""
        with open(filepath, "r") as f:
            return cls.model_validate_json(f.read())


class CommitData(BaseModel):
    extracted_instructions: list[str]
    aggregated_results: list[TextResult]


class ExtractedInstruction(BaseModel):
    instruction: str


EXTRACTED_INSTRUCTION_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "instruction": {"type": "string", "description": "A user instruction extracted from the session history"},
    },
    "required": ["instruction"],
}

def generate_state_schema(state_options: dict[str, str]) -> tuple[str, dict[str, Any]]:
    """Generate a tool schema for state clarification.

    Args:
        state_options: dict mapping state names to descriptions.
                       e.g. {"success": "Task completed", "failed": "Task failed"}

    Returns:
        Tuple of (tool_description, parameters_schema).
    """
    state_names = list(state_options.keys())
    state_desc_lines = [f"- {k}: {v}" for k, v in state_options.items()]
    state_description = "Available states:\n" + "\n".join(state_desc_lines)

    description = f"Determine the current state based on context.\n\n{state_description}"
    parameters = {
        "type": "object",
        "properties": {
            "state": {
                "type": "string",
                "description": state_description,
                "enum": state_names,
            },
            "reason": {
                "type": "string",
                "description": "Reason for choosing this state",
            },
        },
        "required": ["state", "reason"],
    }
    return description, parameters

class StateClarification(BaseModel):
    state: str
    reason: str