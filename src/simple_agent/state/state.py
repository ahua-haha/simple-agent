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
    """A node in the task tree with persisted execution state.

    Each task has a type (determines tools/prompts), a state (drives the
    state machine), its own message queue, and tree links to parent and
    children.  The ``running_task`` chain encodes the single active
    execution path — exactly one leaf is non-FINISHED at any checkpoint.
    """

    type: str = "single_run"
    state: str = "PENDING"
    input: str
    result: list[TextResult] = None
    messages: list[AgentMessage] = None
    parent: "Task | None" = None
    running_task: "Task | None" = None
    finished_tasks: list["Task"] = None
    subTasks: list["Task"] = None  # backward compat — alias for finished_tasks
    start_snapshot: str | None = None
    end_snapshot: str | None = None
    repo_watcher: RepoWatcher | None = None

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data):
        super().__init__(**data)
        if self.finished_tasks is None:
            self.finished_tasks = []
        if self.subTasks is None:
            self.subTasks = []
        if self.result is None:
            self.result = []
        if self.messages is None:
            self.messages = []

    def context(self) -> list[AgentMessage]:
        """Return the ancestor message chain: root.messages + ... + self.messages.

        Each task type can override or wrap this to customize context
        construction (filter, prefix, truncate).
        """
        msgs: list[AgentMessage] = []
        if self.parent is not None:
            msgs.extend(self.parent.context())
        msgs.extend(self.messages)
        return msgs

    def find_active(self) -> "Task":
        """Walk ``running_task`` chain to find the single active node."""
        if self.state != "FINISHED" and self.running_task is None:
            return self
        if self.running_task is not None:
            return self.running_task.find_active()
        return self

    def to_checkpoint(self) -> str:
        """Serialize tree to JSON, stripping parent refs to avoid cycles.

        Use ``Task.from_checkpoint(json_str)`` to restore parent refs.
        """
        import json

        def _to_dict(node: "Task") -> dict:
            return {
                "type": node.type,
                "state": node.state,
                "input": node.input,
                "result": [r.model_dump(mode="json") for r in (node.result or [])],
                "messages": [m.model_dump(mode="json") for m in (node.messages or [])],
                "running_task": _to_dict(node.running_task) if node.running_task else None,
                "finished_tasks": [_to_dict(c) for c in (node.finished_tasks or [])],
                "subTasks": [_to_dict(c) for c in (node.subTasks or [])],
                "start_snapshot": node.start_snapshot,
                "end_snapshot": node.end_snapshot,
            }

        return json.dumps(_to_dict(self), indent=2, default=str)

    @classmethod
    def from_checkpoint(cls, json_str: str) -> "Task":
        """Deserialize tree and rebuild parent references."""

        def _fix_parents(node: "Task") -> None:
            if node.running_task is not None:
                node.running_task.parent = node
                _fix_parents(node.running_task)
            for child in node.finished_tasks or []:
                child.parent = node
                _fix_parents(child)

        import json
        data = json.loads(json_str)
        task = cls.model_validate(data)
        _fix_parents(task)
        return task

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