"""State module for agent state management."""

from __future__ import annotations

from pydantic import BaseModel

from pi.agent.types import AgentMessage, AgentToolResult
from pi.ai.types import ToolCall


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

    ``metadata`` is a runtime-only dict for objects like RepoWatcher
    and pre-built context messages.  It is NOT persisted.
    """

    type: str = "single_run"
    state: str = "PENDING"
    input: str
    result: list[TextResult] = None
    messages: list[AgentMessage] = None
    result_msg: list[AgentMessage] = []
    repo_path: str = "."
    start_snapshot: str | None = None
    end_snapshot: str | None = None
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
        if self.result_msg is None:
            self.result_msg = []
        # runtime cache, not persisted
        object.__setattr__(self, "metadata", {})

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
                result_msg=r.get("result_msg", []),
                repo_path=r.get("repo_path", "."),
                start_snapshot=r.get("start_snapshot"),
                end_snapshot=r.get("end_snapshot"),
            )
            tasks_by_id[task.id] = task

        for task in tasks_by_id.values():
            if task.running_task_id is not None:
                task.running_task = tasks_by_id.get(task.running_task_id)

        return tasks_by_id

class StateClarification(BaseModel):
    state: str
    reason: str