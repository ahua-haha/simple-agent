"""Builder for agent-created next tasks."""

from __future__ import annotations

from collections.abc import Callable

from pi.agent import AgentTool, AgentToolResult
from pi.ai.types import TextContent

from simple_agent.task_manager.lifecycle import SessionState, TaskLifecycleError
from simple_agent.task_manager.models import ManagedTask, RepoMemoryTask, TodoTask


SUPPORTED_TASK_KINDS: tuple[str, ...] = ("todo", "repo_memory")


class NextTaskBuilder:
    """Create next-task tools bound to a SessionState."""

    def __init__(
        self,
        session_state: SessionState,
        *,
        enabled_task_kinds: list[str] | tuple[str, ...] | None = None,
        current_assistant_message_id: Callable[[], int | None] | None = None,
    ):
        self._session_state = session_state
        self._enabled_task_kinds = list(enabled_task_kinds or SUPPORTED_TASK_KINDS)
        self._current_assistant_message_id = current_assistant_message_id
        invalid = [kind for kind in self._enabled_task_kinds if kind not in SUPPORTED_TASK_KINDS]
        if invalid:
            raise TaskLifecycleError(f"Unsupported task kind enabled: {invalid[0]}")

    def instruction_text(self) -> str:
        lines = [
            "Next task builder:",
            "- Use create_next_task before switching to a different unit of work.",
        ]
        if "todo" in self._enabled_task_kinds:
            lines.append(
                "- kind=todo: use for the next small atomic implementation, debugging, or inspection step."
            )
        if "repo_memory" in self._enabled_task_kinds:
            lines.append(
                "- kind=repo_memory: use when the next step is to write durable repo memory with AgentIndex."
            )
        lines.append("- Put task-specific fields in metadata.")
        lines.append("- Create only one next task at a time.")
        return "\n".join(lines)

    def create_tools(self) -> list[AgentTool]:
        return [self.create_task_tool()]

    def create_task_tool(self) -> AgentTool:
        tool = AgentTool(
            name="create_next_task",
            description=(
                "Create the next task for this session. Use this before moving "
                "from the current task to a todo or repo-memory task."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": self._enabled_task_kinds,
                        "description": "The type of next task to create.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Short title for the next task.",
                    },
                    "metadata": {
                        "type": "object",
                        "description": (
                            "Task-specific metadata. For repo_memory include "
                            "repo_path and index_db_path. Todo tasks usually omit this."
                        ),
                        "additionalProperties": True,
                    },
                },
                "required": ["kind", "title"],
            },
        )

        async def execute(tool_call_id, params, cancel_event=None, on_update=None):
            task = self.create_task(
                kind=params["kind"],
                title=params["title"],
                metadata=params.get("metadata"),
            )
            return AgentToolResult(content=[TextContent(text=f"Created next task: {task.kind} {task.title}")])

        tool.execute = execute
        return tool

    def create_task(
        self,
        *,
        kind: str,
        title: str,
        metadata: dict | None = None,
    ) -> ManagedTask:
        if kind not in self._enabled_task_kinds:
            raise TaskLifecycleError(f"Task kind is disabled: {kind}")
        parent = self._require_parent_task()
        metadata = metadata or {}
        if kind == "todo":
            task: ManagedTask = TodoTask(
                id=self._session_state.allocate_task_id(),
                parent_id=parent.id,
                title=title,
                start_message_id=self._read_current_assistant_message_id(),
            )
        elif kind == "repo_memory":
            repo_path = metadata.get("repo_path")
            index_db_path = metadata.get("index_db_path")
            if index_db_path is None:
                raise TaskLifecycleError("repo_memory task requires index_db_path")
            task = RepoMemoryTask(
                id=self._session_state.allocate_task_id(),
                parent_id=parent.id,
                title=title,
                repo_path=repo_path or ".",
                index_db_path=index_db_path,
            )
        else:
            raise TaskLifecycleError(f"Unsupported next task kind: {kind}")

        parent.children.append(task)
        parent.touch()
        self._session_state.set_next_task(task, keep_instance=True)
        return task

    def _read_current_assistant_message_id(self) -> int | None:
        if self._current_assistant_message_id is None:
            return None
        return self._current_assistant_message_id()

    def _require_parent_task(self) -> ManagedTask:
        parent = self._session_state.next_task
        if parent is None:
            raise TaskLifecycleError("Session state has no active task to attach next task")
        if parent.id is None:
            raise TaskLifecycleError("Active task must have an id before creating a next task")
        return parent
