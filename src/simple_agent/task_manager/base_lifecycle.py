"""Shared task lifecycle state and helpers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from jinja2 import Environment, StrictUndefined
from pi.agent import AgentTool, AgentToolResult
from pi.agent.types import AgentMessage
from pi.ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage

from simple_agent.message_store import MessageEntry
from simple_agent.task_manager.models import CommonTask, ManagedTask, RepoMemoryTask, ToolCallTask

if TYPE_CHECKING:
    from simple_agent.db.db import Database
    from simple_agent.process.agent_process import AgentProcess
    from sqlmodel import Session as SqlSession


class TaskLifecycleError(RuntimeError):
    """Raised when a task lifecycle cannot complete an operation."""


_PROMPT_ENV = Environment(
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_prompt_template(template: str, **context) -> str:
    return _PROMPT_ENV.from_string(template).render(**context).strip()


TASK_INSTRUCTION_TEMPLATE = """\
{% if has_common_task %}
## Common Task
When to use:
- Use a common task for a recursive subtask that should have the same behavior as the current task.
- Use it when the next unit of work may need its own decomposition, tool calls, finish handling, and compaction.
- Keep the common task focused on one meaningful sub-goal, not the whole user request.

How to create:
- Call `create_next_task(kind="common", title="<focused subtask goal>", metadata={})`.
- Put the concrete subtask goal in the title.
- Omit metadata or pass an empty object.
- Example: {"kind":"common","title":"Summarize session runner lifecycle flow","metadata":{}}

{% endif %}
{% if has_repo_memory_task %}
## Repo Memory Task
When to use:
- Use a repo memory task when the next step is to write durable repository memory with AgentIndex.
- Use it after exploring or changing repository structure when the useful result should be saved for future runs.
- Use it when the task is about recording concise descriptions of files, modules, or architecture.

How to create:
- Call `create_next_task(kind="repo_memory", title="<memory writing goal>", metadata={...})`.
- Metadata must include `index_db_path`.
- Metadata may include `repo_path`; omit it when the current repository root is correct.
- Metadata shape: {"repo_path":"<repo path>","index_db_path":"<index database path>"}.
- Example: {"kind":"repo_memory","title":"Write memory for task lifecycle design","metadata":{"repo_path":".","index_db_path":".agent-index.db"}}

{% endif %}
## Task Creation Rules
- Do not invent metadata keys unless the selected task kind asks for them.
- Keep the created task focused on the next unit of work, not the whole user request.
"""


def task_instruction_text(*, has_common_task: bool, has_repo_memory_task: bool) -> str:
    return render_prompt_template(
        TASK_INSTRUCTION_TEMPLATE,
        has_common_task=has_common_task,
        has_repo_memory_task=has_repo_memory_task,
    )


USER_TASK_SYSTEM_PROMPT = """You are a helpful coding agent.

Be concise, practical, and honest about uncertainty. Use available tools
when they are needed, and explain outcomes clearly.

Runtime steering instructions may arrive in a user message wrapped with
<system-instruction> tags. Treat the content inside that tag as high-priority
instruction for the current turn, without repeating the tag in your response.
"""

USER_TASK_COMPACT_SYSTEM_PROMPT = """Compact the finished user task.
Use only the compact tools to record useful tool-call log IDs. When all useful
tool-call logs are recorded, respond without tool calls to finish compaction."""


@dataclass
class SessionState:
    messages: list[MessageEntry]
    workspace_dir: str
    session_id: str | None = None
    database: Database | None = None
    next_message_id: int = 1
    next_tool_call_log_id: int = 0
    next_task_id_to_allocate: int | None = None
    next_task_id_to_run: int | None = None
    next_task: ManagedTask | None = None

    def allocate_message_id(self) -> int:
        message_id = self.next_message_id
        self.next_message_id += 1
        return message_id

    def allocate_task_id(self) -> int:
        if self.next_task_id_to_allocate is None:
            raise TaskLifecycleError("Session state is missing task allocation state")
        task_id = self.next_task_id_to_allocate
        self.next_task_id_to_allocate += 1
        return task_id

    def allocate_tool_call_log_id(self) -> int:
        tool_call_log_id = self.next_tool_call_log_id
        self.next_tool_call_log_id += 1
        return tool_call_log_id

    def message_values(self) -> list[AgentMessage]:
        return [entry.message for entry in self.messages]

    def append_message(self, message: AgentMessage) -> MessageEntry:
        entry = MessageEntry(id=self.allocate_message_id(), message=message)
        self.messages.append(entry)
        return entry

    def append_messages(self, messages: list[AgentMessage | MessageEntry]) -> list[MessageEntry]:
        entries: list[MessageEntry] = []
        for message in messages:
            if isinstance(message, MessageEntry):
                entry = message
                self.messages.append(entry)
                self.next_message_id = max(self.next_message_id, entry.id + 1)
            else:
                entry = self.append_message(message)
            entries.append(entry)
        return entries

    def replace_message_range(
        self,
        *,
        start_message_id: int,
        end_message_id: int,
        replacement_messages: list[AgentMessage],
    ) -> list[MessageEntry]:
        start_index = self._message_index(start_message_id)
        end_index = self._message_index(end_message_id)
        if end_index < start_index:
            raise RuntimeError("Compact end message is before compact start message")
        replacement_entries = [
            MessageEntry(id=self.allocate_message_id(), message=message)
            for message in replacement_messages
        ]
        self.messages = [
            *self.messages[:start_index],
            *replacement_entries,
            *self.messages[end_index + 1:],
        ]
        return replacement_entries

    def create_tool_call_record_task_entries(
        self,
        *,
        assistant_message: AssistantMessage,
        tool_result_messages: list[ToolResultMessage],
        parent_task: ManagedTask,
    ) -> tuple[list[tuple[int, ToolCall | None, ToolResultMessage]], list[ToolCallTask]]:
        tool_call_records: list[tuple[int, ToolCall | None, ToolResultMessage]] = []
        tool_call_tasks: list[ToolCallTask] = []
        for tool_result_message in tool_result_messages:
            log_id = self.allocate_tool_call_log_id()
            tool_call = _tool_call_for_result(
                assistant_message=assistant_message,
                tool_result_message=tool_result_message,
            )
            tool_call_records.append((log_id, tool_call, tool_result_message))
            tool_call_tasks.append(
                ToolCallTask(
                    id=self.allocate_task_id(),
                    status="done",
                    parent_id=parent_task.id,
                    tool_call_log_id=log_id,
                    tool_call_name=tool_call.name if tool_call is not None else tool_result_message.tool_name,
                    tool_call_args=tool_call.arguments if tool_call is not None else None,
                )
            )
        return tool_call_records, tool_call_tasks

    def compacted_tool_calls(self, tool_call_log_ids: list[int]) -> list[tuple[ToolCall | None, ToolResultMessage]]:
        database = self._require_database()
        session_id = self._require_session_id()
        records: list[tuple[ToolCall | None, ToolResultMessage]] = []
        with self.create_database_session() as session:
            for tool_call_log_id in tool_call_log_ids:
                record = database.get_runner_tool_call(
                    session_id,
                    tool_call_log_id,
                    session=session,
                )
                if record is None:
                    continue
                tool_call = None
                if record.tool_call_json != "null":
                    tool_call = ToolCall.model_validate_json(record.tool_call_json)
                records.append((tool_call, ToolResultMessage.model_validate_json(record.tool_result_json)))
        return records

    def create_database_session(self) -> SqlSession:
        return self._require_database().create_session()

    def append_messages_to_database(
        self,
        *,
        messages: list[MessageEntry],
        session: SqlSession,
    ) -> None:
        database = self._require_database()
        session_id = self._require_session_id()
        for message in messages:
            database.insert_runner_message(
                session_id,
                message.message,
                id=message.id,
                session=session,
            )

    def append_tool_calls_to_database(
        self,
        *,
        tool_calls: list[tuple[int, ToolCall | None, ToolResultMessage]],
        session: SqlSession,
    ) -> None:
        database = self._require_database()
        session_id = self._require_session_id()
        for log_id, tool_call, tool_result in tool_calls:
            database.insert_runner_tool_call(
                id=log_id,
                session_id=session_id,
                tool_call_id=tool_result.tool_call_id,
                tool_name=tool_result.tool_name,
                tool_call_json=tool_call.model_dump_json() if tool_call is not None else "null",
                tool_result_json=tool_result.model_dump_json(),
                session=session,
            )

    def append_tasks_to_database(
        self,
        *,
        tasks: list[ManagedTask],
        session: SqlSession,
    ) -> None:
        database = self._require_database()
        for task in tasks:
            database.upsert_managed_task(task, session=session)

    def replace_message_range_in_database(
        self,
        *,
        start_message_id: int,
        end_message_id: int,
        replacement_messages: list[MessageEntry],
        session: SqlSession,
    ) -> None:
        database = self._require_database()
        session_id = self._require_session_id()
        start_seq = database.get_runner_message_seq(
            session_id,
            start_message_id,
            session=session,
        )
        end_seq = database.get_runner_message_seq(
            session_id,
            end_message_id,
            session=session,
        )
        if end_seq < start_seq:
            raise RuntimeError("Compact end message is before compact start message")
        database.delete_runner_message_seq_range(
            session_id,
            start_seq=start_seq,
            end_seq=end_seq,
            session=session,
        )
        for message in replacement_messages:
            database.insert_runner_message(
                session_id,
                message.message,
                id=message.id,
                session=session,
            )

    def set_next_task(self, task_id: int | None, task: ManagedTask | None) -> None:
        self.next_task_id_to_run = task_id
        self.next_task = task

    def _message_index(self, message_id: int) -> int:
        for index, entry in enumerate(self.messages):
            if entry.id == message_id:
                return index
        raise RuntimeError(f"Could not find message id {message_id}")

    def _require_database(self) -> Database:
        if self.database is None:
            raise TaskLifecycleError("Session state is missing database")
        return self.database

    def _require_session_id(self) -> str:
        if self.session_id is None:
            raise TaskLifecycleError("Session state is missing session id")
        return self.session_id


@dataclass
class AgentTurnResult:
    assistant_message: AssistantMessage
    assistant_entry: MessageEntry
    tool_result_entries: list[MessageEntry]
    tool_call_records: list[tuple[int, ToolCall | None, ToolResultMessage]]
    tool_call_tasks: list[ToolCallTask]
    has_tool_call: bool


class BaseTaskLifecycle:
    _session_state: SessionState
    task: ManagedTask | None
    created_task: list[ManagedTask]
    task_to_start: ManagedTask | None
    finished_task: ManagedTask | None

    def clear_data(self) -> None:
        self.task = None
        self.created_task = []
        self.task_to_start = None
        self.finished_task = None

    def set_data(self, session_state: SessionState) -> None:
        self._session_state = session_state
        self.task = None
        self.created_task = []
        self.task_to_start = None
        self.finished_task = None
        raise NotImplementedError(f"{type(self).__name__}.set_data is not implemented")

    async def run_agent_turn(
        self,
        *,
        agent_process: AgentProcess,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        parent_task: ManagedTask,
        cancel_event: asyncio.Event | None = None,
    ) -> AgentTurnResult:
        """Run one LLM/tool turn and build transient records for post handling.

        Mutates only allocation counters on SessionState while building message
        entries, tool-call log records, and tool-call task records. Executed
        tools may mutate lifecycle turn indicators such as created_task,
        task_to_start, or finished_task, but tool handlers must not allocate task ids, stamp
        message metadata, append tasks, route next_task, or sync database data.
        Those metadata and persistence changes belong in the caller's post-turn
        handler so ordering stays deterministic.
        """
        assistant_message = await agent_process.call_llm_step(
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            cancel_event=cancel_event,
        )
        if _assistant_is_error(assistant_message):
            raise TaskLifecycleError(
                assistant_message.error_message or "assistant response stopped with error"
            )

        assistant_entry = MessageEntry(
            id=self._session_state.allocate_message_id(),
            message=assistant_message,
        )
        tool_results: list[ToolResultMessage] = []
        tool_call_records: list[tuple[int, ToolCall | None, ToolResultMessage]] = []
        tool_call_tasks: list[ToolCallTask] = []
        has_tool_call = _assistant_has_tool_calls(assistant_message)
        if has_tool_call:
            tool_results = await agent_process.run_tool_calls_step(
                tools=tools,
                assistant_message=assistant_message,
                cancel_event=cancel_event,
            )
            tool_call_records, tool_call_tasks = self._session_state.create_tool_call_record_task_entries(
                assistant_message=assistant_message,
                tool_result_messages=tool_results,
                parent_task=parent_task,
            )

        tool_result_entries = [
            MessageEntry(id=self._session_state.allocate_message_id(), message=tool_result)
            for tool_result in tool_results
        ]
        return AgentTurnResult(
            assistant_message=assistant_message,
            assistant_entry=assistant_entry,
            tool_result_entries=tool_result_entries,
            tool_call_records=tool_call_records,
            tool_call_tasks=tool_call_tasks,
            has_tool_call=has_tool_call,
        )

    def create_next_task_tools(
        self,
        *,
        enabled_task_kinds: list[str] | tuple[str, ...] | None = None,
    ) -> list[AgentTool]:
        return [self.build_create_next_task_tool(enabled_task_kinds=enabled_task_kinds)]

    def build_create_next_task_tool(
        self,
        *,
        enabled_task_kinds: list[str] | tuple[str, ...] | None = None,
    ) -> AgentTool:
        enabled_kinds = self._validate_enabled_task_kinds(enabled_task_kinds)

        tool = AgentTool(
            name="create_next_task",
            description=(
                "Create the next task for this session. Use this before moving "
                "from the current task to a common or repo-memory task."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": enabled_kinds,
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
                            "repo_path and index_db_path. Common tasks usually omit this."
                        ),
                        "additionalProperties": True,
                    },
                },
                "required": ["kind", "title"],
            },
        )

        async def execute(tool_call_id, params, cancel_event=None, on_update=None):
            task = self.create_next_task(
                kind=params["kind"],
                title=params["title"],
                metadata=params.get("metadata"),
                enabled_task_kinds=enabled_kinds,
            )
            return AgentToolResult(content=[TextContent(text=f"Created next task: {task.kind} {task.title}")])

        tool.execute = execute
        return tool

    def build_start_next_task_tool(self) -> AgentTool:
        tool = AgentTool(
            name="start_next_task",
            description="Start an existing task by id and make it the next task to run.",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "integer",
                        "description": "The id of the task to start.",
                    },
                },
                "required": ["task_id"],
            },
        )

        async def execute(tool_call_id, params, cancel_event=None, on_update=None):
            task = self.start_next_task(task_id=params["task_id"])
            return AgentToolResult(content=[TextContent(text=f"Start next task: {task.kind} {task.id}")])

        tool.execute = execute
        return tool

    def create_next_task(
        self,
        *,
        kind: str,
        title: str,
        metadata: dict | None = None,
        enabled_task_kinds: list[str] | tuple[str, ...] | None = None,
    ) -> ManagedTask:
        enabled_kinds = self._validate_enabled_task_kinds(enabled_task_kinds)
        if kind not in enabled_kinds:
            raise TaskLifecycleError(f"Task kind is disabled: {kind}")
        parent = self._require_task()
        metadata = metadata or {}
        if kind == "common":
            task: ManagedTask = CommonTask(
                parent_id=parent.id,
                title=title,
            )
        elif kind == "repo_memory":
            repo_path = metadata.get("repo_path")
            index_db_path = metadata.get("index_db_path")
            if index_db_path is None:
                raise TaskLifecycleError("repo_memory task requires index_db_path")
            task = RepoMemoryTask(
                parent_id=parent.id,
                title=title,
                repo_path=repo_path or self._session_state.workspace_dir,
                index_db_path=index_db_path,
            )
        else:
            raise TaskLifecycleError(f"Unsupported next task kind: {kind}")

        self.created_task.append(task)
        return task

    def start_next_task(self, *, task_id: int) -> ManagedTask:
        task = self._find_task_to_start(task_id=task_id)
        self.task_to_start = task
        return task

    def set_next_task(self, task_id: int | None, task: ManagedTask | None) -> None:
        self._session_state.set_next_task(task_id, task)

    def stamp_finished_task(self, *, end_message_id: int) -> ManagedTask | None:
        task = self.finished_task
        if task is None:
            return None
        if hasattr(task, "end_message_id"):
            task.end_message_id = end_message_id
        task.touch()
        return task

    def clear_turn_indicators(self) -> None:
        self.created_task = []
        self.task_to_start = None
        self.finished_task = None

    def _validate_enabled_task_kinds(
        self,
        enabled_task_kinds: list[str] | tuple[str, ...] | None,
    ) -> list[str]:
        enabled_kinds = list(enabled_task_kinds or ("common", "repo_memory"))
        invalid = [kind for kind in enabled_kinds if kind not in ("common", "repo_memory")]
        if invalid:
            raise TaskLifecycleError(f"Unsupported task kind enabled: {invalid[0]}")
        return enabled_kinds

    def _require_task(self) -> ManagedTask:
        parent = self.task
        if parent is None:
            raise TaskLifecycleError("Lifecycle has no current task to attach next task")
        if parent.id is None:
            raise TaskLifecycleError("Current task must have an id before creating a next task")
        return parent

    def _find_task_to_start(self, *, task_id: int) -> ManagedTask:
        for created_task in self.created_task:
            if created_task.id == task_id:
                return created_task
        parent = self._require_task()
        for child in parent.children:
            if child.id == task_id:
                return child
        raise TaskLifecycleError(f"Task does not exist under current task: {task_id}")

    async def run(
        self,
        *,
        agent_process: AgentProcess,
        cancel_event: asyncio.Event | None = None,
    ) -> SessionState:
        raise NotImplementedError(f"{type(self).__name__}.run is not implemented")


def _assistant_has_tool_calls(message: AssistantMessage) -> bool:
    return any(isinstance(content, ToolCall) for content in message.content)


def _assistant_is_error(message: AssistantMessage) -> bool:
    return message.stop_reason == "error" or bool(message.error_message)


def _tool_call_for_result(
    *,
    assistant_message: AssistantMessage,
    tool_result_message: ToolResultMessage,
) -> ToolCall | None:
    for content in assistant_message.content:
        if isinstance(content, ToolCall) and content.id == tool_result_message.tool_call_id:
            return content
    return None
