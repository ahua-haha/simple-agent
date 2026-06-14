"""Shared task lifecycle state and helpers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from jinja2 import Environment, StrictUndefined
from pi.agent import AgentTool
from pi.agent.types import AgentMessage
from pi.ai.types import AssistantMessage, ToolCall, ToolResultMessage

from simple_agent.message_store import MessageEntry
from simple_agent.task_manager.models import ManagedTask, ToolCallTask

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
- Use a common task to explore, search, and gather context using tools — not to generate text responses.
- Use it when you need to investigate the codebase, inspect files, or collect information that informs the parent task.
- Do NOT create a common task whose goal is to produce a summary, explanation, or text answer. Sub-tasks should use tools to find and return facts.
- Keep the common task focused on one concrete exploration or search goal.

Example sub-task title: "Explore task lifecycle dispatch flow in session runner"

{% endif %}
{% if has_repo_memory_task %}
## Repo Memory Task
When to use:
- Use a repo memory task when the next step is to write durable repository memory with AgentIndex.
- Use it after exploring or changing repository structure when the useful result should be saved for future runs.
- Use it when the task is about recording concise descriptions of files, modules, or architecture.

Repo memory example: {"repo_path":".","index_db_path":".index.db"}

{% endif %}
## Task Creation Rules
- Do not invent metadata keys unless the selected task kind asks for them.
- Keep the created task focused on the next unit of work, not the whole user request.
- Use tools to explore and gather context — never generate text responses when a tool can provide the answer.
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


@dataclass
class SessionState:
    messages: list[MessageEntry]
    workspace_dir: str
    session_id: str | None = None
    database: Database | None = None
    next_message_id: int = 1
    next_tool_call_log_id: int = 0
    next_task_id_to_allocate: int | None = None
    current_task_id: int | None = None
    current_task: ManagedTask | None = None
    next_phase: str | None = None  # "common_task" | "orchestrator" | None (use task.kind)
    task_plan: str | None = None  # orchestrator: markdown task plan for the current task tree

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

    def set_current_task(self, task_id: int | None, task: ManagedTask | None) -> None:
        self.current_task_id = task_id
        self.current_task = task

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
    tool_call_log_ids: list[int]
    has_tool_call: bool


class BaseTaskLifecycle:
    _session_state: SessionState
    task: ManagedTask | None
    finished_task: ManagedTask | None

    def clear_data(self) -> None:
        self.task = None
        self.finished_task = None

    def set_data(self, session_state: SessionState) -> None:
        self._session_state = session_state
        self.task = None
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
        tools may mutate task data and lifecycle turn indicators such as
        task_to_start or finished_task, but tool handlers must not stamp
        message metadata, route next_task, or sync database data.
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
        tool_call_log_ids: list[int] = []
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
            tool_call_log_ids = [t.tool_call_log_id for t in tool_call_tasks if t.tool_call_log_id is not None]

        tool_result_entries = [
            MessageEntry(id=self._session_state.allocate_message_id(), message=tool_result)
            for tool_result in tool_results
        ]
        return AgentTurnResult(
            assistant_message=assistant_message,
            assistant_entry=assistant_entry,
            tool_result_entries=tool_result_entries,
            tool_call_records=tool_call_records,
            tool_call_log_ids=tool_call_log_ids,
            has_tool_call=has_tool_call,
        )

    def set_current_task(self, task_id: int | None, task: ManagedTask | None) -> None:
        self._session_state.set_current_task(task_id, task)

    def stamp_finished_task(self, *, end_message_id: int) -> ManagedTask | None:
        task = self.finished_task
        if task is None:
            return None
        if hasattr(task, "end_message_id"):
            task.end_message_id = end_message_id
        task.touch()
        return task

    def clear_turn_indicators(self) -> None:
        self.finished_task = None

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
