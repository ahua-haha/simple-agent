"""SessionRunner owns the persisted Session.run workflow."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Literal

from pi.ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage
from simple_agent.fractional_index import key_after
from simple_agent.json_utils import json_safe
from simple_agent.state.state import RunnerMessageEntry
from simple_agent.token_estimation import estimate_messages_tokens
from simple_agent.tool.common_tools import create_all_coding_tools

if TYPE_CHECKING:
    from simple_agent.db.db import Database
    from simple_agent.process.agent_process import AgentProcess
    from simple_agent.task_manager import TaskManager
    from pi.agent.types import AgentMessage

_log = logging.getLogger(__name__)

RunnerPhase = Literal["idle", "new_user_task", "running", "compact", "done", "error"]

SYSTEM_PROMPT = """You are a helpful coding agent.

IMPORTANT: Manage your task list for the current session with the todo
tools. For complex or long-running tasks, you must decompose the work and
use create_todo to explicitly define the next thing to do before doing
that work.

Todos must be small and atomic. Do not create broad todos that combine
multiple steps. Before you try to do the next unit of work, including
calling any non-todo tool, first create the todo that describes that unit
of work. Only one todo may be active at a time.

Call finish_todo immediately when the active todo is complete. If
something fails, call error_todo for the active todo and create a revised
todo when there is a clear next step.

Keep responses concise and use available tools to do the work.
"""

DEFAULT_CONTEXT_TOKEN_THRESHOLD = 120_000
DEFAULT_TOOL_CALL_THRESHOLD = 200

COMPACT_SYSTEM_PROMPT = """Compact finished todos into one compacted todo.
Use only the compact tools. Create exactly one compacted todo, record useful
tool-call log IDs, then finish the compacted todo."""


def _tool_result_payload(result: ToolResultMessage) -> dict[str, Any]:
    return {
        "content": [json_safe(item) for item in result.content],
        "details": json_safe(result.details),
    }


def _tool_result_error(result: ToolResultMessage) -> str | None:
    if not result.is_error:
        return None
    text_parts = [
        item.text
        for item in result.content
        if isinstance(item, TextContent)
    ]
    return "\n".join(text_parts) if text_parts else "Tool call failed"


class SessionRunner:
    """Persisted runner for one Session.run invocation at a time."""

    _session_id: str
    _db: Database
    _task_manager: TaskManager
    _agent_process: AgentProcess
    _cancel_event: asyncio.Event
    _phase: RunnerPhase
    _active_user_task_id: int | None
    _messages: list[RunnerMessageEntry]
    _next_message_seq: str | None
    _context_token_threshold: int
    _tool_call_threshold: int
    _user_paused: bool

    def __init__(
        self,
        *,
        session_id: str,
        db: Database,
        task_manager: TaskManager,
        agent_process: AgentProcess,
        cancel_event: asyncio.Event,
        context_token_threshold: int = DEFAULT_CONTEXT_TOKEN_THRESHOLD,
        tool_call_threshold: int = DEFAULT_TOOL_CALL_THRESHOLD,
    ):
        self._session_id = session_id
        self._db = db
        self._task_manager = task_manager
        self._agent_process = agent_process
        self._cancel_event = cancel_event
        self._phase = "idle"
        self._active_user_task_id = None
        self._messages = []
        self._next_message_seq = key_after(None)
        self._context_token_threshold = context_token_threshold
        self._tool_call_threshold = tool_call_threshold
        self._user_paused = False

    def subscribe(self, callback: Callable) -> None:
        self._agent_process.subscribe(callback)

    def unsubscribe(self, callback: Callable) -> None:
        self._agent_process.unsubscribe(callback)

    def pause(self) -> None:
        self._user_paused = True
        self._cancel_event.set()

    def load(self) -> None:
        metadata = self._db.get_runner_state_metadata(self._session_id)
        self._messages = self._db.list_runner_message_entries(self._session_id)
        self._next_message_seq = key_after(self._messages[-1].seq if self._messages else None)
        if metadata is None:
            self._phase = "idle"
            self._active_user_task_id = None
            self._task_manager.load(None)
            return
        self._phase = metadata.phase
        self._active_user_task_id = metadata.active_user_task_id
        self._task_manager.load(metadata.active_user_task_id)

    def save_metadata(self, *, status: str | None = None, last_error: str | None = None, session=None) -> None:
        self._db.upsert_runner_state_metadata(
            self._session_id,
            phase=self._phase,
            status=status or self._phase,
            active_user_task_id=self._active_user_task_id,
            last_error=last_error,
            session=session,
        )

    def append_messages(self, messages: list[AgentMessage], *, session=None) -> None:
        if session is None:
            with self._db.create_session() as session:
                self.append_messages(messages, session=session)
                session.commit()
            return

        entries = self._create_message_entries(messages)
        self._messages.extend(entries)
        self._next_message_seq = key_after(self._messages[-1].seq if self._messages else None)
        for entry in entries:
            self._db.insert_runner_message_entry(self._session_id, entry, session=session)

    def record_tool_call(
        self,
        tool_call: ToolCall | None,
        tool_result: ToolResultMessage,
        *,
        started_at: float,
        finished_at: float,
        session=None,
    ) -> int:
        if session is None:
            with self._db.create_session() as session:
                log_id = self.record_tool_call(
                    tool_call,
                    tool_result,
                    started_at=started_at,
                    finished_at=finished_at,
                    session=session,
                )
                self._task_manager.save(session=session)
                session.commit()
                return log_id

        log_id = self._db.next_runner_tool_call_id(self._session_id, session=session)
        self._task_manager.record_tool_call(log_id)
        self._db.insert_runner_tool_call(
            id=log_id,
            session_id=self._session_id,
            tool_call_id=tool_result.tool_call_id,
            tool_name=tool_result.tool_name,
            params=tool_call.arguments if tool_call is not None else {},
            result=_tool_result_payload(tool_result),
            status="error" if tool_result.is_error else "success",
            started_at=started_at,
            finished_at=finished_at,
            error=_tool_result_error(tool_result),
            session=session,
        )
        return log_id

    def record_tool_calls(
        self,
        tool_call_records: list[tuple[ToolCall | None, ToolResultMessage, float, float]],
        *,
        session=None,
    ) -> None:
        if session is None:
            with self._db.create_session() as session:
                self.record_tool_calls(tool_call_records, session=session)
                self._task_manager.save(session=session)
                session.commit()
            return

        for tool_call, tool_result, started_at, finished_at in tool_call_records:
            self.record_tool_call(
                tool_call,
                tool_result,
                started_at=started_at,
                finished_at=finished_at,
                session=session,
            )

    def save_current_data(
        self,
        messages: list[AgentMessage] | None = None,
        *,
        status: str | None = None,
        last_error: str | None = None,
        save_tasks: bool = True,
        tool_call_records: list[tuple[ToolCall | None, ToolResultMessage, float, float]] | None = None,
    ) -> None:
        messages = messages or []
        tool_call_records = tool_call_records or []
        with self._db.create_session() as session:
            self.record_tool_calls(tool_call_records, session=session)
            self.append_messages(messages, session=session)
            if save_tasks:
                self._task_manager.save(session=session)
            self.save_metadata(status=status or self._phase, last_error=last_error, session=session)
            session.commit()

    def _create_tools(self):
        return [
            self._task_manager.create_create_todo_tool(),
            self._task_manager.create_finish_todo_tool(),
            self._task_manager.create_error_todo_tool(),
            *create_all_coding_tools("."),
        ]

    async def run(self, user_input: str | None):
        self._user_paused = False
        self._cancel_event.clear()
        self.load()
        self.handle_input(user_input)
        try:
            while self._phase != "done":
                if self._user_paused:
                    break
                if self._phase in ("idle", "error"):
                    break
                if self._phase in ("new_user_task", "running"):
                    await self.handle_running(user_input)
                    continue
                if self._phase == "compact":
                    await self.handle_compact(user_input)
                    continue
                raise RuntimeError(f"Unknown runner phase: {self._phase}")
        except Exception as exc:
            self.handle_error(exc)
            raise

        if self._active_user_task_id is None:
            return None
        return self._db.get_managed_task(self._active_user_task_id)

    def handle_input(self, user_input: str | None) -> None:
        if user_input is None:
            return

        self.finish_previous_user_task()
        self._task_manager.load(None)
        user_task = self._task_manager.create_user_task(user_input)
        self._active_user_task_id = user_task.id
        self._phase = "new_user_task"
        self._cancel_event.clear()
        self.save_current_data(status="new_user_task")

    def finish_previous_user_task(self) -> None:
        previous_user_task = self._task_manager.active_user_task
        if previous_user_task is None:
            return

        if previous_user_task.status != "done":
            if self._task_manager.active_todo_id is not None:
                self._task_manager.error_task("Interrupted by new user input")
            self._task_manager.finish_user_task()
            self.save_current_data(messages=[], status="done")
        self._active_user_task_id = None

    async def handle_running(self, user_input: str | None) -> None:
        if self._phase == "new_user_task" and user_input is not None:
            self.save_current_data(messages=[self._create_user_message(user_input)], status="new_user_task")

        while self._phase in ("new_user_task", "running"):
            tools = self._create_tools()
            assistant_message = await self._agent_process.call_llm_step(
                system_prompt=SYSTEM_PROMPT,
                messages=self._agent_messages(),
                tools=tools,
                cancel_event=self._cancel_event,
            )
            tool_results: list[ToolResultMessage] = []
            tool_call_records: list[tuple[ToolCall | None, ToolResultMessage, float, float]] = []
            if self._assistant_has_tool_calls(assistant_message) and not self._cancel_event.is_set():
                tool_step_started_at = time.time()
                tool_results = await self._agent_process.run_tool_calls_step(
                    tools=tools,
                    assistant_message=assistant_message,
                    cancel_event=self._cancel_event,
                )
                tool_step_finished_at = time.time()
                tool_call_records = self._create_tool_call_records(
                    assistant_message,
                    tool_results,
                    tool_step_started_at,
                    tool_step_finished_at,
                )

            if self._phase == "new_user_task":
                self._phase = "running"
            self.save_current_data(
                [assistant_message, *tool_results],
                status=self._phase,
                tool_call_records=tool_call_records,
            )
            self.pause_for_compaction_if_needed()

            if self._phase == "compact":
                return
            if self._task_manager.active_todo_id is None:
                self._task_manager.finish_user_task()
                self._phase = "done"
                self.save_current_data(status="done")
                return
            if self._user_paused:
                return
            if not tool_results:
                return

    def pause_for_compaction_if_needed(self) -> None:
        context_tokens = estimate_messages_tokens(self._agent_messages())
        tool_calls = len(self._db.list_runner_tool_calls(self._session_id))
        if (
            context_tokens <= self._context_token_threshold
            and tool_calls <= self._tool_call_threshold
        ):
            return

        self._phase = "compact"
        self.save_current_data(messages=[], status="compact", save_tasks=False)
        self._cancel_event.set()

    async def handle_compact(self, user_input: str | None) -> None:
        scope = self._task_manager.compact_scope()
        if scope is None:
            self._phase = "running"
            self._cancel_event.clear()
            self.save_current_data(messages=[], status="running", save_tasks=False)
            return

        start_tool_call_id = scope.compact_todos[0].create_tool_call_id
        if start_tool_call_id is None:
            raise RuntimeError("Compact start todo is missing create_tool_call_id")
        start_seq = self.find_assistant_message_seq_for_tool_call(start_tool_call_id)

        end_tool_call_id = scope.compact_todos[-1].end_tool_call_id
        if end_tool_call_id is None:
            raise RuntimeError("Compact end todo is missing end_tool_call_id")
        end_seq = self.find_tool_result_message_seq_for_tool_call(end_tool_call_id)
        if end_seq < start_seq:
            raise RuntimeError("Compact end message is before start message")

        messages_before_start = self._messages[:start_seq]
        preserved_messages = [entry.message for entry in self._messages[end_seq + 1:]]

        self._task_manager.begin_compact_buffer()
        await self._agent_process.run(
            system_prompt=COMPACT_SYSTEM_PROMPT,
            messages=[entry.message for entry in self._messages[start_seq:end_seq + 1]],
            tools=self._task_manager.create_compact_tools(),
            user_prompt=user_input,
            cancel_event=self._cancel_event,
        )

        with self._db.create_session() as session:
            compacted_todo = self._task_manager.replace_compact_scope(session=session)
            compact_messages = [self.format_compacted_todo_message(compacted_todo)]
            replacement_messages = [*compact_messages, *preserved_messages]
            start_seq_key = self._messages[start_seq].seq
            replacement_entries = self._create_message_entries(replacement_messages, start_seq=start_seq_key)
            self._db.delete_runner_messages_by_seq_range(
                self._session_id,
                start_seq_key,
                session=session,
            )
            for entry in replacement_entries:
                self._db.insert_runner_message_entry(self._session_id, entry, session=session)
            self._phase = "running"
            self.save_metadata(status="running", session=session)
            session.commit()

        self._messages = [*messages_before_start, *replacement_entries]
        self._next_message_seq = key_after(self._messages[-1].seq if self._messages else None)
        self._cancel_event.clear()

    def find_assistant_message_seq_for_tool_call(self, tool_call_id: str) -> int:
        for seq, entry in enumerate(self._messages):
            message = entry.message
            if not isinstance(message, AssistantMessage):
                continue
            for content in message.content:
                if isinstance(content, ToolCall) and content.id == tool_call_id:
                    return seq
        raise RuntimeError(f"Could not find assistant message for tool call {tool_call_id}")

    def find_tool_result_message_seq_for_tool_call(self, tool_call_id: str) -> int:
        for seq, entry in enumerate(self._messages):
            message = entry.message
            if isinstance(message, ToolResultMessage) and message.tool_call_id == tool_call_id:
                return seq
        raise RuntimeError(f"Could not find tool result message for tool call {tool_call_id}")

    def _assistant_has_tool_calls(self, message: AssistantMessage) -> bool:
        return any(isinstance(content, ToolCall) for content in message.content)

    def _create_tool_call_records(
        self,
        assistant_message: AssistantMessage,
        tool_results: list[ToolResultMessage],
        started_at: float,
        finished_at: float,
    ) -> list[tuple[ToolCall | None, ToolResultMessage, float, float]]:
        tool_calls = {
            content.id: content
            for content in assistant_message.content
            if isinstance(content, ToolCall)
        }
        return [
            (tool_calls.get(result.tool_call_id), result, started_at, finished_at)
            for result in tool_results
        ]

    def format_compacted_todo_message(self, compacted_todo) -> AgentMessage:
        tool_refs = [
            child.tool_call_log_id
            for child in compacted_todo.children
            if child.kind == "tool_call" and child.tool_call_log_id is not None
        ]
        text = (
            f"Compacted todo: {compacted_todo.result or compacted_todo.title}\n"
            f"Useful tool calls: {tool_refs}"
        )
        return AssistantMessage(role="assistant", content=[TextContent(text=text)])

    def handle_error(self, exc: Exception) -> None:
        _log.exception("session runner failed: session=%s", self._session_id)
        self._phase = "error"
        self.save_current_data(status="error", last_error=str(exc), save_tasks=False)

    def _agent_messages(self) -> list[AgentMessage]:
        return [entry.message for entry in self._messages]

    def _create_user_message(self, user_input: str) -> AgentMessage:
        return UserMessage(
            content=[TextContent(text=user_input)],
            timestamp=int(time.time() * 1000),
        )

    def _create_message_entries(
        self,
        messages: list[AgentMessage],
        *,
        start_seq: str | None = None,
    ) -> list[RunnerMessageEntry]:
        seq = start_seq or self._next_message_seq
        if seq is None:
            raise RuntimeError("SessionRunner must be loaded before creating message entries")
        entries: list[RunnerMessageEntry] = []
        for message in messages:
            entries.append(RunnerMessageEntry(seq=seq, message=message))
            seq = key_after(seq)
        return entries
