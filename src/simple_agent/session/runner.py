"""SessionRunner owns the persisted Session.run workflow."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Literal

from pi.agent import AgentTool, AgentToolResult, AgentToolUpdateCallback
from pi.ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage
from simple_agent.fractional_index import key_after
from simple_agent.json_utils import json_safe
from simple_agent.state.state import RunnerMessageEntry
from simple_agent.token_estimation import estimate_messages_tokens
from simple_agent.tool.common_tools import create_all_coding_tools

if TYPE_CHECKING:
    from simple_agent.db.db import Database
    from simple_agent.process.agent_process import AgentProcess
    from simple_agent.task_manager import TaskManager
    from pi.agent import AgentEvent
    from pi.agent.types import AgentMessage

_log = logging.getLogger(__name__)

RunnerPhase = Literal["idle", "new_user_task", "running", "compact", "done", "error"]

SYSTEM_PROMPT = """You are a helpful coding agent.

Use create_todo before starting a coherent unit of work.
Call finish_todo when the active todo is complete.
Call error_todo if the active todo cannot be completed.
Keep responses concise and use available tools to do the work.
"""

DEFAULT_CONTEXT_TOKEN_THRESHOLD = 120_000
DEFAULT_TOOL_CALL_THRESHOLD = 200

COMPACT_SYSTEM_PROMPT = """Compact finished todos into one compacted todo.
Use only the compact tools. Create exactly one compacted todo, record useful
tool-call log IDs, then finish the compacted todo."""


def _tool_result_payload(result: AgentToolResult) -> dict[str, Any]:
    return {
        "content": [json_safe(item) for item in result.content],
        "details": json_safe(result.details),
    }


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

    def save_current_data(
        self,
        messages: list[AgentMessage] | None = None,
        *,
        status: str | None = None,
        last_error: str | None = None,
        save_tasks: bool = True,
    ) -> None:
        messages = messages or []
        entries = self._create_message_entries(messages)
        with self._db.create_session() as session:
            for entry in entries:
                self._db.insert_runner_message_entry(self._session_id, entry, session=session)
            if save_tasks:
                self._task_manager.save(session=session)
            self.save_metadata(status=status or self._phase, last_error=last_error, session=session)
            session.commit()
        self._messages.extend(entries)
        self._next_message_seq = key_after(self._messages[-1].seq if self._messages else None)

    def _create_tools(self):
        tools = [
            self._task_manager.create_create_todo_tool(),
            self._task_manager.create_finish_todo_tool(),
            self._task_manager.create_error_todo_tool(),
            *create_all_coding_tools("."),
        ]
        return self.wrap_tools(tools)

    def wrap_tool(self, tool: AgentTool) -> AgentTool:
        original = tool.execute

        async def execute(
            tool_call_id: str,
            params: dict[str, Any],
            cancel_event: asyncio.Event | None = None,
            on_update: AgentToolUpdateCallback | None = None,
        ) -> AgentToolResult:
            started_at = time.time()
            try:
                result = await original(tool_call_id, params, cancel_event, on_update)
            except Exception as exc:
                self._db.insert_runner_tool_call(
                    session_id=self._session_id,
                    tool_call_id=tool_call_id,
                    tool_name=tool.name,
                    params=params,
                    result=None,
                    status="error",
                    started_at=started_at,
                    finished_at=time.time(),
                    error=str(exc),
                )
                raise

            log_id = self._db.insert_runner_tool_call(
                session_id=self._session_id,
                tool_call_id=tool_call_id,
                tool_name=tool.name,
                params=params,
                result=_tool_result_payload(result),
                status="success",
                started_at=started_at,
                finished_at=time.time(),
                error=None,
            )
            self._task_manager.record_tool_call(log_id)
            return result

        tool.execute = execute
        return tool

    def wrap_tools(self, tools: list[AgentTool]) -> list[AgentTool]:
        return [self.wrap_tool(tool) for tool in tools]

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
        def save_current_data(event: AgentEvent) -> None:
            messages = [event.message, *event.tool_results]
            if self._phase == "new_user_task":
                self._phase = "running"
            self.save_current_data(messages, status=self._phase)
            self.pause_for_compaction_if_needed()

        hooks = {
            "turn_end": [save_current_data],
        }

        agent_user_prompt = user_input if self._phase == "new_user_task" else None
        await self._agent_process.run(
            system_prompt=SYSTEM_PROMPT,
            messages=self._agent_messages(),
            tools=self._create_tools(),
            user_prompt=agent_user_prompt,
            cancel_event=self._cancel_event,
            hooks=hooks,
        )
        if self._phase == "compact":
            return
        if self._user_paused:
            return
        if self._task_manager.active_todo_id is None:
            self._task_manager.finish_user_task()
        self._phase = "done"
        self.save_current_data(status="done")

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
            hooks={},
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
